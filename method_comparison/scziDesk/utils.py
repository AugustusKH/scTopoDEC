import collections
import functools
import json
import logging
import operator
from typing import Dict, List, Optional, Tuple, Union

import igraph  # Standardized from legacy jgraph
import numpy as np
import scipy.sparse
import tqdm

logger = logging.getLogger(__name__)


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def in_ipynb() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True   # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other interpreter type
    except NameError:
        return False      # Standard Python interpreter


def smart_tqdm():
    if in_ipynb():
        return tqdm.tqdm_notebook
    return tqdm.tqdm


def with_self_graph(fn):
    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        with self.graph.as_default():
            return fn(self, *args, **kwargs)
    return wrapped


def minibatch(batch_size: int, desc: str, use_last: bool = False, progress_bar: bool = True):
    """Wraps a batch function into a minibatch processing version."""
    def minibatch_wrapper(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            total_size = args[0].shape[0]
            if use_last:
                n_batch = int(np.ceil(total_size / float(batch_size)))
            else:
                n_batch = max(1, int(np.floor(total_size / float(batch_size))))
                
            for batch_idx in smart_tqdm()(
                range(n_batch), desc=desc, unit="batches",
                leave=False, disable=not progress_bar
            ):
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, total_size)
                this_args = (item[start:end] for item in args)
                func(*this_args, **kwargs)
        return wrapped_func
    return minibatch_wrapper


def encode_integer(label: Union[List, np.ndarray], sort: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    label = np.array(label).ravel()
    classes = np.unique(label)
    if sort:
        classes.sort()
    mapping = {v: i for i, v in enumerate(classes)}
    return np.array([mapping[v] for v in label]), classes


def encode_onehot(label: Union[List, np.ndarray], sort: bool = False, ignore: Optional[List] = None) -> scipy.sparse.csr_matrix:
    i, c = encode_integer(label, sort)
    onehot = scipy.sparse.csc_matrix((
        np.ones_like(i, dtype=np.int32), (np.arange(i.size), i)
    ))
    if ignore is None:
        ignore = []
    return onehot[:, ~np.in1d(c, ignore)].tocsr()


class CellTypeDAG(object):

    def __init__(self, graph: Optional[igraph.Graph] = None, vdict: Optional[Dict] = None):
        self.graph = igraph.Graph(directed=True) if graph is None else graph
        self.vdict = {} if vdict is None else vdict

    @classmethod
    def load(cls, file: str) -> "CellTypeDAG":
        if file.endswith(".json"):
            return cls.load_json(file)
        elif file.endswith(".obo"):
            return cls.load_obo(file)
        else:
            raise ValueError("Unexpected file format!")

    @classmethod
    def load_json(cls, file: str) -> "CellTypeDAG":
        with open(file, "r") as f:
            d = json.load(f)
        dag = cls()
        dag._build_tree(d)
        return dag

    @classmethod
    def load_obo(cls, file: str) -> "CellTypeDAG":
        """Builds directed graph matching modern Pronto 2.0+ syntax semantics."""
        import pronto
        ont = pronto.Ontology(file)
        graph, vdict = igraph.Graph(directed=True), {}
        
        # In pronto 2.0+, terms are explicitly tracked in .terms()
        for term_id in ont.terms():
            item = ont[term_id]
            if not item.id.startswith("CL"):
                continue
            if item.obsolete:
                continue
                
            # Definition strings replace old .desc definitions
            desc_str = str(item.definition) if item.definition else ""
            synonyms_list = [f"{s.description} ({s.scope})" for s in item.synonyms]
            
            graph.add_vertex(
                name=item.id, 
                cell_ontology_class=item.name,
                desc=desc_str, 
                synonyms=synonyms_list
            )
            
            assert item.id not in vdict
            vdict[item.id] = item.id
            if item.name:
                vdict[item.name] = item.id
            for syn in item.synonyms:
                if syn.scope == "EXACT" and syn.description != item.name:
                    vdict[syn.description] = item.id
                    
        for source in graph.vs:
            term_item = ont[source["name"]]
            # Track superclasses explicitly via hierarchical is_a links
            for target_term in term_item.superclasses(distance=1, relationships={"is_a"}):
                if target_term.id == term_item.id:
                    continue
                if not target_term.id.startswith("CL"):
                    continue
                try:
                    target_vertex = graph.vs.find(name=target_term.id.split()[0])
                    graph.add_edge(source["name"], target_vertex["name"])
                except ValueError:
                    continue  # Target missing from pruned sub-graph configurations
                    
        return cls(graph, vdict)

    def _build_tree(self, d: Dict, parent=None):
        self.graph.add_vertex(name=d["name"])
        v = self.graph.vs.find(d["name"])
        if parent is not None:
            self.graph.add_edge(v, parent)
        self.vdict[d["name"]] = d["name"]
        if "alias" in d:
            for alias in d["alias"]:
                self.vdict[alias] = d["name"]
        if "children" in d:
            for subd in d["children"]:
                self._build_tree(subd, v)

    def get_vertex(self, name: str) -> igraph.Vertex:
        return self.graph.vs.find(self.vdict[name])

    def is_related(self, name1: str, name2: str) -> bool:
        return self.is_descendant_of(name1, name2) or self.is_ancestor_of(name1, name2)

    def is_descendant_of(self, name1: str, name2: str) -> bool:
        if name1 not in self.vdict or name2 not in self.vdict:
            return False
        # Calculate single element shortest path matrix arrays
        shortest_path = self.graph.distances(
            source=self.get_vertex(name1), target=self.get_vertex(name2)
        )[0][0]
        return np.isfinite(shortest_path)

    def is_ancestor_of(self, name1: str, name2: str) -> bool:
        if name1 not in self.vdict or name2 not in self.vdict:
            return False
        shortest_path = self.graph.distances(
            source=self.get_vertex(name2), target=self.get_vertex(name1)
        )[0][0]
        return np.isfinite(shortest_path)

    def conditional_prob(self, name1: str, name2: str) -> float:
        """Computes topological conditional probability p(name1 | name2)."""
        if name1 not in self.vdict or name2 not in self.vdict:
            return 0.0
            
        self.graph.vs["prob"] = 0.0
        v2_vertex = self.get_vertex(name2)
        v1_vertex = self.get_vertex(name1)
        
        # bfs iterations tracking modes standardized to explicit strings
        v2_parents = [self.graph.vs[idx] for idx in self.graph.bfs(v2_vertex, mode="out")[0]]
        v1_parents = [self.graph.vs[idx] for idx in self.graph.bfs(v1_vertex, mode="out")[0]]
        
        for v in v2_parents:
            v["prob"] = 1.0
            
        while True:
            changed = False
            for v1_parent in v1_parents[::-1]:
                if v1_parent["prob"] != 0.0:
                    continue
                neighbors_out = v1_parent.neighbors(mode="out")
                if len(neighbors_out) > 0:
                    v1_parent["prob"] = float(np.prod([
                        v["prob"] / len(v.neighbors(mode="in")) for v in neighbors_out
                    ]))
                if v1_parent["prob"] != 0.0:
                    changed = True
            if not changed:
                break
                
        return float(self.get_vertex(name1)["prob"])

    def similarity(self, name1: str, name2: str, method: str = "probability") -> float:
        if method == "probability":
            return (self.conditional_prob(name1, name2) + self.conditional_prob(name2, name1)) / 2.0
        raise ValueError("Invalid method!")

    def count_reset(self):
        self.graph.vs["raw_count"] = 0
        self.graph.vs["prop_count"] = 0
        self.graph.vs["count"] = 0

    def count_set(self, name: str, count: int):
        self.get_vertex(name)["raw_count"] = count

    def count_update(self):
        origins = [v for v in self.graph.vs if v["raw_count"] > 0]
        for origin in origins:
            bfs_indices = self.graph.bfs(origin, mode="out")[0]
            for idx in bfs_indices:
                v = self.graph.vs[idx]
                if v != origin:
                    v["prop_count"] += origin["raw_count"]
                    
        self.graph.vs["count"] = list(map(
            operator.add, self.graph.vs["raw_count"], self.graph.vs["prop_count"]
        ))

    def best_leaves(self, thresh: float, retrieve: str = "name") -> List:
        valid_indices = [v.index for v in self.graph.vs if v["count"] >= thresh]
        subgraph = self.graph.subgraph(valid_indices)
        leaves, max_count = [], 0
        
        for leaf in subgraph.vs:
            if leaf.indegree() == 0:  # Isolates root/leaf terminal boundaries
                if leaf["count"] > max_count:
                    max_count = leaf["count"]
                    leaves = [leaf[retrieve]]
                elif leaf["count"] == max_count:
                    leaves.append(leaf[retrieve])
        return leaves


class DataDict(collections.OrderedDict):

    def shuffle(self, random_state=np.random):
        shuffled = DataDict()
        shuffle_idx = None
        for item in self:
            shuffle_idx = random_state.permutation(self[item].shape[0]) \
                if shuffle_idx is None else shuffle_idx
            shuffled[item] = self[item][shuffle_idx]
        return shuffled

    @property
    def size(self) -> int:
        data_size = set([item.shape[0] for item in self.values()])
        assert len(data_size) == 1
        return data_size.pop()

    @property
    def shape(self) -> List[int]:
        return [self.size]

    def __getitem__(self, fetch):
        if isinstance(fetch, (slice, np.ndarray)):
            return DataDict([
                (item, self[item][fetch]) for item in self
            ])
        return super(DataDict, self).__getitem__(fetch)


def densify(arr: Union[np.ndarray, scipy.sparse.spmatrix]) -> np.ndarray:
    if scipy.sparse.issparse(arr):
        return arr.toarray()
    return arr


def empty_safe(fn, dtype):
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)
    return _fn


decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8") if hasattr(_x, "decode") else str(_x)), str)
encode = empty_safe(np.vectorize(lambda _x: str(_x).encode("utf-8")), "S")
upper = empty_safe(np.vectorize(lambda x: str(x).upper()), str)
lower = empty_safe(np.vectorize(lambda x: str(x).lower()), str)
tostr = empty_safe(np.vectorize(str), str)