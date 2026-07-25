# Simulate continuous data with SymSim
library(SymSim)
library(Matrix)

# 1. Simulate True Counts
true_res <- SimulateTrueCounts(ncells_total=2000,
                               ngenes=2000,
                               nevf=20,
                               evf_type="continuous",
                               n_de_evf=12,
                               vary="s",
                               Sigma=0.4,
                               phyla=Phyla5(),
                               randseed=1
)

# 2. Add Technical Noise to get Observed Counts
simulated_gene_lengths <- sample(500:5000, size=2000, replace=TRUE)

obs_res <- True2ObservedCounts(
  true_counts = true_res[[1]],
  meta_cell = true_res[[3]],
  protocol = "UMI",
  alpha_mean = 0.05,
  alpha_sd = 0.02,
  lenslope = 0.02,
  gene_len = simulated_gene_lengths,
  depth_mean = 1e5,
  depth_sd = 3000
)
# 3. Extract the components
obs_counts <- obs_res[[1]]
counts_t <- t(obs_counts)

sparse_counts <- Matrix(counts_t, sparse = TRUE)
cell_meta <- true_res[[3]]
gene_meta <- data.frame(gene_id = paste0("Gene_", 1:ncol(counts_t)))

# 4. Save to general file
writeMM(sparse_counts, file = "symsim_counts.mtx")
write.csv(cell_meta, file = "symsim_cell_meta.csv", row.names = FALSE)
write.csv(gene_meta, file = "symsim_gene_meta.csv", row.names = FALSE)