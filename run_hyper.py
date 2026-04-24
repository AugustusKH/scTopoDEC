# run_hyper.py
import sys
import os

# Optional: add current directory to path if not installed via pip
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scTopoDEC.hyper import hyper


class Args:
    input = "CLL_RT_data.h5ad" # Path to single-cell data
    outputdir = "./results"
    transpose = False
    hypern = 20                # How many trials to run
    hyperepoch = 50            # How many epochs per trial
    ground_truth = "cell_type" # Label column name
    t = 8                      # Diffusion time for manifold ground-truth
    homology_dim = 1           # Dimension of Betti numbers (1 = loops/trajectories)


if __name__ == "__main__":
    args = Args()
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)
        
    print(f"Starting Hyperopt Suite on {args.input}...")
    
    try:
        best_config = hyper(args)
        print("\nOptimization Finished Successfully.")
    except KeyboardInterrupt:
        print("\n[!] Optimization interrupted by user.")
        print(f"Checking {args.outputdir}/hyperopt_results for partial results...")