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

if __name__ == "__main__":
    args = Args()
    best_config = hyper(args)
    # The return value from hyper() arrives here as 'best_config'