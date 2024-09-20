from main import run
from mave_calibration.plotting.generate_aggregate_figs import generate_figs
import os
from pathlib import Path

if __name__ == '__main__':
    # location of this file
    pth = os.path.dirname(os.path.abspath(__file__))
    # dataset name
    dataset_name = "Findlay_BRCA1_SGE"
    # dataset directory
    dataset_dir = Path(os.path.join(pth,dataset_name))
    # run the calibration
    best_fit = run(data_filepath=dataset_dir / "samples.csv",
                   max_iters=10000, n_inits=100, verbose=True,
                   save_path=dataset_dir / "fits")
    # plot the results
    generate_figs(*list((dataset_dir / "fits").glob("*.json")),
                    samples_filepath=dataset_dir / "samples.csv",
                    save_dir=dataset_dir,
                    control_sample_name='synonymous')