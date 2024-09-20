from main import run
from mave_calibration.plotting import generate_aggregate_figs
import numpy as np
from io import StringIO
from pathlib import Path
import os

def test_run():
    scores = np.concatenate((np.random.normal(-3, 1, 100), np.random.normal(3, 1, 100), np.random.normal(-3,1,50), np.random.normal(3,1,50)))
    sample_names = ['sample1'] * 100 + ['sample2'] * 100 + ['sample3'] * 100
    buff = StringIO()
    buff.write('score,sample_name\n')
    for score,sample_name in zip(scores, sample_names):
        buff.write(f'{score},{sample_name}\n')
    buff.seek(0)
    save_path = Path("/tmp/testfits/")
    if save_path.exists():
       for root, dirs, files in os.walk(save_path,topdown=False):
            for name in files:
               Path(os.path.join(root,name)).unlink()
            for name in dirs:
                Path(os.path.join(root,name)).rmdir()
        
    save_path.mkdir(parents=True, exist_ok=True)
    fit = run(data_filepath=buff, num_fits=1, bootstrap=False, core_limit=1, verbose=False,save_path=save_path)
    if len(fit.component_params) != 2:
        raise ValueError('Expected 2 components')
    if fit.weights.shape != (3,2):
        raise ValueError('Expected 3x2 weights')
    if len(fit.likelihoods) == 0:
        raise ValueError('Expected likelihoods')
    buff.seek(0)
    
    generate_aggregate_figs.generate_figs(*list(save_path.glob("*.json")),samples_filepath=buff,save_dir=save_path / "figs",control_sample_name='sample2')