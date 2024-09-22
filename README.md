# MAVE Calibration
> Software to calibrate multiplex assays of variant effect (MAVE) data as a line of evidence for clinical variant classification

## Method Overview
Given assay scores for control variants (e.g. ClinVar P/LP and B/LB) along with those for variants from the target population (e.g. gnomAD), model the assay score distributions of ClinVar P/LP, ClinVar B/LB and gnomAD and obtain a mapping from assay score to strength of evidence in classifying a variant's pathogenicity/benignity.

![method overview](./docs/method_overview.png)

## Usage
```python
from main import singleFit

modelFit = singleFit(observations, sample_indicators)

```
### See [examples/example.py](examples/example.py)
