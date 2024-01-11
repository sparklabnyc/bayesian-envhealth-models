# bayesian-envhealth-models

## Project description

Bayesian models for environmental health.
This code is used for the paper Name, Y., Parks, R.M. et al. (202X). Title with hyperlink. _Journal Name_.

Aims:

- Aim 1. File `src/bayesian_envhealth_models/models.py`.

## Directory structure

```md
.
├── .gitignore
├── .pre-commit-config.yaml
├── README.md
├── data
├── src/bayesian_envhealth_models
│   └── models.py
├── output
├── scripts
│   └── model_run.py
└── reports
```

| Folder                         | File              | Description                                            |
|--------------------------------|-------------------|--------------------------------------------------------|
| `src/bayesian_envhealth_models`| `models.py`       | numpyro models                                         |
| `scripts`                      | `model_run.py`    | Script to run the numpyro model                        |

## Running the code

1. Create a fresh `conda` environment `mamba create -n envhealth-env hatch` and install the required packages `pip install -e ."[dev]"`.
2. Edit the config file `scripts/conf/model_config.yaml` to specify the data and output directories.
3. `python scripts/model_run.py`

## Data availability

Data used in the analysis are controlled by the XX who do not have permission to release data to third parties.
Individual mortality data can be requested through XX (e.g. the US CDC).
If you would like a file containing simulated data that allow you to test the code, please contact <robbie.parks@columbia.edu>.

Adapted from:

- https://num.pyro.ai/en/stable/examples/mortality.html
- https://github.com/theorashid/mortality-statsmodel/blob/master/mortality_statsmodel/car.py
