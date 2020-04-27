# Code and data from Waskom & Kiani (2018)

This repository contains data and analysis code for the following paper:

Waskom ML, Kiani R (2018). [Decision making through integration of sensory evidence at prolonged timescales](https://www.cell.com/current-biology/fulltext/S0960-9822(18)31350-2). *Current Biology 28*(23): 2350–3856.

## Data

The behavioral data are contained in two tidy tables:

- [`trial_data`](data/trial_data.csv): Trial-level data, including behavioral choice and response accuracy.
- [`pulse_data`](data/pulse_data.csv): Pulse-level data, including timing and strength of information.

These tables can be merged on a combination of the `subject`, `timing`, `session`, `run`, and `trial` identity variables.

## Modeling code

Code for fitting the models, simulating performance, and computing analytical predictions for behavioral assays can be found in the [`models`](./models) package.

A few differences in nomenclature between the code and paper should be noted. First, the term "pulse" is usually used in the code where "sample" is used in the paper". Second, two of the behavioral assays are named differently: the mPMF is the `evidence_func` in the code, and the cPMF is the `sample_func`. Finally, the leak rate parameter in the Leaky Integration model is called `alpha` in the code, because `lambda` is a reserved word in Python.

## Behavioral analyses

The [demo notebook](./demo.ipynb) demonstrates how the main behavioral assays were computed and shows how visualize the model fits with respect to each assay.

The [statistics notebook](./statistics.ipynb) contains a quantitative summary of the model fit and comparisons along with the behavioral analyses reported in the paper.

## Dependencies

The code is written for Python 3.6. A list of library versions corresponding to the paper can be found in [`requirements.txt`](./requirements.txt).

## License

These files are being released openly in the hope that they might be useful but with no promise of support. If using them leads to a publication, please cite the paper.

The dataset is released under a CC-BY 4.0 license.

The code is released under a [BSD license](./LICENSE.md).
