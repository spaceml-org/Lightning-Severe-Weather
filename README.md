# Severe Weather Prediction with GLM and a Time Series Model
by Iván Venzor-Cárdenas, Maria J. Molina, Marek Slipski, Nadia Ahmed, Mark Cheung, Clemens Tillier, Samantha Edgington, and Gregory Renard.

Submitted to: Neural Computing and Applications.

This repository contains Python scripts and Jupyter notebooks to reproduce 
our work as well as a Colab notebook to produce figures from the manuscript.

## Abstract
Increases in flash rates detected in ground-based lightning data can be a precursor signal to severe thunderstorm hazards, such as tornadoes and large hail. Lightning data from the Geostationary Lightning Mapper (GLM) onboard the GOES-16 satellite is underutilized in severe thunderstorm research and operational forecasting. We harness the spatial and temporal advantages of GLM data to create a machine learning (ML) model that could augment current forecasts of severe weather events. A convolutional time series ML model was trained to classify spring season thunderstorms with a lead time of 15 minutes prior to the occurrence of large hail or tornadoes across the central United States. Our results suggest that false alarms for warned thunderstorms could be decreased by 70% and that tornadoes and large hail could be correctly identified approximately 3 out of 4 times using only GLM data. Results also show that lightning time series data are characterized by different precursor patterns for severe and non-severe events. These results highlight the value of GLM data and our convolutional time series ML approach, motivating further work to integrate satellite-based lightning observations into an operational forecasting framework.

# Setup
Calculations and figures are run in Jupyter notebooks in `notebooks`.

## Getting the code
git clone https://github.com/spaceml-org/Lightning-Severe-Weather.git

## Dependencies
Dependencies are given in `setup.py`. We recommend setting up a virtual enivronment
and installing the package in develop mode:
```
pip install -e .
```

## Data
Geostationary Lightning Mapper L2 data is available...
We created gridded GLM data using the `glmtools` package available here: https://github.com/deeplycloudy/glmtools. We created full CONUS grids using the following commands:...

The grids used for this work can be found here:...
Those are available here:...


# Reproducing results
The results can be reproducing by running the Colab notebook. 

We created timeseries of quantites from gridded GLM data (`src/generate_timeseries.py`) for tornado, hail, and null events. Those timeseries serve as the input into ROCKET (`src/...`).

# License
See `License`.
