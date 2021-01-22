# Severe Weather Prediction with GLM and a Time Series Model
by Iván Venzor-Cárdenas, Maria J. Molina, Marek Slipski1, Nadia Ahmed, Mark Cheung, Clemens Tillier, Samantha Edgington, and Gregory Renard.

Submitted to: Neural Computing and Applications.

This repository contains Python scripts and Jupyter notebooks to reproduce 
our work as well as a Colab notebook to produce figures from the manuscript.

## Abstract
Increases in flash rates detected in ground-based lightning data canbe a precursor signal to severe thunderstorm hazards, such as tornadoes andlarge hail. Lightning data from the Geostationary Lightning Mapper (GLM)onboard the GOES-16 satellite is underutilized in severe thunderstorm researchand operational forecasting. We harness the spatial and temporal advantagesof GLM data to create a machine learning (ML) model that could augmentcurrent forecasts of severe weather events. A convolutional time series MLmodel was trained to classify spring season thunderstorms with a lead time of15 minutes prior to the occurrence of large hail or tornadoes across the centralUnited States. Our results suggest that false alarms for warned thunderstormscould be decreased by 70% and that tornadoes and large hail could be correctlyidentified approximately 3 out of 4 times using only GLM data. Results alsoshow that lightning time series data are characterized by different precursorpatterns for severe and non-severe events. These results highlight the value ofGLM data and our convolutional time series ML approach, motivating furtherwork to integrate satellite-based lightning observations into an operationalforecasting framework.

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

# Reproducing results
The results can be reproducing by running the Colab notebook. 

# License
See `License`.
