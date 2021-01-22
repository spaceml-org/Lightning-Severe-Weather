from setuptools import find_packages, setup

setup(
    name="pikajoule",
    author="FDL 2020 Lightning & Extreme Weather team",
    author_email="",
    description="Use GOES GLM & ABI data for ML models",
    url="https://gitlab.com/frontierdevelopmentlab/fdl-us-2020-lightning/pikajoule",
    version="0.1.0",
    packages=find_packages(),
    package_dir={'pikajoule': 'src'},
    install_requires=[
    #    "bokeh==2.1.1",
    #    "click==7.1.2",
    #    "dask[complete]",
    #    "flake8==3.7.7",
    #    "geopandas==0.8.1",
    #    "google-cloud-storage==1.29.0",
    #    "gsutil==4.52",
    #    "jupyter==1.0.0",
    #    "jupyterlab==0.34.0",
        "matplotlib==3.3.0",
        "netCDF4==1.5.3",
        "pandas==1.0.5",
    #    "siphon==0.8.0",
    #    "sklearn==0.0",
    #    "sktime==0.4.1",
    #    "torch==1.5.1",
        "xarray==0.16.0",
    #    "lmatools @ git+https://github.com/deeplycloudy/lmatools.git",
    #    "glmtools @ git+https://github.com/fluxtransport/glmtools.git",
    ],
    license="MIT License",
)