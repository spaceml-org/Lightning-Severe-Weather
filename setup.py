from setuptools import find_packages, setup

setup(
    name="pikajoule",
    author="FDL 2020 Lightning & Extreme Weather team",
    author_email="",
    description="Timeseries classification of severe weather events using GLM data.",
    url="https://github.com/spaceml-org/Lightning-Severe-Weather",
    version="0.1.0",
    packages=find_packages(),
    package_dir={'': 'src'},
    install_requires=[
        "gcsfs==0.8.0",
        "jupyter==1.0.0",
        "matplotlib==3.3.0",
        "netCDF4==1.5.3",
        "pandas==1.0.5",
        "xarray==0.16.0",
    ],
    license="MIT License",
)
