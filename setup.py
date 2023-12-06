import setuptools
from setuptools import setup, Extension
import numpy


setup(
    name="scoreTree",
    version="0.1.0",
    author="Blinded Authors",
    description="Building Trees for Probabilistic Prediction via Scoring Rules",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "pandas",
        "matplotlib",
        "scipy",
        "seaborn",
        "joblib",
    ],
    include_dirs=[numpy.get_include()],
)