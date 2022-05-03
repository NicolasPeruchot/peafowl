"""Manage dependencies."""
import pathlib

from setuptools import find_packages, setup


def _read(fname: str) -> str:
    with open(pathlib.Path(fname)) as fh:
        data = fh.read()
    return data


base_packages = [
    "numpy",
    "pandas",
    "datasets",
    "pyLDAvis",
    "spacy",
    "tomotopy",
    "ipywidgets",
    "seaborn",
    "umap-learn[plot]",
    "hdbscan",
    "streamlit",
]

dev_packages = [
    "black",
    "darglint",
    "flake8",
    "flake8-bandit",
    "flake8-annotations",
    "flake8-bugbear",
    "flake8-docstrings",
    "ipykernel",
    "isort",
    "pre-commit",
]


test_packages = [
    "pytest",
]


setup(
    name="peafowl",
    version="0.0.1",
    packages=find_packages(exclude=["notebooks", "data"]),
    long_description=_read("README.md"),
    install_requires=base_packages,
    extras_require={
        "dev": dev_packages + test_packages,
        "test": test_packages,
    },
)
