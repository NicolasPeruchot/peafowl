# peafowl

![Platform](https://img.shields.io/badge/python-3.9-blue.svg)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>


`peafowl` is a data exploration tool focused on NLP tasks. It implements unsupervised and semi-supervised algorithms for classification.

## Installation

```
make develop
```

## Overview

- Latent Dirichlet Allocation
- Guided Dirichlet Allocation
- Document clustering
- Vocab clustering

## LDA Interface

This interface combines our custom LDA model with a representation of the dataset with pyLDAvis. [Also deployed on Streamlit](https://nicolasperuchot-peafowl-lda-cloud-jonti8.streamlit.app/)

```
make lda-app
```
