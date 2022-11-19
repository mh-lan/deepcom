Research tools for communication-efficient deep learning, developed by Muhang Lan.

Latest version: 0.0.3

# Installation

Install using pip

```shell
pip install deepcom
```

Check version number

```python
import deepcom as dc
print(dc.__version__)
```

# Function list

## Basic tools

### Deep learning perspective

- Convert model parameters to a numpy array: ```model2params()```

- Load a numpy array as model parameters: ```params2model()```

### Communication perspective

- Calculating mutual information: ```mutual_info()```

## Compression for training model

## Compression for post-training model

- SuRP algorithm as a sparse compression for Laplacian sequence: ```surp_algorithm()```
