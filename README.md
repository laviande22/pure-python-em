# Development Environment

- Ubuntu 16.04
- Python3.6

# Requirement

- No requirements

# Usage

## Data Format

- Data should consist of 4 bytes of floating numbers in binary format.
- No delimiters are allowed. For this reason, you should know the dimension of a vecter and the number of entries in the dataset.

## How to Run

1. Change the working directory to this project's home directory.
2. Run `python` or `ipython` on your terminal.
3. Import the module and create an instance of `Cluster`(Be careful with capital letters) class.
4. Call fit() method to fit the model to the data. `epsilon` for the threshold of loss is an optional parameter, which is for an early stopping.
5. You can check the member vectors of a cluster calling `cluster_vectors(cluster)` method.
6. You can check the resulting centroids of clusters by checking `centroids` attribute.
7. Every cluster can be checked by checking `labels` attribute.

```
from clustering import Cluster

cls = Cluster('floats.data', 32, 25000)
cls.fit(epsilon=0.05)
# cls.fit()    # Without the optional parameter epsilon.

cls.cluster_vectors(3)    # Check the vectors of cluster index 3.
print(cls.centroids)      # Check the centroids.
print(cls.labels)
```
