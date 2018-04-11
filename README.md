# Machine Learning Reports

This repository provides python script that automatically generate reports for your machine learning predictions tasks


## Binary Classification Report

Here is an example of an output of the following code : 

```
# Imports
import numpy as np
from binary_classification_report import binary_classification_report
%matplotlib inline

# Generate synthetic data
n = 10000
p = np.random.rand(n)
Y_pred = np.random.rand(n)**p
Y_true = p < 0.6

# Plot
binary_classification_report(Y_true, Y_pred)
```
![alt text](https://github.com/SimJeg/ml_reports/blob/master/binary_classification_output.png)
