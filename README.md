# Cross validation in Machine Learning
```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn import datasets
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, KFold, StratifiedKFold, LeaveOneOut
)
```
Let's Load an inbuilt dataset
Source: Collected at AT&T Laboratories Cambridge between April 1992 and April 1994.Variations: For each of the 40 people, 10 different pictures were taken under varying lighting conditions, with open/closed eyes, smiling/non-smiling expressions, and with/without glasses.Common Use Cases: Frequently used for image recognition, dimensionality reduction (like PCA), clustering, and face completion tasks.
```python
X,y = datasets.fetch_olivetti_faces(return_X_y=True)
```
⚡️return_X_y=True makes the loader return just (X, y) as a tuple instead of a full Bunch object with extra metadata (description, image arrays, etc.).
⚡️Each face is really a 64×64 grid of pixel brightness values — 4096 numbers arranged in rows and columns. But X stores each face as one long row of 4096 numbers in a row, X[i], with no row/column structure.
