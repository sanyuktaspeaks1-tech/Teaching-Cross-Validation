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
```python
# peek at a few faces, one at a time
for i in range(5):
    plt.imshow(X[i].reshape(64, 64), cmap="gray")
    plt.title(f"Photo {i} (person {y[i]})")
    plt.show()
```
## NOTE: the raw data is ORDERED by person (person 0's 10 photos first, then person 1's, etc). That matters later.
 ### STEP 2: Train/test split (stratified so every person appears in both sets)
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("X_train:", X_train.shape, " X_test:", X_test.shape)
```
### STEP 3: Pick two classifiers with arbitrary hyperparameters
```python
clf_svc = SVC(kernel="rbf", gamma=0.1, C=10.0, random_state=42)
clf_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
```
### STEP 4: The WRONG way: evaluate on the training set
```python
clf_svc.fit(X_train, y_train)
clf_rf.fit(X_train, y_train)
```
```python
y_pred_svc = clf_svc.predict(X_train)
print(f"SVC TRAIN accuracy: {accuracy_score(y_train, y_pred_svc):.2f}")

y_pred_rf = clf_rf.predict(X_train)
print(f"RF TRAIN accuracy: {accuracy_score(y_train, y_pred_rf):.2f}")
```
A model scoring ~1.00 on the data it was trained on isn't proof it learned — it may have just memorized those exact examples. That's why training accuracy is never trusted for model selection; only accuracy on unseen data (validation/test/CV) tells you if the model actually generalizes.

### Instead of testing on training data (Step 4), we carve out a chunk of the TRAINING set that the model will never train on. This chunk
### is called the "validation set". We take the last 20% of X_train, y_train and set it aside purely for evaluation.
### We are NOT touching X_test/y_test here -- that stays locked away until Step 10, once model selection is completely finished.

```python
n_val = int(0.2 * len(X_train))           # how many samples = 20%?
n_train_only = len(X_train) - n_val       # the rest stay for training
 
X_tr = X_train[:n_train_only]   # first 80% -> train on this
X_val = X_train[n_train_only:]  # last 20%  -> only used to evaluate
 
y_tr = y_train[:n_train_only]
y_val = y_train[n_train_only:]
 
print("X_tr (train):", X_tr.shape, " X_val (validation):", X_val.shape)
 
# Retrain each model, but now ONLY on X_tr (not the full X_train)
clf_svc.fit(X_tr, y_tr)
clf_rf.fit(X_tr, y_tr)
 
# Predict on X_val, which the models have never seen
y_pred_svc = clf_svc.predict(X_val)
y_pred_rf = clf_rf.predict(X_val)
 
print(f"SVC VAL accuracy: {accuracy_score(y_val, y_pred_svc):.2f}")
print(f"RF  VAL accuracy: {accuracy_score(y_val, y_pred_rf):.2f}")
```
### Compare this to Step 4's ~1.00 scores. This is much more honest, because X_val was never used for training.

### CAVEAT: we only tried ONE particular 80/20 slice. If we'd sliced the data differently, we might get different numbers. That instability is exactly what Step 6 fixes.
 
 
### STEP 6: K-fold cross-validation
### Idea: instead of one validation slice, make several. Split the data into K folds. Repeat K times: train on K-1 folds, test on the fold left out. Every sample gets tested exactly once, and weaverage the K scores into one overall estimate.
