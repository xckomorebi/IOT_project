import numpy as np
import os
from sklearn.decomposition import PCA
from src.models.regression import CustomLogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification

from src.settings import DATA_PATH


params = dict(C=[0.1, 0.2, 0.3], pca=[None, PCA(n_components=100), PCA(n_components=200)])

train_data_dir = os.path.join(DATA_PATH, "dataset_archive", "train_full_feature_interp.npz")

data = np.load(train_data_dir)
X = data["X"]
y = data["y"]

lr = CustomLogisticRegression(max_iter=2000)

clf = GridSearchCV(lr, params, n_jobs=-1, cv=10, refit=True, verbose=3)
clf.fit(X, y)
clf.predict

print(clf.best_estimator_)