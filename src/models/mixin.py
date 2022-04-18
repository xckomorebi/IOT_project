import numpy as np


class PreprocessingMixin:
    def _fit_transform(self, X, y):
        if self.pca:
            X = self.pca.fit_transform(X)
            covm = np.cov(X)
            self.covm = covm
        
        if self.sir:
            X = self.sir.fit_transform(X)

        return X

    def _transform(self, X):
        if self.pca:
            X = self.pca.transform(X)
        
        if self.sir:
            X = self.sir.fit_transform(X)
        
        return X