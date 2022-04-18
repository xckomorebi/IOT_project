from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from src.models.mixin import PreprocessingMixin


class CustomLogisticRegression(LogisticRegression, PreprocessingMixin):
    def __init__(self, penalty='l2', *, dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='lbfgs', max_iter=100,
                 multi_class='auto', verbose=0, warm_start=False, n_jobs=None,
                 l1_ratio=None, pca: PCA = None, sir=None):
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio
        self.pca = pca
        self.sir = sir

    def fit(self, X, y, sample_weight=None):
        X = self._fit_transform(X, y)
        return super().fit(X, y, sample_weight)

    def predict_proba(self, X):
        X = self._transform(X)
        return super().predict_proba(self, X)

    def predict_log_proba(self, X):
        X = self._transform(X)
        return super().predict_log_proba(X)

    def decision_function(self, X):
        X = self._transform(X)
        return super().decision_function(X)
