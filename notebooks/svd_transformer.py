from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
import numpy as np

class SVDTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, svd_cumvar_threshold=0.9):
        """
        Initialize the SVDTransformer.

        Parameters:
            svd_cumvar_threshold (float): Cumulative explained variance threshold 
                                          (default: 0.9 for 90%).
        """
        self.svd_cumvar_threshold = svd_cumvar_threshold
        self.svd = None
        self.n_components_ = None

    def fit(self, X, y=None):
        """
        Fit the TruncatedSVD transformer to the input data.

        Parameters:
            X (array-like): The input feature matrix.
            y (array-like, optional): Ignored, present for compatibility.

        Returns:
            self: The fitted transformer.
        """
        # Initialize SVD with maximum components
        max_components = min(X.shape[0], X.shape[1] - 1)  # SVD limitation
        self.svd = TruncatedSVD(n_components=max_components)
        self.svd.fit(X)

        # Calculate cumulative explained variance
        cum_var = self.svd.explained_variance_ratio_.cumsum()
        self.n_components_ = np.searchsorted(cum_var, self.svd_cumvar_threshold) + 1
        
        print(f"Number of components selected: {self.n_components_}")
        print(f"Cumulative explained variance: {cum_var[self.n_components_ - 1]:.4f}")
        
        return self

    def transform(self, X):
        """
        Transform the input data using the fitted TruncatedSVD transformer.

        Parameters:
            X (array-like): The input feature matrix.

        Returns:
            X_svd (array-like): The transformed feature matrix with reduced dimensions.
        """
        if self.svd is None:
            raise ValueError("The transformer has not been fitted yet.")
        
        # Transform data and select the required number of components
        X_svd_full = self.svd.transform(X)
        return X_svd_full[:, :self.n_components_]

    def fit_transform(self, X, y=None):
        """
        Fit the transformer and transform the input data in one step.

        Parameters:
            X (array-like): The input feature matrix.
            y (array-like, optional): Ignored, present for compatibility.

        Returns:
            X_svd (array-like): The transformed feature matrix with reduced dimensions.
        """
        self.fit(X, y)
        return self.transform(X)
