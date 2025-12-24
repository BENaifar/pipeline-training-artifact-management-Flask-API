import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
import re

class ColumnAdder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=[], column_name='', drop=True, fill_value=0):
        self.cols = cols
        self.column_name = column_name
        self.drop = drop
        self.fill_value = fill_value
        self._output_config = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        X[self.cols] = X[self.cols].fillna(self.fill_value)
        X[self.column_name] = X[self.cols[0]]
        
        for i in range(1, len(self.cols)):
            X[self.column_name] = X[self.column_name].add(X[self.cols[i]])
        
        if self.drop == True:
            X = X.drop(columns=self.cols)

        if self._output_config == "pandas":
            return X
        
        return X.to_numpy()

    def set_output(self, *, transform=None):
        """Enable set_output API compatibility."""
        self._output_config = transform
        return self
    
class TitleCreator(BaseEstimator, TransformerMixin):
    def __init__(self, col='Name'):
        self.col = col
        self._output_config = None

    def fit(self, X, y=None):
        return self

    def extract_title_df(self, X):
        titles = X[self.col].apply(
            lambda name: re.search(r",\s*([^\.]+)\.", name).group(1)
            if pd.notnull(name) and re.search(r",\s*([^\.]+)\.", name) else None
        )
        return titles.to_frame(name='Title')

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        title_df = self.extract_title_df(X)
        X = X.join(title_df)
        
        if self._output_config == "pandas":
            return X
        
        return X.to_numpy()
    

    def set_output(self, *, transform=None):
        """Enable set_output API compatibility."""
        self._output_config = transform
        return self
    
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, cols=[]):
        self.cols = cols
        self._output_config = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        X = X.drop(columns=self.cols)
        
        if self._output_config == "pandas":
            return X
        
        return X.to_numpy()

    def set_output(self, *, transform=None):
        """Enable set_output API compatibility."""
        self._output_config = transform
        return self