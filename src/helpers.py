# Basic python
import re

# Data manipulation
import polars as pl
import pandas as pd

# Transformers
from sklearn.base import BaseEstimator, TransformerMixin

# Regex helper functions
def get_similar_variables(df_: pl.DataFrame, kword_:str)-> list:
    """
    Searches column names containing an specific keyword.

    Parameters:
    -----------

        df_: pl.DataFrame
            Target polars dataframe

        kword_: str
            Keyword used to search for similar variables.

    Returns:
    -------

        List of all matching variables names given a keyword.

    """

    vars = df_.columns
    reg = re.compile(f"[\s\S]*{kword_}[\s\S]*")

    return [reg.search(i).group() for i in vars if reg.search(i) != None ]

# count of similar variables names given a regex pattern
def count_similar_vars(df_, pattern_ = "([a-z\_]+)(_[\d]+)"):
    """
    Counts all similar variables names given a group of a regex pattern.

    Parameters:
    -----------

        df_: pl.DataFrame
            Target polars dataframe

        pattern_: str
            Regex pattern

    Returns:
    --------

        Python dictionary containing variables as keys and repetition as values

    Examples:
    --------
    Counting similar variables ignoring the numbers after variable names.


    >>> df = pl.DataFrame(
    ...     {
    ...         "id" : [1,2,3],
    ...         "creation_date": [date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 1)],
    ...         "operation_date": [date(2024, 5, 1), date(2024, 7, 1), date(2024, 9, 1)],
    ...     }
    ... )
    >>> df
    ┌─────┬───────────────┬────────────────┐
    │ id  ┆ date_1        ┆ date_2         │
    │ --- ┆ ---           ┆ ---            │
    │ i64 ┆ date          ┆ date           │
    ╞═════╪═══════════════╪════════════════╡
    │ 1   ┆ 2024-01-01    ┆ 2024-05-01     │
    │ 2   ┆ 2024-01-01    ┆ 2024-07-01     │
    │ 3   ┆ 2024-01-01    ┆ 2024-09-01     │
    └─────┴───────────────┴────────────────┘
    >>> count_similar_vars(df, "[\s\S]*(date)")
    {id: 1, date: 2}
    
    """


    common = re.compile(pattern_)
    base_name_counter = {}

    for i in df_.columns:
        try:
            base_name = common.search(i).group(1)

        except:
            base_name = i
        
        if base_name in base_name_counter.keys():
            base_name_counter[base_name] +=1

        else:
            base_name_counter[base_name] = 1

    return base_name_counter

class Mode_imputer(BaseEstimator, TransformerMixin):
    """
    Fill null values with most frequent value

    Parameters:
    -----------

        df_: pd.DataFrame
            Target pandas dataframe

        column_name_: str
            Feature name to be transformed

    Returns:
    -------

        Return Dataframe with imputed values

    """

    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self  # The fit method typically does nothing for transformers
    
    def transform(self, X):
        X_transformed = X.copy()  # Copy the input DataFrame to avoid modifying the original

        X_transformed[self.column_name] = X_transformed[self.column_name].fillna(X_transformed[self.column_name].mode()[0])
        return X_transformed


class Transformer_Date(BaseEstimator, TransformerMixin):
    """
    Transform a dataframe feature dtype

    Parameters:
    -----------

        df_: pd.DataFrame
            Target pandas dataframe

        column_name_: str
            Feature name to be transformed

    Returns:
    -------

        Return Dataframe having transformed features to datetime

    """

    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self  # The fit method typically does nothing for transformers
    
    def transform(self, X):
        X_transformed = X.copy()  # Copy the input DataFrame to avoid modifying the original

        X_transformed[self.column_name] = pd.to_datetime(X_transformed[self.column_name]) 
        return X_transformed
    
class Onehot_transformer(BaseEstimator, TransformerMixin):
    """
    Transform a dataframe categorical feature to multiple features, similar to OneHotEncoder

    Parameters:
    -----------

        df_: pd.DataFrame
            Target pandas dataframe

        column_name_: str
            Feature name to be transformed

    Returns:
    -------

        Return Dataframe with created features

    """

    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self  # The fit method typically does nothing for transformers
    
    def transform(self, X):
        X_transformed = X.copy()  # Copy the input DataFrame to avoid modifying the original

        X_transformed = pd.get_dummies(X_transformed, columns = [self.column_name], dtype = int) 
        return X_transformed