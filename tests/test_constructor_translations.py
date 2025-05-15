import pytest
from tests.conftest import translate_test_code

# "C": pd.Series(1, index=list(range(4)), dtype="float32"),
def test_complex_constructor_translation():
    """
    Test that pd.Categorical is translated to pl.Series(values=[...], dtype=pl.Categorical),
    and that np.array is left unchanged.
    """
    pandas_code = '''
import pandas as pd
import numpy as np

df2 = pd.DataFrame({"A": 1.0, "B": pd.Timestamp("20130102"), 
                    "D": np.array([3] * 4, dtype="int32"), 
                    "E": pd.Categorical(["test", "train", "test", "train"])}
)
'''
    polars_code = translate_test_code(pandas_code, postprocess_imports=True)

    # Check that pd.Categorical is translated to pl.Series with dtype=pl.Categorical
    assert 'pl.Series(values=[' in polars_code
    assert 'dtype=pl.Categorical' in polars_code

    # Check that the DataFrame is renamed
    assert 'df2_pl = pl.DataFrame(' in polars_code

    # Check that np.array is still present (not yet translated)
    assert "np.array([3] * 4, dtype='int32')" in polars_code

    # Check that pd.Timestamp is translated
    assert "datetime.datetime('20130102')" in polars_code 

def test_pd_timestamp_translation():
    pandas_code = '''
import pandas as pd

ts = pd.Timestamp("20130102")
ts
'''
    polars_code = translate_test_code(pandas_code, postprocess_imports=True)
    # Check that pd.Timestamp is translated to datetime.datetime
    assert 'datetime.datetime(' in polars_code
    # Check that import datetime is present
    assert 'import datetime' in polars_code 

@pytest.mark.parametrize(
    "pandas_code,expected_polars",
    [
        # Basic DataFrame with columns parameter
        (
            "pd.DataFrame(data, columns=['A', 'B', 'C'])",
            "pl.DataFrame(data, schema=['A', 'B', 'C'])"
        ),
        # DataFrame with data as dictionary and columns
        (
            "pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, columns=['A', 'B'])",
            "pl.DataFrame({'A': [1, 2], 'B': [3, 4]}, schema=['A', 'B'])"
        ),
        # DataFrame with additional parameters
        (
            "pd.DataFrame(data, columns=['A', 'B'], copy=True)",
            "pl.DataFrame(data, schema=['A', 'B'], copy=True)"
        ),
    ]
)
def test_dataframe_columns_mapping(pandas_code, expected_polars):
    """Test the translation of pandas DataFrame columns parameter to polars schema parameter."""
    assert translate_test_code(pandas_code.strip()) == expected_polars.strip() 