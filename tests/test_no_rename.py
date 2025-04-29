import pytest
from pd2pl import translate_code
from pd2pl.config import TranslationConfig

@pytest.fixture(autouse=True)
def reset_config():
    TranslationConfig.reset()
    yield
    TranslationConfig.reset()

def test_dataframe_assignment_no_rename():
    pandas_code = """
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3]})
"""
    polars_code = translate_code(pandas_code)
    assert 'df =' in polars_code
    assert 'df_pl' not in polars_code

# def test_series_assignment_no_rename():
#     pandas_code = """
# import pandas as pd
# s = pd.Series([1, 2, 3])
# """
#     polars_code = translate_code(pandas_code)
#     assert 's =' in polars_code
#     assert 's_pl' not in polars_code 