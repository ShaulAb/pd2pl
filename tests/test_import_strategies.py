import pytest
from pd2pl import translate_code
from pd2pl.config import TranslationConfig

def test_auto_strategy_single_line():
    """Test auto strategy with single line of code."""
    code = "df[['A']].sort_values()"
    result = translate_code(code, import_strategy="auto")
    assert "import polars as pl" in result
    assert "df_pl" in result

def test_auto_strategy_with_imports():
    """Test auto strategy with code that already has imports."""
    code = """
import pandas as pd
df = pd.DataFrame({'A': [1, 2]})
"""
    result = translate_code(code, import_strategy="auto")
    assert "import polars as pl" in result
    assert "pl.DataFrame" in result

def test_always_strategy():
    """Test always strategy adds imports regardless of input."""
    code = "df.head()"
    result = translate_code(code, import_strategy="always")
    assert "import polars as pl" in result

def test_never_strategy():
    """Test never strategy doesn't add imports."""
    code = "df.head()"
    result = translate_code(code, import_strategy="never")
    assert "import polars as pl" not in result

def test_preserve_strategy():
    """Test preserve strategy maintains existing import structure."""
    code = """
import pandas as pd
import numpy as np
df = pd.DataFrame({'A': np.array([1, 2])})
"""
    result = translate_code(code, import_strategy="preserve")
    assert "import pandas as pd" not in result
    assert "import numpy as np" in result
    assert "import polars as pl" in result
    # Check order preservation
    lines = result.split('\n')
    np_import_idx = next(i for i, line in enumerate(lines) if "numpy" in line)
    pl_import_idx = next(i for i, line in enumerate(lines) if "polars" in line)
    assert np_import_idx < pl_import_idx

def test_invalid_strategy():
    """Test that invalid strategy raises ValueError."""
    with pytest.raises(ValueError, match="Invalid import strategy"):
        translate_code("df.head()", import_strategy="invalid")

def test_auto_strategy_complex_code():
    """Test auto strategy with more complex code patterns."""
    code = """
# Some comments
def process_data(df):
    return df.groupby('A').agg({'B': 'sum'})

result = process_data(df)
"""
    result = translate_code(code, import_strategy="auto")
    assert "import polars as pl" in result
    assert "groupby" in result

def test_preserve_strategy_with_selector_imports():
    """Test preserve strategy with code needing selector imports."""
    code = """
import pandas as pd
df = pd.DataFrame({'A': [1, 2]})
df.select_dtypes(include=['number'])
"""
    result = translate_code(code, import_strategy="preserve")
    assert "import polars.selectors as cs" in result

def test_strategy_with_format_output():
    """Test interaction between import strategy and format_output."""
    code = "df.head()"
    result = translate_code(
        code, 
        import_strategy="always",
        format_output=True
    )
    assert "import polars as pl\n" in result  
    assert "import polars.selectors as cs\n\n" in result  # Check formatting

def test_auto_strategy_docstring():
    """Test auto strategy with docstring."""
    code = '''"""
This is a module docstring.
"""
df.head()'''
    result = translate_code(code, import_strategy="auto")
    assert '"""' in result
    assert "import polars as pl" in result
    # Check that import comes after docstring
    assert result.index('"""') < result.index("import polars")

def test_preserve_strategy_multiple_imports():
    """Test preserve strategy with multiple import styles."""
    code = """
from pandas import DataFrame
import numpy as np
from typing import List
import pandas as pd

df = DataFrame({'A': np.array([1, 2])})
"""
    result = translate_code(code, import_strategy="preserve")
    # Check that non-pandas imports are preserved
    assert "from typing import List" in result
    assert "import numpy as np" in result
    # Check that pandas imports are replaced
    assert "from pandas import DataFrame" not in result
    assert "import pandas as pd" not in result
    assert "import polars as pl" in result

def test_auto_strategy_empty_code():
    """Test auto strategy with empty or whitespace code."""
    assert translate_code("", import_strategy="auto") == ""
    assert translate_code("   \n  \t  ", import_strategy="auto") == ""

def test_preserve_strategy_comments():
    """Test preserve strategy maintains comments."""
    code = """
# Import pandas
import pandas as pd
# Import numpy
import numpy as np

# Create dataframe
df = pd.DataFrame({'A': np.array([1, 2])})
"""
    result = translate_code(code, import_strategy="preserve")
    assert "# Import numpy" in result
    assert "# Create dataframe" in result
    # Original pandas comment should be preserved if we're keeping comments
    assert "# Import pandas" in result 