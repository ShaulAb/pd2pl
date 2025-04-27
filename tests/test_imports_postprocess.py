import pytest
from pd2pl.imports_postprocess import process_imports

def test_add_polars_imports_to_empty_code():
    code = "print('hello world')"
    result = process_imports(code)
    assert 'import polars as pl' in result
    assert 'import polars.selectors as cs' in result
    assert result.index('import polars as pl') < result.index('print')

def test_no_duplicate_polars_imports():
    code = "import polars as pl\nimport polars.selectors as cs\nprint('hi')"
    result = process_imports(code)
    assert result.count('import polars as pl') == 1
    assert result.count('import polars.selectors as cs') == 1

def test_deduplicate_imports():
    code = "import polars as pl\nimport polars as pl\nimport polars.selectors as cs\nimport polars.selectors as cs\nprint('hi')"
    result = process_imports(code)
    assert result.count('import polars as pl') == 1
    assert result.count('import polars.selectors as cs') == 1

def test_remove_pandas_import():
    code = "import pandas as pd\nimport polars as pl\nprint('hi')"
    result = process_imports(code, remove_pandas_import=True)
    assert 'import pandas as pd' not in result
    assert 'import polars as pl' in result

def test_keep_pandas_import():
    code = "import pandas as pd\nprint('hi')"
    result = process_imports(code, remove_pandas_import=False)
    assert 'import pandas as pd' in result

def test_insert_after_comments():
    code = "# This is a comment\n# Another comment\nprint('hi')"
    result = process_imports(code)
    assert result.splitlines()[2].startswith('import polars as pl') or result.splitlines()[3].startswith('import polars as pl')

def test_insert_after_docstring():
    code = '"""Module docstring"""\nprint("hi")'
    result = process_imports(code)
    # Should insert after docstring
    assert 'import polars as pl' in result
    assert result.index('import polars as pl') > result.index('"""Module docstring"""')

def test_idempotency():
    code = "import polars as pl\nimport polars.selectors as cs\nprint('hi')"
    once = process_imports(code)
    twice = process_imports(once)
    assert once == twice

def test_combined_options():
    code = "import pandas as pd\nimport polars as pl\nimport polars as pl\nprint('hi')"
    result = process_imports(code, remove_pandas_import=True, deduplicate_imports=True)
    assert 'import pandas as pd' not in result
    assert result.count('import polars as pl') == 1 