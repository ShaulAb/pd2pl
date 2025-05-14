import pytest
from pd2pl.imports_postprocess import process_imports, ImportStrategy

def test_always_strategy_adds_polars_and_removes_pandas():
    code = "import pandas as pd\nprint('hi')"
    result = process_imports(
        code,
        import_strategy=ImportStrategy.ALWAYS,
        needs_polars_import=True,
        needs_selector_import=True
    )
    assert 'import polars as pl' in result
    assert 'import polars.selectors as cs' in result
    assert 'import pandas as pd' not in result

def test_always_strategy_no_imports_adds_both():
    code = "print('hi')"
    result = process_imports(
        code,
        import_strategy=ImportStrategy.ALWAYS,
        needs_polars_import=True,
        needs_selector_import=True
    )
    assert 'import polars as pl' in result
    assert 'import polars.selectors as cs' in result

def test_always_strategy_idempotency():
    code = "import polars as pl\nimport polars.selectors as cs\nprint('hi')"
    once = process_imports(
        code,
        import_strategy=ImportStrategy.ALWAYS,
        needs_polars_import=True,
        needs_selector_import=True
    )
    twice = process_imports(
        once,
        import_strategy=ImportStrategy.ALWAYS,
        needs_polars_import=True,
        needs_selector_import=True
    )
    assert once == twice

def test_never_strategy_preserves_all():
    code = "import pandas as pd\nimport polars as pl\nprint('hi')"
    result = process_imports(
        code,
        import_strategy=ImportStrategy.NEVER,
        needs_polars_import=True,
        needs_selector_import=True
    )
    assert 'import pandas as pd' in result
    assert 'import polars as pl' in result
    assert result.count('import polars as pl') == 1

def test_auto_strategy_adds_only_if_needed():
    code = "print('hi')"
    # Should not add imports if flags are False
    result = process_imports(
        code,
        import_strategy=ImportStrategy.AUTO,
        needs_polars_import=False,
        needs_selector_import=False
    )
    assert 'import polars as pl' not in result
    # Should add if flags are True
    result2 = process_imports(
        code,
        import_strategy=ImportStrategy.AUTO,
        needs_polars_import=True,
        needs_selector_import=True
    )
    assert 'import polars as pl' in result2
    assert 'import polars.selectors as cs' in result2

def test_auto_strategy_real_usecase():
    code = "import pandas as pd\nimport numpy as np\n\nd = {'col1': [1, 2], 'col2': [3, 4]}\ndf = pd.DataFrame(data=d)"
    # Should not add imports if flags are False
    result = process_imports(
        code,
        import_strategy=ImportStrategy.AUTO,
    )
    assert 'import polars as pl' not in result

def test_preserve_strategy_replaces_pandas_only_if_needed():
    code = "import pandas as pd\nprint('hi')"
    # Should preserve pandas if not needed
    result = process_imports(
        code,
        import_strategy=ImportStrategy.PRESERVE,
        needs_polars_import=False,
        needs_selector_import=False
    )
    assert 'import pandas as pd' in result
    # Should preserve pandas and add polars if polars is needed
    result2 = process_imports(
        code,
        import_strategy=ImportStrategy.PRESERVE,
        needs_polars_import=True,
        needs_selector_import=False
    )
    assert 'import pandas as pd' in result2
    assert 'import polars as pl' in result2

def test_preserve_strategy_preserves_comments_and_docstrings():
    code = '"""Docstring"""\n# Comment\nimport pandas as pd\nprint("hi")'
    result = process_imports(
        code,
        import_strategy=ImportStrategy.PRESERVE,
        needs_polars_import=False
    )
    assert '"""Docstring"""' in result
    assert '# Comment' in result
    assert 'import pandas as pd' in result

def test_strategy_unrelated_imports_preserved():
    code = "import numpy as np\nimport pandas as pd\nprint('hi')"
    result = process_imports(
        code,
        import_strategy=ImportStrategy.ALWAYS,
        needs_polars_import=True
    )
    assert 'import numpy as np' in result
    assert 'import polars as pl' in result
    assert 'import pandas as pd' not in result

def test_strategy_blank_lines_and_order():
    code = "import pandas as pd\n\nprint('hi')"
    result = process_imports(
        code,
        import_strategy=ImportStrategy.ALWAYS,
        needs_polars_import=True
    )
    # Should have a blank line after imports
    lines = result.splitlines()
    import_lines = [i for i, l in enumerate(lines) if l.startswith('import')]
    if import_lines:
        last_import = import_lines[-1]
        assert last_import + 1 < len(lines)
        assert lines[last_import + 1].strip() == '' 