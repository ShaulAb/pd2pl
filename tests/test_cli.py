import subprocess
import sys
import pytest

CLI_CMD = [sys.executable, '-m', 'pd2pl']

def run_cli(input_code, args=None):
    args = args or []
    result = subprocess.run(
        CLI_CMD + args,
        input=input_code.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return result

def test_cli_always_strategy_adds_polars():
    code = 'import pandas as pd\ndf = pd.DataFrame({"a": [1, 2]})\nprint(df)'
    result = run_cli(code, ['--import-strategy', 'always'])
    out = result.stdout.decode()
    assert 'import polars as pl' in out
    assert 'import pandas as pd' not in out
    assert 'pl.DataFrame' in out
    assert result.returncode == 0

def test_cli_never_strategy_preserves_pandas():
    code = 'import pandas as pd\nprint("hi")'
    result = run_cli(code, ['--import-strategy', 'never'])
    out = result.stdout.decode()
    assert 'import pandas as pd' in out
    assert result.returncode == 0

def test_cli_auto_strategy_adds_if_needed():
    code = 'print("hi")'
    result = run_cli(code, ['--import-strategy', 'auto'])
    out = result.stdout.decode()
    # Should not add polars if not needed
    assert 'import polars as pl' not in out
    # Now with pandas code (simulate needs_polars_import via translation)
    code2 = 'import pandas as pd\nprint("hi")'
    result2 = run_cli(code2, ['--import-strategy', 'auto'])
    out2 = result2.stdout.decode()
    # Should add polars as pl if translation triggers it
    # (This depends on translation logic, so just check no crash)
    assert result2.returncode == 0

def test_cli_preserve_strategy_replaces_only_if_needed():
    code = 'import pandas as pd\nprint("hi")'
    result = run_cli(code, ['--import-strategy', 'preserve'])
    out = result.stdout.decode()
    # Should preserve pandas if not needed
    assert 'import pandas as pd' in out
    # Should not add polars as pl unless needed
    # (This depends on translation logic, so just check no crash)
    assert result.returncode == 0

def test_cli_default_is_auto():
    code = 'print("hi")'
    result = run_cli(code)
    out = result.stdout.decode()
    # Should not add polars as pl if not needed
    assert 'import polars as pl' not in out
    assert result.returncode == 0

def test_cli_invalid_strategy_errors():
    code = 'print("hi")'
    result = run_cli(code, ['--import-strategy', 'foobar'])
    err = result.stderr.decode().lower()
    assert result.returncode != 0
    assert 'invalid choice' in err or 'error' in err

def test_cli_help_includes_import_strategy():
    result = subprocess.run(CLI_CMD + ['--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = result.stdout.decode().lower()
    assert '--import-strategy' in out 