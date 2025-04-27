import re
from typing import List

def _has_import(code_lines: List[str], import_stmt: str) -> bool:
    return any(line.strip() == import_stmt for line in code_lines)

def _find_import_insertion_index(code_lines: List[str]) -> int:
    """
    Returns the index after the module docstring (if present),
    or after initial comments, or 0 if neither.
    """
    idx = 0
    n = len(code_lines)
    # Skip initial comments
    while idx < n and code_lines[idx].strip().startswith('#'):
        idx += 1
    # Check for module docstring
    if idx < n and re.match(r'\s*[\'\"]{3}', code_lines[idx]):
        docstring_line = code_lines[idx].strip()
        # Single-line docstring
        if docstring_line.count('"""') == 2 or docstring_line.count("'''") == 2:
            idx += 1
        else:
            # Multi-line docstring
            idx += 1
            while idx < n and not re.match(r'.*[\'\"]{3}', code_lines[idx]):
                idx += 1
            if idx < n:
                idx += 1  # Skip closing docstring line
    return idx

def _deduplicate_imports(code_lines: List[str]) -> List[str]:
    seen = set()
    result = []
    for line in code_lines:
        if line.strip().startswith('import') or line.strip().startswith('from'):
            if line.strip() not in seen:
                seen.add(line.strip())
                result.append(line)
        else:
            result.append(line)
    return result

def _remove_pandas_import(code_lines: List[str]) -> List[str]:
    return [line for line in code_lines if not re.match(r'\s*import\s+pandas\s+as\s+pd', line)]

def _format_with_black(code: str) -> str:
    try:
        import black
        return black.format_str(code, mode=black.Mode())
    except ImportError:
        # Black not installed; return code unchanged
        return code
    except Exception:
        # Any formatting error, return code unchanged
        return code

def process_imports(
    code: str,
    add_polars_imports: bool = True,
    deduplicate_imports: bool = True,
    remove_pandas_import: bool = False,
    format_with_black: bool = False
) -> str:
    """
    Post-processes code to manage import statements.
    - Adds 'import polars as pl' and 'import polars.selectors as cs' if missing (if add_polars_imports is True)
    - Deduplicates import statements (if deduplicate_imports is True)
    - Removes 'import pandas as pd' (if remove_pandas_import is True)
    - Ensures a blank line after imports for readability
    - Optionally formats code with Black (if format_with_black is True)
    """
    code_lines = code.splitlines()
    imports_to_add = []
    if add_polars_imports:
        if not _has_import(code_lines, 'import polars as pl'):
            imports_to_add.append('import polars as pl')
        if not _has_import(code_lines, 'import polars.selectors as cs'):
            imports_to_add.append('import polars.selectors as cs')
    if imports_to_add:
        insert_idx = _find_import_insertion_index(code_lines)
        code_lines = code_lines[:insert_idx] + imports_to_add + code_lines[insert_idx:]
    if remove_pandas_import:
        code_lines = _remove_pandas_import(code_lines)
    if deduplicate_imports:
        code_lines = _deduplicate_imports(code_lines)
    # Ensure a single blank line after the last import (if any imports present)
    last_import_idx = -1
    for idx, line in enumerate(code_lines):
        if line.strip().startswith('import') or line.strip().startswith('from'):
            last_import_idx = idx
    if last_import_idx != -1 and last_import_idx + 1 < len(code_lines):
        # Only add a blank line if not already present
        if code_lines[last_import_idx + 1].strip() != '':
            code_lines.insert(last_import_idx + 1, '')
    result = '\n'.join(code_lines)
    if format_with_black:
        result = _format_with_black(result)
    # Ensure import datetime if datetime.datetime is used
    if 'datetime.datetime' in code and 'import datetime' not in code:
        result = 'import datetime\n' + result
    return result 