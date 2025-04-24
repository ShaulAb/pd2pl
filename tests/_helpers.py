import pandas as pd
import polars as pl
import ast
from dataclasses import dataclass, asdict
from typing import Union, Any, Dict, Optional, List
import polars.selectors as cs
import numpy as np
from polars.testing import assert_frame_equal

@dataclass(slots=True)
class CompareConfig:
    check_dtype: bool = False
    check_index_type: bool = False
    rtol: float = 1e-12
    atol: float = 1e-12
    # add others as needed

DEFAULT_CFG  = CompareConfig()
INDEX_CFG   = CompareConfig(check_index_type=True)

def is_frame_equal(pdf: pd.DataFrame, pldf: pl.DataFrame, cfg: CompareConfig = DEFAULT_CFG) -> bool:
    try:
        pd.testing.assert_frame_equal(pdf, pldf, **asdict(cfg))
        return True
    except AssertionError:
        return False

def execute_code_snippet(code_string: str, namespace: dict) -> Any:
    """Executes a code snippet and returns the result of the last expression/assignment."""
    if not code_string or code_string.isspace():
        return None

    try:
        tree = ast.parse(code_string.strip())
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax in code snippet: {e}\nCode: {code_string}") from e

    if not tree.body:
        return None

    exec_context = namespace.copy()
    
    # Execute all nodes except the last one to set up context
    if len(tree.body) > 1:
        exec_nodes = tree.body[:-1]
        exec_module = ast.Module(body=exec_nodes, type_ignores=[])
        try:
            compiled_code = compile(exec_module, filename="<string>", mode="exec")
            exec(compiled_code, exec_context)
        except Exception as e:
            raise RuntimeError(f"Error executing preceding statements: {e}\nCode: {code_string}") from e

    last_node = tree.body[-1]

    # Handle the last node
    try:
        if isinstance(last_node, ast.Expr):
            # Last node is an expression, evaluate it
            eval_module = ast.Expression(body=last_node.value)
            compiled_expr = compile(eval_module, filename="<string>", mode="eval")
            result = eval(compiled_expr, exec_context)
            return result
        elif isinstance(last_node, ast.Assign):
            # Last node is an assignment, execute it and return the assigned value
            assign_module = ast.Module(body=[last_node], type_ignores=[])
            compiled_assign = compile(assign_module, filename="<string>", mode="exec")
            exec(compiled_assign, exec_context)
            # Try to get the value of the first target variable
            if last_node.targets and isinstance(last_node.targets[0], ast.Name):
                target_var = last_node.targets[0].id
                return exec_context.get(target_var)
            return None # Cannot determine assigned value easily
        else:
            # Last node is another statement (import, def, etc.), execute it
            stmt_module = ast.Module(body=[last_node], type_ignores=[])
            compiled_stmt = compile(stmt_module, filename="<string>", mode="exec")
            exec(compiled_stmt, exec_context)
            return None # Statements don't have a return value
    except Exception as e:
        raise RuntimeError(f"Error executing final statement/expression: {e}\nCode: {code_string}") from e

def compare_frames(pandas_expr: str, polars_expr: str, df: pd.DataFrame) -> bool:
    """
    Apply two expressions on a dataframe and compare the result.

    Args:
        pandas_expr: The expression to evaluate on the pandas DataFrame.
        polars_expr: The expression to evaluate on the polars DataFrame.
        df: The input dataframe to evaluate the expressions on.

    Returns:
        True if the resulting dataframes are equal, False otherwise.
    """

    if df is None:
        raise ValueError("Input DataFrame 'df' cannot be None for compare_frames")

    # Execute pandas code
    pd_ns = {'df': df.copy(), 'pd': pd, 'np': np}
    try:
        pandas_result = execute_code_snippet(pandas_expr, pd_ns)
    except Exception as e:
        print(f"Error executing pandas code: {pandas_expr}\n{e}")
        return False

    # Execute polars code
    df_pl = pl.from_pandas(df)
    polars_ns = {
        'df_pl': df_pl, 
        'pl': pl, 
        'np': np,
        'cs': cs
    }
    try:
        polars_result = execute_code_snippet(polars_expr, polars_ns)
    except Exception as e:
        print(f"Error executing polars code: {polars_expr}\n{e}")
        return False
    
    # Convert polars back to pandas
    if isinstance(polars_result, Union[pl.DataFrame, pl.Series]):
        polars_result = polars_result.to_pandas()
    
    if not isinstance(pandas_result, pd.DataFrame) or not isinstance(polars_result, pd.DataFrame):
        return "Results are not both DataFrames"
        
    # Compare shapes
    if pandas_result.shape != polars_result.shape:
        return f"Shape mismatch: Pandas shape={pandas_result.shape}, Polars shape={polars_result.shape}"

    # TODO: consider removing this check , columns should not be modified   
    if pandas_result.columns.tolist() != polars_result.columns.tolist():
        return f"Column mismatch: Pandas cols={pandas_result.columns.tolist()}, Polars cols={polars_result.columns.tolist()}"

    # Canonicalise
    pandas_result = (pandas_result
                    .reset_index(drop=True)
                    .reindex(sorted(pandas_result.columns), axis=1))
            
    polars_result = (polars_result
                    .reset_index(drop=True)
                    .reindex(sorted(polars_result.columns), axis=1))
    
    # check if the frames are equal
    return is_frame_equal(pandas_result, polars_result)

def _extract_right_on_keys(pandas_code_str: str) -> Optional[List[str]]:
    """Parse pandas code to find the value of the right_on argument."""
    try:
        tree = ast.parse(pandas_code_str.strip())
        # Assume the code is a single expression like pd.merge(...)
        if not isinstance(tree.body[0], ast.Expr) or not isinstance(tree.body[0].value, ast.Call):
            return None # Not a simple call structure

        call_node = tree.body[0].value
        # Simplistic check for pd.merge
        if not isinstance(call_node.func, ast.Attribute) or call_node.func.attr != 'merge':
             return None

        for kw in call_node.keywords:
            if kw.arg == 'right_on':
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    return [kw.value.value]
                elif isinstance(kw.value, (ast.List, ast.Tuple)):
                    keys = []
                    for elt in kw.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            keys.append(elt.value)
                        else:
                             return None # Unsupported key type in list
                    return keys
                else:
                    return None # Unsupported right_on value type
        return None # right_on not found
    except Exception:
        # Ignore parsing errors, just return None
        return None

def compare_dataframe_ops(
    pandas_code_str: str,
    polars_code_str: str,
    pandas_dfs: Dict[str, pd.DataFrame],
    atol: float = 1e-8, # Tolerance for float comparison
):
    """
    Executes pandas and polars code strings and compares the resulting DataFrames.

    Sorts results by all common columns before comparison to handle order differences.

    Args:
        pandas_code_str: The string of pandas code to execute.
        polars_code_str: The string of polars code to execute.
        pandas_dfs: Dict mapping variable names to input pandas DataFrames.
        atol: Absolute tolerance for float comparisons in assert_frame_equal.

    Raises:
        RuntimeError: If code execution fails.
        TypeError: If results are not DataFrames.
        AssertionError: If the resulting DataFrames are not equal.
        ValueError: If no common columns are found for sorting.
    """
    # Setup Pandas Environment
    pd_env = {'pd': pd}
    pd_env.update(pandas_dfs)

    # Setup Polars Environment
    pl_env = {'pl': pl}
    for name, df_pd in pandas_dfs.items():
        try:
            # Ensure consistent naming convention (df -> df_pl)
            pl_name = f"{name}_pl" if name.startswith('df') else name
            pl_env[pl_name] = pl.from_pandas(df_pd)
        except Exception as e:
            raise RuntimeError(f"Failed to convert pandas DataFrame '{name}' to Polars: {e}") from e

    # Execute Pandas Code
    try:
        pd_result = eval(pandas_code_str, pd_env)
    except Exception as e:
        raise RuntimeError(f"Pandas code execution failed:\nCode: {pandas_code_str}\nError: {e}") from e

    # Execute Polars Code
    try:
        pl_result = eval(polars_code_str, pl_env)
    except Exception as e:
        raise RuntimeError(f"Polars code execution failed:\nCode: {polars_code_str}\nError: {e}") from e

    # Verify Result Types
    if not isinstance(pd_result, pd.DataFrame):
        raise TypeError(f"Pandas code did not return a DataFrame. Got: {type(pd_result)}")
    if not isinstance(pl_result, pl.DataFrame):
        raise TypeError(f"Polars code did not return a DataFrame. Got: {type(pl_result)}")

    # --- Extract right_on keys from pandas code (for potential column dropping) ---
    right_on_keys = _extract_right_on_keys(pandas_code_str)

    # Convert Pandas Result to Polars
    try:
        pl_from_pd_result = pl.from_pandas(pd_result)
    except Exception as e:
        raise RuntimeError(f"Failed to convert pandas result DataFrame to Polars: {e}") from e

    # --- Drop Extra 'right_on' Keys from Pandas Result if Necessary ---
    if right_on_keys:
        pd_cols = set(pl_from_pd_result.columns)
        pl_cols = set(pl_result.columns)
        cols_to_drop = [
            key for key in right_on_keys
            if key in pd_cols and key not in pl_cols
        ]
        if cols_to_drop:
            try:
                pl_from_pd_result = pl_from_pd_result.drop(cols_to_drop)
            except Exception as e:
                raise RuntimeError(f"Failed to drop columns {cols_to_drop} for alignment: {e}") from e

    # --- Align Default Suffixed Columns --- 
    pl_from_pd_cols = pl_from_pd_result.columns
    pl_cols = pl_result.columns
    rename_map = {}

    for col_name in pl_from_pd_cols:
        if col_name.endswith('_x'):
            base_name = col_name[:-2]
            expected_y_col = f"{base_name}_y"
            expected_pl_right_col = f"{base_name}_right"

            # Check if pandas default suffixes were likely used and Polars equivalent exists
            if (
                expected_y_col in pl_from_pd_cols and
                base_name in pl_cols and # Polars keeps original name from left df
                expected_pl_right_col in pl_cols # Polars uses _right suffix by default
            ):
                 if base_name not in rename_map:
                     rename_map[col_name] = base_name
                     rename_map[expected_y_col] = expected_pl_right_col

    if rename_map:
        try:
            pl_from_pd_result = pl_from_pd_result.rename(rename_map)
        except Exception as e:
             raise RuntimeError(f"Failed to rename columns in pandas result for alignment: {e}") from e

    # --- Determine Sort Columns --- (Now uses potentially renamed columns)
    common_columns = sorted(list(set(pl_from_pd_result.columns) & set(pl_result.columns)))
    if not common_columns:
        # If no common columns, we can't sort meaningfully, but frames might still be "equal" if both empty
        # Or maybe they differ only by suffix column names? Check shape first.
        if pl_from_pd_result.shape == pl_result.shape and pl_from_pd_result.shape == (0, 0):
             assert_frame_equal(pl_from_pd_result, pl_result, check_dtypes=True, check_column_order=False, atol=atol)
             return
        elif pl_from_pd_result.shape != pl_result.shape:
             raise ValueError(f"Result shapes differ and no common columns found. PD: {pl_from_pd_result.shape}, PL: {pl_result.shape}")
        else:
             # Potentially different column names (e.g. due to suffixes).
             # Try a shape-only comparison as a fallback? Or assert False?
             # For now, error out if columns differ but shape is non-empty.
             raise ValueError(f"No common columns found for sorting non-empty results. PD cols: {pl_from_pd_result.columns}, PL cols: {pl_result.columns}")

    # Sort DataFrames by common columns
    try:
        sorted_pl_from_pd = pl_from_pd_result.sort(common_columns)
        sorted_pl_result = pl_result.sort(common_columns)
    except Exception as e:
        raise RuntimeError(f"Failed to sort DataFrames by columns {common_columns}: {e}") from e

    # --- Align Integer/Float Dtypes from Pandas Upcasting ---
    cols_to_cast = []
    for col_name in common_columns:
        pd_dtype = sorted_pl_from_pd[col_name].dtype
        pl_dtype = sorted_pl_result[col_name].dtype
        # Check if pandas result is float while polars is integer
        if pd_dtype == pl.Float64 and pl_dtype in pl.INTEGER_DTYPES:
            cols_to_cast.append(col_name)

    if cols_to_cast:
        try:
            sorted_pl_result = sorted_pl_result.with_columns([
                pl.col(c).cast(pl.Float64) for c in cols_to_cast
            ])
        except Exception as e:
            raise RuntimeError(f"Failed to cast Polars columns {cols_to_cast} to Float64 for comparison: {e}") from e

    # Compare Sorted Frames
    try:
        assert_frame_equal(sorted_pl_from_pd, sorted_pl_result, check_dtypes=True, check_column_order=False, atol=atol)
    except AssertionError as e:
        print("AssertionError: DataFrames do not match.")
        print("--- Pandas Code ---")
        print(pandas_code_str)
        print("--- Polars Code ---")
        print(polars_code_str)
        print("--- Pandas Result (Original) ---")
        print(pd_result)
        print("--- Polars Result ---")
        print(pl_result)
        print("--- Pandas Result (Converted to Polars, Sorted) ---")
        print(sorted_pl_from_pd)
        print("--- Polars Result (Sorted) ---")
        print(sorted_pl_result)
        raise e
