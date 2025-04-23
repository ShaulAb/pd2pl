import pandas as pd
import polars as pl
import ast
from dataclasses import dataclass, asdict
from typing import Union, Any, Dict, Optional
import polars.selectors as cs
import numpy as np

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
