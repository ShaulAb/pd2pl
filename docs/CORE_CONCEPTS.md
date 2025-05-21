# Core Concepts of the pd2pl Translator

This document explains the fundamental concepts and mechanisms that power the Pandas-to-Polars (`pd2pl`) translator. A solid grasp of these ideas is essential for effective contribution.

## 1. Abstract Syntax Trees (ASTs)

At its core, `pd2pl` is an AST manipulator.

*   **What is an AST?** When Python code is executed, it's first parsed into an Abstract Syntax Tree. An AST is a tree representation of the syntactic structure of the code. Each node in the tree represents a construct occurring in the source code, like a function call, an assignment, a variable name, or an arithmetic operation.
*   **Why ASTs?** Working with ASTs allows the translator to understand the structure and semantics of the pandas code programmatically, rather than just doing simple text replacement. This enables more intelligent and robust translations.
*   **Python's `ast` Module**: The project heavily relies on Python's built-in `ast` module. This module provides tools to:
    *   Parse Python source code into an AST (`ast.parse()`).
    *   Traverse and inspect AST nodes.
    *   Modify AST nodes (e.g., change function names, arguments, or even replace entire subtrees).
    *   Convert a modified AST back into Python source code (`ast.unparse()`).

**Example AST Node (`ast.Call`):**
A pandas call like `df.sum()` would be represented in the AST (simplified) as:

```
Call(
  func=Attribute(
    value=Name(id='df', ctx=Load()),  # Represents 'df'
    attr='sum',                      # Represents '.sum'
    ctx=Load()
  ),
  args=[],
  keywords=[]
)
```

The translator inspects such `ast.Call` nodes, identifies them as pandas operations, and transforms them into Polars equivalents.

## 2. The `PandasToPolarsTransformer` (`translator.py`)

This class is the workhorse of the translation process.

*   **`ast.NodeTransformer`**: It subclasses `ast.NodeTransformer`, which provides a convenient way to traverse an AST and modify its nodes.
*   **Visitor Pattern**: It uses the visitor pattern. For each type of AST node (e.g., `ast.Call`, `ast.Attribute`, `ast.Name`), you can define a `visit_NodeType` method (e.g., `visit_Call`, `visit_Attribute`).
*   **Transformation Logic**: Inside these `visit_...` methods, the transformer:
    1.  Identifies if the current node represents a pandas operation.
    2.  If so, it consults the mapping system (see below) to determine the Polars equivalent.
    3.  Constructs new AST nodes representing the Polars operation.
    4.  Returns the new Polars AST node(s), which replace the original pandas node(s) in the tree.
    5.  If a node is not pandas-related, it's typically returned unchanged or its children are visited recursively.
*   **Centralized inplace Handling**: The transformer implements a centralized mechanism for handling `inplace=True` parameters, which are common in pandas but don't have direct Polars equivalents:
    1.  When a method with `inplace=True` is detected, the parameter is removed from the kwargs.
    2.  The resulting Polars operation is wrapped in an assignment expression that reassigns the result back to the original variable.
    3.  This approach ensures a consistent translation pattern for all methods that support inplace operations, without requiring special handling in each individual method mapping.

## 3. The Mapping System (`pd2pl/mapping/`)

This is where the specific translation rules are defined. It's a collection of Python modules that map pandas API elements to their Polars counterparts.

*   **`method_maps.py`**: Defines translations for pandas DataFrame and Series methods.
    *   **Simple Renames**: e.g., `df.sort_values()` -> `df.sort()`.
    *   **Argument Changes**: e.g., `df.drop(labels='col_name', axis=1)` -> `df.drop('col_name')`.
    *   **Complex Transformations**: Some pandas methods require more intricate transformations, potentially involving multiple Polars operations or restructuring of the call. These are often handled by dedicated transformer functions within `method_maps.py`.
    *   **ChainableMethodTranslation**: A helper class/pattern might be used to define how methods are translated, especially in chained calls, specifying Polars method names, argument transformations, and whether they produce expressions or modify DataFrames directly.
*   **`function_maps.py`**: Handles top-level pandas functions (e.g., `pd.concat()`, `pd.merge()`).
*   **`dtype_maps.py`**: Translates pandas dtype specifications (e.g., `df.astype('category')`) to Polars dtypes (`df.cast(pl.Categorical)`).
*   **`constructor_maps.py`**: Translates pandas object constructors (e.g., `pd.Series()`, `pd.DataFrame()`, `pd.Categorical()`) to their Polars equivalents, often involving `pl.Series()`, `pl.DataFrame()`, and specific dtype handling.
*   **`window_maps.py`**: Provides specialized translation logic for pandas window operations (`.rolling()`, `.expanding()`, `.ewm()`), which often have complex parameter mappings and require careful construction of Polars expressions.
*   **`string_maps.py`**: Handles translation of pandas string operations accessed through the `.str` accessor (e.g., `df['col'].str.lower()`, `df['col'].str.contains()`), mapping them to their Polars equivalents and transforming their parameters.

**Translation Strategy**: The mapping system aims to produce *idiomatic* Polars code, not just a literal one-to-one translation. This means the generated Polars code should look like code a human Polars user would naturally write.

## 4. Variable Tracking & Type Inference (The Role of `astroid`)

Simply looking at an AST node in isolation is sometimes not enough to determine if it's a pandas object or how to translate it.

*   **Challenge**: Consider `obj.some_method()`. Is `obj` a pandas DataFrame, a string, or something else? The translation of `some_method` depends on the type of `obj`.
*   **`astroid`**: This library extends Python's `ast` module by performing some level of static analysis and type inference. It can help determine the likely type of variables.
    *   `pd2pl` uses `astroid` (when available and configured) to:
        *   Track variable assignments (e.g., `df = pd.DataFrame(...)`).
        *   Infer if a variable (`df` in the example) holds a pandas DataFrame or Series.
        *   This allows the `PandasToPolarsTransformer` to be more confident when it encounters `df.some_method()` that it should indeed apply a pandas-to-polars translation.
*   **State in Transformer**: The `PandasToPolarsTransformer` maintains some state, such as a set of identified pandas DataFrame variable names (`self.dataframe_vars`), often populated with help from `astroid` or explicit annotations.

## 5. Configuration (`config.py`)

The translation process can be customized through a configuration system.

*   **Options**: Examples include:
    *   `rename_dataframe`: Whether to append `_pl` to DataFrame variable names (e.g., `df` becomes `df_pl`).
    *   `import_strategy`: How to handle `import polars as pl` statements (e.g., always add, add if needed).
*   **Impact**: Configuration options affect how the `PandasToPolarsTransformer` behaves and how the final code is generated by `imports_postprocess.py`.

## 6. Post-Processing (`imports_postprocess.py`)

After the AST is transformed and unparsed back into a code string, some final touches might be needed.

*   **Import Management**: The `PandasToPolarsTransformer` flags whether a Polars import (`import polars as pl`) or selectors import (`import polars.selectors as cs`) is needed based on the transformations performed. The `process_imports` function in `imports_postprocess.py` then adds these imports to the generated code string, avoiding duplicates and following the chosen `import_strategy`.
*   **Code Formatting**: Optionally, the generated code can be formatted using `black` for consistency.

## Date Range Translation and Period Alignment

When translating `pd.date_range` to `pl.date_range`, the translator aligns the end date to the correct period boundary, matching pandas behavior. For example:
- For month-end frequencies (`'M'`, `'ME'`), the end date is the last day of the month.
- For quarter-end frequencies (`'Q'`, `'QE'`), the end date is the last day of the quarter (Mar, Jun, Sep, Dec).
- For year-end frequencies (`'Y'`, `'YE'`), the end date is December 31st.
- For period-start frequencies (`'MS'`, `'QS'`, `'YS'`), the end date is the first day of the period.

This ensures that generated date ranges in Polars match the semantics and boundaries of pandas date ranges.

By understanding these core concepts, you'll be better equipped to read, debug, and extend the `pd2pl` translator.

Next, see the [Contribution Workflow Guide](./CONTRIBUTION_WORKFLOW.md) to learn how to apply this knowledge to make contributions. 