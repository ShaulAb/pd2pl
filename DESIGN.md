

Okay, great! Acknowledging the limitations and focusing on an iterative approach starting with common functionalities is a solid plan. Let's outline a potential design for this pandas_to_polars_translator system.

## I. Overall Approach

The system will take Python code (as a string or file) containing Pandas operations as input. It will parse this code into an Abstract Syntax Tree (AST), traverse the AST to find and replace Pandas API calls with their Polars equivalents based on predefined mapping rules, and finally generate the translated Python code string.

## II. Proposed Directory Structure

```bash
pandas_to_polars_translator/
│
├── pandas_to_polars_translator/
│   │
│   ├── __init__.py         # Exposes the main translate function
│   ├── parser.py           # Parses input Python code into AST
│   ├── translator.py       # Core AST transformation logic (Pandas -> Polars)
│   ├── generator.py        # Generates Python code from the translated AST
│   ├── errors.py           # Custom exception classes
│   ├── logging.py          # Logging setup
│   │
│   └── mapping/            # Directory for translation rules and knowledge
│       ├── __init__.py
│       ├── function_maps.py  # Maps pd.function(...) calls
│       ├── method_maps.py    # Maps df.method(...) calls
│       ├── argument_maps.py  # Handles specific argument translations/defaults
│       └── rules_registry.py # Central place to load/access all mapping rules
│
├── tests/
│   ├── _helpers_.py # utility functions for tests, not part of the main package
│   ├── test_basic_translations.py # Tests for basic, direct translations
│   ├── test_complex_translations.py # Tests for more involved translations
│   └── test_error_handling.py    # Tests for unsupported features etc.
│
├── examples/                 # Usage examples
│   └── basic_translation.py
│
├── README.md                 # Project documentation
├── pyproject.toml            # Build system and dependency configuration (e.g., using Poetry or Flit)
└── requirements.txt          # Or managed via pyproject.toml
```

## III. Core File Responsibilities

    `parser.py`:
        Contains functions like parse_code(code_string: str) -> ast.AST.
        Uses Python's built-in ast module (ast.parse).
        Handles initial SyntaxError exceptions from invalid Python input.

    `mapping/` directory:
        function_maps.py, method_maps.py: Define dictionaries or classes mapping Pandas function/method names to their Polars equivalents. This includes information on how arguments should be handled (renamed, reordered, default values changed, converted to Polars expressions).
            Example Entry: {'read_csv': {'polars_equiv': 'pl.read_csv', 'arg_map': {'filepath_or_buffer': 'source', ...}}}
        argument_maps.py: Might contain helper functions for complex argument transformations (e.g., translating Pandas agg dictionaries to Polars expressions).
        rules_registry.py: Loads and provides access to all the mapping rules defined in the other files. This decouples the translator from the specifics of where rules are defined.

    `translator.py`:
        Contains the main translation logic, likely implemented as a class inheriting from ast.NodeTransformer.
        Overrides methods like visit_Call, visit_Attribute to inspect AST nodes.
        Identifies nodes corresponding to Pandas API usage (e.g., checking if a function call's source is pd or a known DataFrame variable, and the function/method name matches a key in the mapping rules).
        Uses the mapping/rules_registry.py to get the translation rule for the identified Pandas construct.
        Constructs new AST nodes representing the equivalent Polars code (e.g., creating a pl.col(...) expression, changing function names, adjusting arguments).
        Crucially, it needs to handle scope and variable tracking to some extent (knowing which variables likely hold DataFrames). This is complex; initially, it might rely on common variable names (df, data) or simple heuristics.
        If a Pandas construct is encountered that has no rule in the registry, it should raise a specific error from errors.py.

    `generator.py`:
        Contains functions like generate_code(tree: ast.AST) -> str.
        Uses ast.unparse() (available in Python 3.9+) or a library like astor for older Python versions to convert the transformed AST back into a code string.
        May need logic to ensure necessary imports (import polars as pl) are present in the final code. This could be done by adding an import node at the beginning of the AST if Polars usage is detected during translation.

    `errors.py`:
        Defines custom exceptions:
            UnsupportedPandasUsageError(Exception): For Pandas functions/methods/arguments explicitly not supported by the translator.
            TranslationError(Exception): General errors during the translation phase.
            ParsingError(SyntaxError): Maybe wrap initial syntax errors.
        These exceptions should carry informative messages, potentially including line numbers from the original code.

    `__init__.py` (main package):
        Provides the primary public interface, e.g.:

```python
from .parser import parse_code
from .translator import PandasToPolarsTransformer
from .generator import generate_code
from .errors import TranslationError, UnsupportedPandasUsageError

def translate(code_string: str) -> str:
    """Translates a Python string with Pandas code to Polars code."""
    try:
        ast_tree = parse_code(code_string)
        transformer = PandasToPolarsTransformer() # Loads rules internally
        transformed_ast = transformer.visit(ast_tree)
        # Potentially add 'import polars as pl' if transformer indicates usage
        # transformed_ast = add_polars_import(transformed_ast)
        polars_code = generate_code(transformed_ast)
        return polars_code
    except SyntaxError as e:
        raise ParsingError(f"Invalid Python syntax: {e}") from e
    # Catch specific translation errors here if needed for refinement
    # except UnsupportedPandasUsageError as e: ...
```

## IV. Error Handling Strategy

    Parse Time: Catch SyntaxError in parser.py or the main translate function.
    Translation Time: The translator.py's NodeTransformer is the core place. When visiting a node representing a Pandas call:
        Check if a rule exists in the mapping/rules_registry.py.
        If no rule exists, raise UnsupportedPandasUsageError with details (e.g., function name, line number).
        If a rule exists but fails (e.g., complex argument mapping issue), raise TranslationError.
    Generation Time: Errors are less likely here if the AST is valid, but ast.unparse could theoretically fail.
    User Feedback: Errors should clearly state what failed (which function/method), where (line number if possible), and why (e.g., "Unsupported function", "Complex indexing not translated").

## V. Expandability

    Adding support for new Pandas functions/methods primarily involves adding entries to the relevant files in the mapping/ directory and potentially helper functions in argument_maps.py.
    Handling more complex translation scenarios might require enhancing the logic within translator.py's visitor methods.
    The tests suite should be expanded alongside any new features.

This design emphasizes modularity, making it easier to manage, test, and expand the translator iteratively, starting with the most straightforward Pandas API functionalities.
