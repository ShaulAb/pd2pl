# Pandas to Polars Translator

A Python tool that translates Pandas code to idiomatic Polars expressions, helping developers migrate their data processing pipelines from Pandas to Polars.

## Overview

This project provides an automated translator that converts Pandas DataFrame operations into their equivalent Polars expressions. It uses Python's Abstract Syntax Tree (AST) to parse and transform the code, and leverages the astroid library for advanced type inference and variable tracking, ensuring accurate and idiomatic translations even for complex assignment and chaining scenarios.

## Features & Current Status

Currently implemented:
- Hybrid AST + astroid-based code parsing and transformation
- Robust DataFrame variable propagation in assignments and chains
- Fallback to astroid for ambiguous or complex assignment cases
- Basic DataFrame method translations
- String operations translation
- GroupBy operations support
- Window functions (rolling, expanding) support
- DateTime accessor operations
- AST visualization for debugging
- Comprehensive test coverage for assignment, chaining, and method translation
- Callable/factory mapping for special-case constructor translation (e.g., pd.Categorical) implemented and tested
- Categorical columns are now translated to pl.Series(values=[...], dtype=pl.Categorical)

Work in progress:
- More advanced aggregation operations
- Additional edge case and performance tests
- Performance optimizations

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
# Using uv (recommended)
uv pip install -r requirements.txt

# Or using pip
pip install -r requirements.txt
```

## Project Structure

```
pandas_to_polars_translator/
├── pandas_to_polars_translator/
│   ├── __init__.py           # Package initialization
│   ├── translator.py         # Main translation logic
│   ├── errors.py            # Custom error definitions
│   ├── logging.py           # Logging configuration
│   └── mapping/             # Translation mappings
│       ├── __init__.py
│       ├── method_maps.py   # Method translation mappings
│       ├── function_maps.py # Function translation mappings
│       ├── dtype_maps.py    # Dtype and constructor mapping (including callables for special cases)
│       └── method_categories.py # Method categorization
├── tests/                   # Test suite
├── examples/                # Usage examples
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## Testing

To run the test suite, use the following command:
```bash
pytest tests/
```

## Configuration

You can control translation options globally or per translation. For example, to enable DataFrame/Series renaming (appending '_pl' to variable names):

```python
from pd2pl.config import set_config, TranslationConfig
set_config(rename_dataframe=True)  # Enable renaming globally
# ...
TranslationConfig.reset()  # Reset to defaults
```

Or override for a single translation:

```python
from pd2pl import translate_code
polars_code = translate_code(pandas_code, config={"rename_dataframe": True})
```

By default, variable names are kept as-is.
