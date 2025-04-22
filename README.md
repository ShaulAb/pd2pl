# Pandas to Polars Translator

A Python tool that translates Pandas code to idiomatic Polars expressions, helping developers migrate their data processing pipelines from Pandas to Polars.

## Overview

This project provides an automated translator that converts Pandas DataFrame operations into their equivalent Polars expressions. It uses Python's Abstract Syntax Tree (AST) to parse and transform the code, ensuring accurate and idiomatic translations.

## Features & Current Status

Currently implemented:
- AST-based code parsing and transformation
- Basic DataFrame method translations
- String operations translation
- GroupBy operations support
- Window functions (rolling, expanding) support
- DateTime accessor operations
- AST visualization for debugging

Work in progress:
- Complex method chaining translations
- Advanced aggregation operations
- More comprehensive test coverage
- Performance optimizations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pandas_to_polars_translator.git
cd pandas_to_polars_translator
```

2. Install dependencies:
```bash
# Using uv (recommended)
uv pip install -r requirements.txt
...
```

3. Install system dependencies:
```bash
# For AST visualization (Ubuntu/Debian)
sudo apt-get install graphviz
```

## Project Structure

```
pandas_to_polars_translator/
├── pandas_to_polars_translator/
│   ├── __init__.py           # Package initialization
│   ├── translator.py         # Main translation logic
│   ├── ast_visualizer.py     # AST visualization tools
│   ├── errors.py            # Custom error definitions
│   ├── logging.py           # Logging configuration
│   └── mapping/             # Translation mappings
│       ├── __init__.py
│       ├── method_maps.py   # Method translation mappings
│       ├── function_maps.py # Function translation mappings
│       └── method_categories.py # Method categorization
├── tests/                   # Test suite
├── examples/                # Usage examples
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## Development Tools

The project includes several development tools:

1. **AST Visualizer**: Helps understand code structure and debug translations
```python
from pandas_to_polars_translator.ast_visualizer import visualize_ast

# Visualize pandas code structure
visualize_ast("df.groupby('col').mean()", "output.pdf")
```

2. **Test Suite**: Comprehensive tests for translation accuracy
```bash
pytest tests/
```

## Usage Examples

Basic translation example:
```python
from pandas_to_polars_translator import translate_code

# Pandas code
pandas_code = """
result = df.groupby('category')['value'].mean()
"""

# Translate to Polars
polars_code = translate_code(pandas_code)
print(polars_code)
# Output: result = df_pl.groupby('category').agg(pl.col('value').mean())
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

...
