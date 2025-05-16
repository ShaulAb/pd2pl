# Development Setup Guide for pd2pl

This guide will walk you through setting up your local development environment for the Pandas-to-Polars (`pd2pl`) translator project.

## 1. Prerequisites

*   **Python**: Version 3.8 or higher. We recommend using a Python version manager like `pyenv` or `conda` to manage your Python versions and environments.
*   **Git**: For cloning the repository and version control.
*   **pip**: Python's package installer (usually comes with Python).
*   **uv (Optional but Recommended)**: A fast Python package installer and resolver. If you have it, it can speed up dependency installation. You can install it with `pip install uv`.

## 2. Clone the Repository

First, clone the project repository from its source (e.g., GitHub):

```bash
# Replace with the actual repository URL
git clone https://github.com/your-username/pd2pl-project.git
cd pd2pl-project/pandas_to_polars_translator 
# Navigate into the main package directory where pyproject.toml is located
```

## 3. Create and Activate a Virtual Environment

It's crucial to work within a virtual environment to isolate project dependencies.

**Using `venv` (standard Python):**

```bash
# From the pandas_to_polars_translator directory
python -m venv .venv

# Activate the environment
# On macOS and Linux:
source .venv/bin/activate
# On Windows (Git Bash or similar):
# source .venv/Scripts/activate
# On Windows (Command Prompt):
# .venv\Scripts\activate.bat
```

**Using `conda`:**

```bash
# From the pandas_to_polars_translator directory
conda create --name pd2pl-dev python=3.9  # Or your preferred Python 3.8+ version
conda activate pd2pl-dev
```

## 4. Install Dependencies

Once your virtual environment is activated, install the project dependencies, including development dependencies.

The project uses `pyproject.toml` for dependency management.

**Using `uv` (Recommended for speed):**

```bash
# From the pandas_to_polars_translator directory
uv pip install -e .[dev,test]
```

**Using `pip`:**

```bash
# From the pandas_to_polars_translator directory
pip install -e .[dev,test]
```

This command installs the `pd2pl` package in "editable" mode (`-e`), meaning changes you make to the source code will be immediately effective without needing to reinstall. It also installs the optional dependencies listed under `[dev,test]` in `pyproject.toml` (like `pytest`, `black`, `astroid`).

## 5. Verify Installation

To ensure everything is set up correctly:

1.  **Run the test suite**: This is the best way to confirm that the core components are working and all dependencies are correctly installed.

    ```bash
    # From the pandas_to_polars_translator directory
    pytest tests/
    ```
    You should see a series of passing tests. Some might be skipped if they depend on features not yet fully implemented, which is normal.

2.  **Try a simple translation (optional)**:
    Open a Python interpreter or create a small script:

    ```python
    from pd2pl import translate_code

    pandas_code = "import pandas as pd\ndf = pd.DataFrame({'A': [1, 2, 3]})\ndf['A'].sum()"
    polars_code = translate_code(pandas_code, postprocess_imports=True, format_output=True)
    print(polars_code)
    ```
    This should output the Polars equivalent of the input pandas code, properly formatted and with necessary imports.

## 6. Code Editor Setup (VS Code Example)

If you're using VS Code (or a similar modern editor), consider the following for a smoother development experience:

*   **Python Interpreter**: Ensure your editor is configured to use the Python interpreter from the virtual environment you created. VS Code usually prompts you to select an interpreter or you can set it via the command palette (`Python: Select Interpreter`).
*   **Linters and Formatters**: 
    *   Install the Python extension for VS Code.
    *   Configure it to use `black` for formatting and a linter like `flake8` or `pylint` (if you install them in your venv).
    *   Add these to your VS Code `settings.json` (workspace settings):
        ```json
        {
            "python.formatting.provider": "black",
            "editor.formatOnSave": true,
            "python.linting.flake8Enabled": true, // if using flake8
            "python.linting.pylintEnabled": true, // if using pylint
            "python.linting.enabled": true
        }
        ```
*   **Test Explorer**: VS Code's Python extension can discover and run `pytest` tests through its Test Explorer UI, which can be very convenient.

## 7. Keeping Dependencies Updated

To update dependencies to their latest compatible versions as defined in `pyproject.toml` and `uv.lock` (if using uv):

**Using `uv`:**

```bash
# To sync with the lock file
uv pip sync

# To update specific packages and then the lock file
# uv pip install --upgrade <package_name>
# uv pip compile pyproject.toml -o uv.lock
```

**Using `pip` (less precise without a lock file like `uv.lock` or `poetry.lock`):

```bash
pip install --upgrade -e .[dev,test]
```

You are now ready to explore the codebase and start contributing!

Next, head to the [Code Structure Guide](./CODE_STRUCTURE.md) to understand how the project is organized. 