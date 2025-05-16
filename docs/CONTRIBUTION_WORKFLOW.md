# pd2pl Contribution Workflow

This guide outlines the typical workflow for contributing to the `pd2pl` (Pandas-to-Polars) translator project. Following these steps will help ensure your contributions are effective and can be smoothly integrated.

## 1. Before You Start

*   **Onboarding**: Ensure you've gone through the [ONBOARDING.md](./ONBOARDING.md) guide and have a good understanding of the [Project Overview](./PROJECT_OVERVIEW.md), [Development Setup](./DEVELOPMENT_SETUP.md), [Code Structure](./CODE_STRUCTURE.md), and [Core Concepts](./CORE_CONCEPTS.md).
*   **Communication**: Check the project's issue tracker (e.g., GitHub Issues) for existing discussions, bugs, or feature requests related to what you want to work on. If you plan to tackle a significant new feature or change, it's often a good idea to open an issue first to discuss your approach.

## 2. Finding or Defining a Task

*   **`TASKS.md`**: The `../../TASKS.md` file in the project root lists known bugs, planned features, and areas for improvement, often categorized by priority. This is a great place to find a task.
*   **GitHub Issues**: Check the issue tracker for items labeled `good first issue`, `help wanted`, or specific bugs/features that interest you.
*   **Proposing a New Task**: If you've identified a pandas feature that isn't translated correctly or isn't supported yet, and it's not listed, consider:
    1.  Verifying the behavior in both pandas and Polars.
    2.  Opening an issue to discuss the new translation or bug fix.

## 3. Setting Up Your Branch

Always work on a new feature or bug fix in a separate Git branch. This keeps your changes isolated and makes the review process cleaner.

```bash
# Ensure your main branch is up-to-date (e.g., main, master, or develop)
# Replace `main` with the project's primary branch name if different
git checkout main
git pull origin main

# Create a new branch for your feature/bugfix
# Branch naming convention: feature/descriptive-name or fix/descriptive-name
git checkout -b feature/translate-new-method
# or
git checkout -b fix/issue-123-groupby-error
```

## 4. The Test-Driven Development (TDD) Approach (Highly Recommended)

For most translation tasks, especially adding support for new methods or fixing incorrect translations, a TDD approach is very effective:

1.  **Write a Failing Test First**: 
    *   Navigate to the `tests/` directory.
    *   Find an existing test file relevant to the pandas feature you're working on (e.g., `test_groupby_aggregations.py` for a `groupby` related feature) or create a new test file (e.g., `test_new_method_translation.py`).
    *   Add one or more test cases that demonstrate the pandas code you want to translate and the *expected* Polars output.
    *   Your test should initially **fail** because the translation logic doesn't exist or is incorrect.
    *   Use helper functions from `tests/_helpers.py` or existing test patterns to structure your test.

    **Example Test Snippet (Conceptual):**
    ```python
    # In a test file, e.g., tests/test_new_method_translation.py
    from pd2pl import translate_code
    from ._helpers import assert_translation # Assuming a helper

    def test_my_new_method_translation():
        pandas_code = "import pandas as pd; df = pd.DataFrame(); df.my_new_method(param='value')"
        expected_polars_code = "import polars as pl; df_pl = pl.DataFrame(); df_pl.equivalent_polars_method(polars_param='value')" # Fictional
        
        # Initially, this might raise an error or produce wrong code
        assert_translation(pandas_code, expected_polars_code)
    ```

2.  **Run the Test**: Confirm that your new test fails as expected.
    ```bash
    pytest tests/your_test_file.py::test_your_new_test_function
    ```

## 5. Implement the Translation Logic

Now, implement the code to make your failing test pass.

1.  **Locate Mapping Files**: Based on the type of pandas operation, navigate to the appropriate file in `pd2pl/mapping/`:
    *   DataFrame/Series methods: `method_maps.py`
    *   Top-level functions: `function_maps.py`
    *   Dtype conversions: `dtype_maps.py`
    *   Constructors: `constructor_maps.py`
    *   Window functions: `window_maps.py`

2.  **Add or Modify Mapping**: 
    *   Add a new entry or modify an existing one for the pandas operation.
    *   This might involve:
        *   Defining a simple string replacement for the method name.
        *   Creating a function to transform arguments (e.g., renaming, reordering, converting values).
        *   Implementing more complex logic to build the target Polars AST structure.
    *   Refer to existing mappings for examples.

3.  **Update `translator.py` (If Necessary)**: 
    *   Sometimes, especially for new categories of operations or very complex transformations, you might need to add or modify logic directly in `PandasToPolarsTransformer` in `translator.py`.
    *   This is less common for straightforward method translations but can be required for features that change how the AST is traversed or state is managed.

## 6. Run Tests and Iterate

After implementing the translation logic:

1.  **Run Your Specific Test**: 
    ```bash
    pytest tests/your_test_file.py::test_your_new_test_function
    ```
    Iterate on your implementation until your test passes.

2.  **Run All Relevant Tests**: Once your specific test passes, run all tests in the file or module you modified to ensure you haven't introduced regressions.
    ```bash
    pytest tests/your_test_file.py
    # or even broader tests related to the component
    ```

3.  **Run All Tests**: Before committing, it's a good practice to run the entire test suite to catch any unexpected side effects.
    ```bash
    pytest tests/
    ```

## 7. Code Style and Linting

*   Ensure your code adheres to the project's style guidelines (typically PEP 8).
*   Format your code using `black`.
    ```bash
    black .
    ```
*   Run linters (e.g., `flake8`, `pylint`) if configured for the project and address any issues.

## 8. Commit Your Changes

Commit your changes with a clear and descriptive commit message.

```bash
# Stage your changes
git add pd2pl/mapping/your_modified_map.py tests/your_test_file.py

# Commit
git commit -m "feat: Add translation for pandas Series.my_new_method

- Implements translation for my_new_method including parameter mapping.
- Adds comprehensive tests for various use cases.
- Closes #issue-number (if applicable)"
```

Follow conventional commit message formats if the project uses them (e.g., `feat:`, `fix:`, `docs:`, `test:`).

## 9. Push Your Branch and Create a Pull Request (PR)

1.  **Push your branch** to the remote repository:
    ```bash
    git push origin feature/translate-new-method
    ```

2.  **Create a Pull Request**: Go to the project's repository on GitHub (or your hosting platform). You should see a prompt to create a Pull Request from your recently pushed branch. 
    *   **Title**: Clear and concise, summarizing the change.
    *   **Description**: 
        *   Explain what the PR does and why.
        *   Link to any relevant issues (e.g., `Fixes #123`, `Closes #456`).
        *   Describe how you tested your changes.
        *   Include code snippets or examples if they help illustrate the change.
    *   Ensure the PR targets the correct base branch (usually `main` or `develop`).

## 10. Code Review and Iteration

*   Project maintainers will review your PR.
*   Be prepared to discuss your changes and make further modifications based on feedback.
*   Once the PR is approved and passes any automated checks (CI), it will be merged.

## 11. After Merging

*   You can pull the latest changes from the main branch to your local main branch.
*   You can delete your feature/fix branch locally and remotely if it's no longer needed.

    ```bash
    git checkout main
    git pull origin main
    git branch -d feature/translate-new-method
    git push origin --delete feature/translate-new-method # Optional
    ```

Thank you for contributing to `pd2pl`!

Next, if you are working on window functions, review the [Window Function Translation Guide](./WINDOW_FUNCTION_TRANSLATION.md). 