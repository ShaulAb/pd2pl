# Pandas to Polars Translation Tasks

This document outlines the planned test coverage and implementation tasks for the pandas-to-polars translator, organized by API categories and priority.

## High Priority

These operations are commonly used in data analysis workflows and should be implemented first:

- [x] Filtering Operations
  - [x] Basic boolean filtering (`df[df['col'] > 5]`)
  - [x] Complex boolean conditions (`&`, `|`, `~`)
  - [x] `isin` filtering
  - [x] `isna`/`notna` filtering
  - [x] String contains/startswith/endswith filtering
- [x] Assignment & Chaining
  - [x] Robust DataFrame variable propagation in assignments and chains (hybrid AST+astroid approach)
  - [x] Fallback to astroid for ambiguous or complex assignment cases
- [x] Dtype Conversion
  - [x] Robust, extensible dtype mapping system implemented
  - [x] Supports: category, int, float, boolean dtypes (including extension dtypes)
  - [x] Handles .astype(), Series/DataFrame creation with dtype
  - [x] Comprehensive tests implemented and passing
  - [x] **Special-case constructor translation (e.g., pd.Categorical to pl.Series with dtype=pl.Categorical) implemented using callable/factory mapping**

- [ ] Aggregation Methods
  - [x] Complete GroupBy operations
    - [x] Multiple column groupby
    - [x] Named aggregations
    - [x] Custom aggregation functions / lambda functions
  - [ ] Window Functions
    - [ ] Complete rolling operations
    - [ ] Complete expanding operations

## Medium Priority

Common but less critical operations:

- [ ] Column Operations
  - [ ] Column Selection
    - [ ] Boolean indexing
    - [ ] Label-based indexing
    - [ ] Integer-based indexing
  - [x] Column Modification
    - [x] `rename`
    - [x] `drop`

- [ ] Row Operations
  - [x] `sort_values`
  - [x] `sample`
  - [x] `drop_duplicates`
  - [ ] `drop_na`
  - [ ] `reset_index`
  - [ ] `set_index`

- [ ] Reshaping Operations
  - [x] `melt`
  - [x] `pivot` (Current implementation requires explicit `values` arg)
  - [x] `pivot_table` (Supports basic string aggfuncs and fill_value; margins, dropna, complex aggfuncs unsupported)

## Lower Priority

Less commonly used but still important features:

- [ ] Join Operations
  - [x] `merge` (Column joins only; index joins and lsuffix unsupported)
  - [ ] `concat`

- [ ] DateTime Operations
  - [ ] `date_range`
  - [ ] Date components (year, month, day)
  - [ ] Time components (hour, minute, second)
  - [ ] Date arithmetic
  - [ ] Timezone handling

- [ ] String Operations
  - [ ] `split`
  - [ ] `replace`
  - [ ] `extract`
  - [ ] Regular expressions


## Implementation Notes

- Assignment and chain translation is now robust via a hybrid AST+astroid approach:
  - Simple assignments and method chains are handled by AST logic.
  - Ambiguous or complex assignments fall back to astroid for type inference and DataFrame detection.
  - This ensures correct DataFrame variable propagation and `_pl` suffixing in both simple and complex cases.
- **Special-case constructor translation (e.g., pd.Categorical) is now handled via callables in mapping/dtype_maps.py, making it easy to extend for future cases.**

For each feature, we need to:
1. Add test cases in `tests/test_complex_translations.py` or create new test files
2. Update method mappings in `mapping/method_maps.py`
3. Implement translation logic in `translator.py`
4. Add examples in the documentation
5. Update README.md with new features

## Testing Strategy

Each feature should have:
1. Basic functionality tests
2. Edge case tests
3. Performance comparison tests
4. Error handling tests

## Priority Guidelines

When implementing features:
1. Focus on operations that are most commonly used in data analysis
2. Prioritize features that are essential for production workflows
3. Consider the complexity of implementation vs. value added
4. Pay special attention to operations that might have performance implications 

# Optimization Notes

Known tradeoffs:

In the `drop_duplicates` translation:
**Current Implementation (`maintain_order=True`)**
   - ✅ Exactly matches pandas behavior
   - ✅ Preserves row order as users expect from pandas
   - ❌ Cannot use streaming engine
   - ❌ Performance penalty for large datasets
   - ❌ Higher memory usage

**Potential Optimized Version (`maintain_order=False`)**
   - ✅ Can use streaming engine
   - ✅ Better performance on large datasets
   - ✅ Lower memory usage
   - ❌ Different row ordering than pandas
   - ❌ Might break user expectations

## Suggested Enhancements

1. **Global Optimization Flag:**
```python
# pd2pl/__init__.py
class TranslationConfig:
    optimize: bool = False  # Default to pandas-like behavior

# Usage
pd2pl.TranslationConfig.optimize = True
translated = translate_code("df.drop_duplicates()")  # -> df_pl.unique()  # maintain_order not set
```

2. **Per-Translation Optimization:**
```python
# More granular control
translated = translate_code(
    "df.drop_duplicates()", 
    optimizations={
        "drop_duplicates": {
            "streaming": True,  # Don't use maintain_order
            "parallel": True,   # Future: Other optimizations
        }
    }
)
```

3. **Documentation-Based Warning:**
```python
def _transform_drop_duplicates_chain(args: List[Any], kwargs: Dict[str, Any]) -> List[Tuple[str, List[Any], Dict[str, Any]]]:
    """
    Performance Note:
    ----------------
    This translation prioritizes pandas compatibility over performance
    by setting maintain_order=True. For better performance with different
    ordering semantics, consider using optimize=True in translation config
    or directly using df_pl.unique() without maintain_order in your code.
    """
```

I think this would make a great future enhancement, especially as we identify more such performance vs. compatibility tradeoffs in other translations. We could:

1. Keep the current behavior as default (safe, compatible)
2. Add the optimization flag in a future PR
3. Document performance implications clearly
4. Potentially add warnings when the optimization flag would make a significant difference
