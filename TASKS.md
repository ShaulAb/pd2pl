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

- [ ] Aggregation Methods
  - [ ] Complete GroupBy operations
    - [ ] Multiple column groupby
    - [ ] Named aggregations
    - [ ] Custom aggregation functions
  - [ ] Window Functions
    - [ ] Complete rolling operations
    - [ ] Complete expanding operations
    - [ ] Custom window functions

## Medium Priority

Common but less critical operations:

- [ ] Column Operations
  - [ ] Column Selection
    - [ ] Boolean indexing
    - [ ] Label-based indexing
    - [ ] Integer-based indexing
  - [ ] Column Modification
    - [ ] `assign`
    - [ ] `rename`
    - [ ] `drop`

- [ ] Row Operations
  - [ ] `sort_values`
  - [ ] `drop_duplicates`
  - [ ] `reset_index`
  - [ ] `set_index`

- [ ] Reshaping Operations
  - [ ] `melt`
  - [ ] `pivot`
  - [ ] `pivot_table`

## Lower Priority

Less commonly used but still important features:

- [ ] Join Operations
  - [ ] `merge`
  - [ ] `join`
  - [ ] `concat`

- [ ] DateTime Operations
  - [ ] Date components (year, month, day)
  - [ ] Time components (hour, minute, second)
  - [ ] Date arithmetic
  - [ ] Timezone handling

- [ ] String Operations
  - [ ] `split`
  - [ ] `replace`
  - [ ] `extract`
  - [ ] Regular expressions

## Nice to Have

Features that would be useful but aren't critical:

- [ ] Advanced Features
  - [ ] `eval`/`query`
  - [ ] Category dtype support
  - [ ] Memory optimization methods
  - [ ] Custom function application

## Implementation Notes

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