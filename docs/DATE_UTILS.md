# Comprehensive Date/Time Translation Subsystem for pandas-to-polars

## Architecture Overview

The date/time translation subsystem will be implemented as a dedicated subpackage with modular components to handle various aspects of date/time translation between pandas and polars.

## Module Structure

1. `pd2pl.datetime_utils/` - New subpackage for all date/time related functionality
   - `__init__.py` - Package exports and public API
   - `parser.py` - Date string/object parsing utilities
   - `frequency.py` - Frequency string parsing and mapping
   - `calculator.py` - Date calculation utilities (periods → dates)
   - `transformers.py` - AST transformation helpers for date/time functions

## Implementation Details

### Core Components

- **Date Parser**: Convert various pandas date representations to a standard format
- **Frequency Parser**: Parse and normalize pandas frequency strings (including prefixed like '2D')
- **Frequency Mapper**: Map pandas frequency strings to polars interval strings
- **Date Calculator**: Calculate end dates based on start, periods, and frequency
- **Function Handlers**: Specialized logic for each pandas date/time function

### Integration Strategy

- Register transformation functions in the constructor map
- Implement preprocessing for string-based date arguments
- Handle each pandas date/time function with a specific transformer

IMPLEMENTATION CHECKLIST:

1. Create the basic directory structure for `pd2pl.datetime_utils` package
2. Create `__init__.py` with package exports
3. Implement `frequency.py` with frequency parsing and mapping
   - Create `parse_frequency()` function to handle numeric prefixes
   - Implement `map_pandas_freq_to_polars_interval()` function
   - Add support for anchored frequencies (W-MON, etc.)
4. Implement `parser.py` with date parsing utilities
   - Create `parse_date_string()` function supporting common formats
   - Implement `normalize_date_arg()` to handle different date representations
5. Implement `calculator.py` with date calculation utilities
   - Create `calculate_end_date()` function for period-based calculations
   - Add support for business day calculations
   - Implement frequency-specific calculation functions
6. Implement `transformers.py` with AST transformation helpers
   - Create `DateRangeTransformer` class
   - Add `PeriodRangeTransformer` class
   - Implement `ResampleTransformer` class
7. Update the main translator to use the new subsystem
   - Modify constructor map to use new transformers
   - Add preprocessing logic for string date handling
8. Create comprehensive unit tests for each component
   - Test frequency parsing with various inputs
   - Test date calculations with different frequencies
   - Test end-to-end translations
9. Integrate with existing code
   - Migrate functionality from `date_maps.py` to new modules
   - Update import statements in relevant files
10. Add support for `date_range()` translation
    - Handle string dates, periods, frequencies
    - Transform to appropriate polars code
11. Add support for `period_range()` translation
12. Add support for `resample()` method translation
13. Add support for other date/time functions as needed
14. Update documentation in `CORE_CONCEPTS.md`
15. Update `TASKS.md` with implementation status
16. Create user documentation for supported date/time translations
17. Write developer documentation for extending the subsystem
