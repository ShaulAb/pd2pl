"""Custom exceptions for the pandas to polars translator."""

class TranslationError(Exception):
    """Base class for translation errors."""
    pass

class UnsupportedPandasUsageError(TranslationError):
    """Raised when encountering pandas functionality that cannot be translated."""
    pass

class ParsingError(SyntaxError):
    """Raised when the input code cannot be parsed."""
    pass

class ValidationError(TranslationError):
    """Raised when the translated code produces different results."""
    pass 