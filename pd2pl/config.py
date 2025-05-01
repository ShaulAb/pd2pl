from dataclasses import dataclass
from typing import Optional, Union
from .import_strategy import ImportStrategy

@dataclass
class TranslationConfig:
    """
    Configuration for pandas to polars translation.
    
    Attributes:
        import_strategy: Strategy for handling imports during translation
        postprocess_imports: Whether to postprocess imports (deprecated, use import_strategy instead)
        format_output: Whether to format the output code
    """
    import_strategy: Union[ImportStrategy, str] = ImportStrategy.AUTO
    postprocess_imports: bool = True  # For backward compatibility
    format_output: bool = True

    def __post_init__(self):
        """Validate configuration values."""
        if isinstance(self.import_strategy, str):
            self.import_strategy = ImportStrategy.from_string(self.import_strategy)

    _defaults = {
        "rename_dataframe": False,  # Default: keep variable names as-is
        "verbosity": 1,             # Default: normal verbosity
    }
    rename_dataframe = _defaults["rename_dataframe"]
    verbosity = _defaults["verbosity"]

    @classmethod
    def set(cls, **kwargs):
        """Set one or more config options."""
        for k, v in kwargs.items():
            if hasattr(cls, k):
                setattr(cls, k, v)

    @classmethod
    def reset(cls):
        """Reset all config options to their default values."""
        for k, v in cls._defaults.items():
            setattr(cls, k, v)

    @classmethod
    def get_config(cls):
        """Return a dict of current config values."""
        return {k: getattr(cls, k) for k in cls._defaults}

def set_config(**kwargs):
    """Convenience function to set config options."""
    TranslationConfig.set(**kwargs) 