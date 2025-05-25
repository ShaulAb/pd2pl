from .imports_postprocess import ImportStrategy

class TranslationConfig:
    """
    Global configuration for pandas-to-polars translation.
    Use set(), reset(), and get_config() to manage options.
    """
    _defaults = {
        "rename_dataframe": False,  # Default: keep variable names as-is
        "verbosity": 1,             # Default: normal verbosity
        "import_strategy": ImportStrategy.AUTO,  # New default
        "use_astroid": False,      # Default: use standard ast module
    }
    rename_dataframe = _defaults["rename_dataframe"]
    verbosity = _defaults["verbosity"]
    import_strategy = _defaults["import_strategy"]
    use_astroid = _defaults["use_astroid"]

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