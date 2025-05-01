from enum import Enum

class ImportStrategy(Enum):
    """
    Defines strategies for handling imports during translation.
    
    AUTO: Automatically determines if imports are needed based on code content
    ALWAYS: Always adds required imports regardless of input
    NEVER: Never adds imports, even if they're needed
    PRESERVE: Preserves existing import structure while replacing pandas imports
    """
    AUTO = "auto"
    ALWAYS = "always"
    NEVER = "never"
    PRESERVE = "preserve"

    @classmethod
    def from_string(cls, value: str) -> 'ImportStrategy':
        """Convert string to ImportStrategy enum value."""
        try:
            return cls(value.lower())
        except ValueError:
            valid_strategies = [m.value for m in cls]
            raise ValueError(f"Invalid import strategy: {value}. Must be one of: {', '.join(valid_strategies)}") 