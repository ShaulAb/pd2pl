"""Categories and base classes for method translations."""
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Callable


class MethodCategory(Enum):
    """Categories of DataFrame methods with similar translation patterns."""
    AGGREGATION = 'aggregation'
    TRANSFORM = 'transform'
    WINDOW = 'window'
    BASIC = 'basic'
    RESHAPE = 'reshape'  # Added for melt/pivot etc.
    STRING_METHODS = 'string'  # Added for string operations

@dataclass
class ChainableMethodTranslation:
    """Enhanced translation class supporting method chaining and selectors."""
    polars_method: str
    category: MethodCategory
    argument_map: Optional[Dict[str, str]] = None
    method_chain: Optional[Callable[[list, dict], Optional[List[Tuple[str, list, dict]]]]] = None
    requires_selector: bool = False
    doc: str = "" 