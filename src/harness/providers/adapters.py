"""Adapter dataclasses for logging compatibility across providers.

The logging layer (logging.py) uses duck-typing (hasattr) to inspect message
objects. Claude SDK objects have the right attributes natively. For OpenAI and
Gemini, these simple dataclasses match the same shape so logging works
transparently.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolCallBlock:
    """Matches: hasattr(block, 'name') and hasattr(block, 'id')."""
    name: str
    id: str
    input: Optional[dict] = None


@dataclass
class TextBlock:
    """Matches: hasattr(block, 'text')."""
    text: str


@dataclass
class ToolResultBlock:
    """Matches: hasattr(block, 'tool_use_id')."""
    tool_use_id: str
    content: Any = ""
    is_error: bool = False


@dataclass
class AdapterMessage:
    """Matches: hasattr(message, 'content')."""
    content: list = field(default_factory=list)
