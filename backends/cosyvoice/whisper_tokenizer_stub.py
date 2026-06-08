"""Minimal `whisper.tokenizer` shim for CosyVoice.

CosyVoice's tokenizer module does ``from whisper.tokenizer import Tokenizer`` at
import time, but the actual CosyVoice2/3 paths use a transformers AutoTokenizer.
The whisper ``Tokenizer`` class is only referenced as a type annotation / in the
unused ``get_tokenizer`` helper, so a permissive stand-in is sufficient.
"""
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple


LANGUAGES: Dict[str, str] = {
    "en": "english",
    "zh": "chinese",
}


@dataclass
class Tokenizer:
    """Permissive stand-in for ``whisper.tokenizer.Tokenizer``."""

    encoding: Any = None
    num_languages: int = 0
    language: Optional[str] = None
    task: Optional[str] = None
    sot_sequence: Tuple[int, ...] = ()
    special_tokens: Dict[str, int] = None

    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = {}

    def encode(self, text, **kwargs):
        if self.encoding is not None:
            return self.encoding.encode(text, **kwargs)
        raise NotImplementedError("whisper tokenizer stub: no encoding configured")

    def decode(self, tokens, **kwargs):
        if self.encoding is not None:
            return self.encoding.decode(tokens, **kwargs)
        raise NotImplementedError("whisper tokenizer stub: no encoding configured")
