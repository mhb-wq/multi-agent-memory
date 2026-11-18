"""Optional baseline adapters for GroupMemBench.

Importing this package is dependency-free.  The third-party memory systems are
imported by their own adapters only when a caller constructs one.
"""

from .amem import AMEMAdapter
from .base import (
    BaselineAdapter,
    Message,
    Passage,
    format_passage,
    judge_user_prompt,
    load_messages,
    qa_user_prompt,
    search_text,
)
from .builders import build_amem, build_amem_paper, build_mem0, build_memos, build_memos_native
from .mem0 import Mem0Adapter
from .memos import MemOSAdapter, MemOSNativeAdapter
from .naive_rag import NaiveRAGAdapter

__all__ = [
    "BaselineAdapter",
    "Message",
    "Passage",
    "AMEMAdapter",
    "Mem0Adapter",
    "MemOSAdapter",
    "MemOSNativeAdapter",
    "NaiveRAGAdapter",
    "build_amem",
    "build_amem_paper",
    "build_mem0",
    "build_memos",
    "build_memos_native",
    "format_passage",
    "judge_user_prompt",
    "load_messages",
    "qa_user_prompt",
    "search_text",
]
