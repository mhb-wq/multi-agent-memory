from .long_context import LongContextAgent
from .rag_naive import NaiveRAGAgent
from .rag_policy import PolicyRAGAgent
from .a_mem import AMemAgent
from .mem0 import Mem0Agent
from .remem import ReMemAgent
from .example_agent import ExampleAgent
from .external_baselines import BM25Agent, DGMemAgent, EGMemoryAgent, MemGPTAgent, LightMemAgent, MemoryBankAgent
from .ours_formal import OursFormalAgent

AGENT_REGISTRY = {
    "long_context": LongContextAgent,
    "rag_naive": NaiveRAGAgent,
    "rag_policy": PolicyRAGAgent,
    "a_mem": AMemAgent,
    "mem0": Mem0Agent,
    "remem": ReMemAgent,
    "example": ExampleAgent,
    "bm25": BM25Agent,
    "dg_mem": DGMemAgent,
    "egmemory": EGMemoryAgent,
    "memgpt": MemGPTAgent,
    "lightmem": LightMemAgent,
    "memorybank": MemoryBankAgent,
    "ours": OursFormalAgent,
}
