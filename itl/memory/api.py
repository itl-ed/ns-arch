"""
Long-term memory module API that exposes only the high-level functionalities
required by the ITL agent: read, maintain and write lexicon, knowledge base, and
concept exemplars
"""
from .lexicon import Lexicon


class LongTermMemoryModule:
    
    def __init__(self):
        self.lexicon = Lexicon()
        self.knowledge_base = ...