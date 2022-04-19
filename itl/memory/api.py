"""
Long-term memory module API that exposes only the high-level functionalities
required by the ITL agent: read, maintain and write lexicon, knowledge base, and
concept exemplars
"""
from .lexicon import Lexicon
from .kb import KnowledgeBase


U_W_PR = 1.0         # How much the agent values information provided by the user

class LongTermMemoryModule:
    
    def __init__(self):
        self.lexicon = Lexicon()
        self.kb = KnowledgeBase()

    def kb_add_from_dialogue(self, utt_id, lang):
        """
        Fetch utterance content by index from dialogue record as ASP rules, then
        add to KB
        """
        rules, _ = lang.utt_to_ASP(utt_id)
        for (rule, source) in rules:
            self.kb.add(rule, U_W_PR, source)
