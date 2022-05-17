"""
Library of plans. Each plan consists of a sequence of action specifications.

Each action is specified by an 'action' method from some module of the agent and
a lambda function used to transform arguments as needed.
"""
class Val:
    """
    Designates value which can be either an object referrable within agent object, or
    pure data
    """
    def __init__(self, referrable=None, data=None):
        assert referrable is None or data is None, "Provide only one"

        self.referrable = referrable
        self.data = data
    
    def extract(self, agent_obj):
        """ Strip off the Val class wrapper to recover designated value """
        if self.referrable is not None:
            value = agent_obj
            for field in self.referrable:
                value = getattr(value, field)
            return value
        else:
            assert self.data is not None
            return self.data


library = {
    # Resolve neologism by requesting definitions in language or exemplars
    "address_neologism": None,

    # Resolve mismatch between agent's vs. user's perception by asking question
    "address_mismatch": None,

    # Handle unanswered question by finding answer and making utterance
    "answer_Q": [
        # Prepare answering utterance to generate
        {
            "action_method": Val(referrable=["lang", "prepare_answer"]),
            "action_args_getter": lambda x: (Val(data=x), Val(referrable=["cognitive"]))
        },
        # Generate the utterance
        {
            "action_method": Val(referrable=["lang", "generate"]),
            "action_args_getter": lambda x: ()
        }
    ]
}
