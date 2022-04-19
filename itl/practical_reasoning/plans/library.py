"""
Library of plans. Each plan consists of a sequence of action specifications and
prerequisites which must be satisfied for execution of the plan.

Each action is specified by an 'action' method from some module of the agent and
a lambda function used to transform arguments as needed.

Each prerequisite is specified by a 'predicate' method (i.e. Boolean-returning method)
from some module of the agent, a lambda function used to transform arguments as
needed, and a Boolean value the predicate method is supposed to return.
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
    # Integrate provided knowledge into agent KB
    "unintegrated_knowledge": (
        # Plan
        [
            # Add from corresponding dialogue record to KB
            {
                "action_method": Val(referrable=["lt_mem", "kb_add_from_dialogue"]),
                "action_args_getter": lambda x: (Val(data=x), Val(referrable=["lang"]))
            }
        ],
        # Prerequisites
        [
            # Utterance doesn't contain any neologisms
            {
                "prereq_method": Val(referrable=["lang", "utt_contains_neologism"]),
                "prereq_args_getter": lambda x: Val(data=x),
                "prereq_value": False
            }
        ]
    ),

    # Resolve neologism by requesting definitions in language or exemplars
    "unresolved_neologism": (None, None),

    # Resolve mismatch between agent's vs. user's perception by asking question
    "unresolved_mismatch": (None, None),

    # Handle unanswered question by finding answer and making utterance
    "unanswered_Q": (
        # Plan
        [
            # Compute answer to the question based on current cognition
            {
                "action_method": Val(referrable=["cognitive", "compute_answer_to_Q"]),
                "action_args_getter": lambda x: (Val(data=x), Val(referrable=["lang"]))
            },
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
        ],
        # Prerequisites
        [
            # Utterance doesn't contain any neologisms
            {
                "prereq_method": Val(referrable=["lang", "utt_contains_neologism"]),
                "prereq_args_getter": lambda x: Val(data=x),
                "prereq_value": False
            }
        ] 
    )
}
