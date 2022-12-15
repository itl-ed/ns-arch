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
    # Resolve neologism by requesting definitions in language or exemplars, in
    # accordance with the agent's learning interaction stategy
    "address_neologism": None,

    # Handle unanswered question by first checking if it can be answered with agent's
    # current knowledge, and if so, adding to agenda to actually answer it
    "address_unanswered_Q": [
        # Prepare answering utterance to generate
        {
            "action_method": Val(referrable=["comp_actions", "attempt_answer_Q"]),
            "action_args_getter": lambda x: (Val(data=x),)
        },
    ],

    # Answer a question by computing answer candidates, selecting an answer, translating
    # to natural language and then generating it
    "answer_Q": [
        # Prepare the answer to be uttered
        {
            "action_method": Val(referrable=["comp_actions", "prepare_answer_Q"]),
            "action_args_getter": lambda x: (Val(data=x),)
        },
        # Generate the prepared answer
        {
            "action_method": Val(referrable=["lang", "generate"]),
            "action_args_getter": lambda x: ()
        }
    ],

    # Resolve mismatch between agent's vs. user's perception by asking question,
    # in accordance with the agent's learning interaction stategy
    "address_mismatch": [
        # Prepare answering utterance to generate
        {
            "action_method": Val(referrable=["comp_actions", "handle_mismatch"]),
            "action_args_getter": lambda x: (Val(data=x),)
        },
        # Generate whatever response queued
        {
            "action_method": Val(referrable=["lang", "generate"]),
            "action_args_getter": lambda x: ()
        }
    ],

    # Resolve agent's confusion between two visual concepts revealed during & via
    # learning dialogue with user
    "address_confusion": [
        # Prepare answering utterance to generate
        {
            "action_method": Val(referrable=["comp_actions", "handle_confusion"]),
            "action_args_getter": lambda x: (Val(data=x),)
        },
        # Generate whatever response queued
        {
            "action_method": Val(referrable=["lang", "generate"]),
            "action_args_getter": lambda x: ()
        }
    ],

    # Nothing special has to be addressed, just acknowledge user input
    "acknowledge": [
        # Acknowledge in a cool way, saying "OK."
        {
            "action_method": Val(referrable=["lang", "acknowledge"]),
            "action_args_getter": lambda x: ()
        },
        # Generate the cool acknowledgement
        {
            "action_method": Val(referrable=["lang", "generate"]),
            "action_args_getter": lambda x: ()
        }
    ]
}
