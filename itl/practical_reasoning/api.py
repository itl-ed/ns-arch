"""
Practical reasoning module API that exposes only the high-level functionalities
required by the ITL agent: maintain agenda as a stack of to-do items, select
actions to carry out to clear executable tasks of the agenda (whenever possible)
"""
class PracticalReasonerModule:
    
    def __init__(self):
        self.agenda = []

    def act(self):
        """
        Just eagerly try to resolve each item in the agenda stack from the top as much
        as possible. I wonder if we'll ever need a more sophisticated mechanism than this
        simple, greedy method for a good while?
        """
        for todo in reversed(self.agenda):
            self._resolve(todo)
    
    def _resolve(self, todo):
        """
        Clear a to-do item off the agenda by choosing and executing an action accordingly
        """
        ...
