"""
Practical reasoning module API that exposes only the high-level functionalities
required by the ITL agent: manage agenda as a set of states that potentially 
violate ITL agent's 'maintenance goals', generating plans to execute for resolving
states violating the maintenance goals (whenever possible)
"""
from .plans.library import library


class PracticalReasonerModule:
    
    def __init__(self):
        self.agenda = []

    def obtain_plan(self, todo):
        """
        Obtain appropriate plan from plan library (maybe this could be extended later
        with automated planning...)
        """
        return library.get(todo)
