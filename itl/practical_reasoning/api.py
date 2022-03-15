"""
Practical reasoning module API that exposes only the high-level functionalities
required by the ITL agent: maintain agenda as a stack of to-do items, select
actions to carry out to clear executable tasks of the agenda (whenever possible)
"""
class PracticalReasonerModule:
    
    def __init__(self):
        self.agenda = []

