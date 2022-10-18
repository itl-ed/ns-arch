import unittest

import numpy as np

from itl.lpmln import Literal, Rule, Program


class TestProgramCompile(unittest.TestCase):

    def test_compile_single_grounded_fact(self):
        prog = Program()

        # logit(0.9) :: p.
        prog.add_rule(Rule(head=Literal("p", [])), 0.9)

        bjt = prog.compile()

        # Weight for {p} in the singleton node should be logit(0.9), and should equal
        # 0.9/0.1 when exp'ed
        p_ind = bjt.graph["atoms_map"][Literal("p", [])]
        p_singleton = frozenset({p_ind})
        p_output = bjt.nodes[p_singleton]["output_beliefs"]

        self.assertAlmostEqual(
            float(np.exp(p_output[p_singleton].primitivize())), 0.9/0.1
        )

    def test_compile_grounded_fact_and_rule(self):
        prog = Program()

        # logit(0.9) :: p.
        # logit(0.8) :: q :- p.
        prog.add_rule(Rule(head=Literal("p", [])), 0.9)
        prog.add_rule(Rule(head=Literal("q", []), body=[Literal("p", [])]), 0.8)

        bjt = prog.compile()

        # Weight for {q} in the singleton node should be logit(0.9)+logit(0.8), and
        # should equal 0.9/0.1 * 0.8/0.2 when exp'ed
        q_ind = bjt.graph["atoms_map"][Literal("q", [])]
        q_singleton = frozenset({q_ind})
        q_output = bjt.nodes[q_singleton]["output_beliefs"]

        self.assertAlmostEqual(
            float(np.exp(q_output[q_singleton].primitivize())), 0.9/0.1 * 0.8/0.2
        )

    def test_compile_grounded_loop(self):
        prog = Program()

        # logit(0.9) :: p :- not q.
        # logit(0.8) :: q :- not p.
        prog.add_rule(
            Rule(head=Literal("p", []), body=[Literal("q", [], naf=True)]), 0.9
        )
        prog.add_rule(
            Rule(head=Literal("q", []), body=[Literal("p", [], naf=True)]), 0.8
        )

        bjt = prog.compile()

        # Weight for {p} in the singleton node should be logit(0.9)+logit(0.8), and
        # should equal 0.9/0.1 * 0.8/0.2 when exp'ed
        p_ind = bjt.graph["atoms_map"][Literal("p", [])]
        p_singleton = frozenset({p_ind})
        p_output = bjt.nodes[p_singleton]["output_beliefs"]

        self.assertAlmostEqual(
            float(np.exp(p_output[p_singleton].primitivize())), 0.9/0.1 * 0.8/0.2
        )

    def test_compile_grounded_loop_unfounded(self):
        prog = Program()

        # logit(0.9) :: p :- q.
        # logit(0.8) :: q :- p.
        prog.add_rule(Rule(head=Literal("p", []), body=[Literal("q", [])]), 0.9)
        prog.add_rule(Rule(head=Literal("q", []), body=[Literal("p", [])]), 0.8)

        bjt = prog.compile()

        # No non-empty models should hold, empty graph
        self.assertEqual(len(bjt.graph["atoms_map"]), 0)

    def test_compile_single_nonabsolute_constraint(self):
        prog = Program()

        # logit(0.9) :: :- not p.
        prog.add_rule(Rule(body=[Literal("q", [], naf=True)]), 0.9)

        bjt = prog.compile()

        # No non-empty models should hold, empty graph
        self.assertEqual(len(bjt.graph["atoms_map"]), 0)

    def test_compile_single_absolute_constraint(self):
        prog = Program()

        # A :: :- not p.
        prog.add_absolute_rule(Rule(body=[Literal("p", [], naf=True)]))

        bjt = prog.compile()

        # Unsatisfiable, BJT should be None
        self.assertIsNone(bjt)

    def test_compile_grounded_composite(self):
        prog = Program()

        # logit(0.9) :: p.
        # logit(0.8) :: q.
        # logit(0.75) :: r.
        prog.add_rule(Rule(head=Literal("p", [])), 0.9)
        prog.add_rule(Rule(head=Literal("q", [])), 0.8)
        prog.add_rule(Rule(head=Literal("r", [])), 0.75)

        # logit(0.9) :: s :- p, not q.
        # logit(0.8) :: t :- q, not r.
        prog.add_rule(
            Rule(
                head=Literal("s", []),
                body=[Literal("p", []), Literal("q", [], naf=True)]
            ),
            0.9
        )
        prog.add_rule(
            Rule(
                head=Literal("t", []),
                body=[Literal("q", []), Literal("r", [], naf=True)]
            ),
            0.8
        )

        bjt = prog.compile()

        # Weight for {s} in the singleton node should equal 1296 when exp'ed
        s_ind = bjt.graph["atoms_map"][Literal("s", [])]
        s_singleton = frozenset({s_ind})
        s_output = bjt.nodes[s_singleton]["output_beliefs"]

        self.assertAlmostEqual(
            float(np.exp(s_output[s_singleton].primitivize())), 1296
        )
    
    def test_compile_generic_composite(self):
        prog = Program()

        # logit(0.9) :: p(0).
        # logit(0.8) :: q(0).
        # logit(0.75) :: r(0).
        prog.add_rule(Rule(head=Literal("p", [(0, False)])), 0.9)
        prog.add_rule(Rule(head=Literal("q", [(0, False)])), 0.8)
        prog.add_rule(Rule(head=Literal("r", [(0, False)])), 0.75)

        # logit(0.25) :: p(1).
        # logit(0.9) :: q(1).
        # logit(0.2) :: r(1).
        prog.add_rule(Rule(head=Literal("p", [(1, False)])), 0.25)
        prog.add_rule(Rule(head=Literal("q", [(1, False)])), 0.9)
        prog.add_rule(Rule(head=Literal("r", [(1, False)])), 0.2)

        # logit(0.9) :: s(X) :- p(X), not q(X).
        # logit(0.8) :: t(X) :- q(X), not r(X).
        prog.add_rule(
            Rule(
                head=Literal("s", [("X", True)]),
                body=[
                    Literal("p", [("X", True)]),
                    Literal("q", [("X", True)], naf=True)
                ]
            ),
            0.9
        )
        prog.add_rule(
            Rule(
                head=Literal("t", [("X", True)]),
                body=[
                    Literal("q", [("X", True)]),
                    Literal("r", [("X", True)], naf=True)
                ]
            ),
            0.8
        )

        bjt = prog.compile()

        # Weight for {t(1)} in the singleton node should equal 432 when exp'ed
        t1_ind = bjt.graph["atoms_map"][Literal("t", [(1, False)])]
        t1_singleton = frozenset({t1_ind})
        t1_output = bjt.nodes[t1_singleton]["output_beliefs"]

        self.assertAlmostEqual(
            float(np.exp(t1_output[t1_singleton].primitivize())), 432
        )
