import os
from collections import defaultdict

from delphin import ace, predicate


class SemanticParser:
    """
    Semantic parser that first processes free-form language inputs into MRS, and then
    translates them into ASP-friendly custom formats
    """
    def __init__(self, grammar, ace_bin):
        self.grammar = grammar
        self.ace_bin = os.path.join(ace_bin, "ace")

    def nl_parse(self, usr_in):
        parse = {
            "relations": {
                "by_args": defaultdict(lambda: []),
                "by_id": {},
                "by_handle": {}
            }
        }

        # For now use the top result
        parsed = ace.parse(self.grammar, usr_in, executable=self.ace_bin)
        parsed = parsed.result(0).mrs()

        # Extract essential parse, assuming flat structure (no significant semantic scoping)
        for ep in parsed.rels:

            args = {a: ep.args[a] for a in ep.args if a.startswith("ARG") and len(a)>3}
            args = tuple([ep.args[f"ARG{i}"] for i in range(len(args))])

            rel = {
                "args": args,
                "lexical": ep.predicate.startswith("_"),
                "id": ep.id,
                "handle": ep.label
            }

            if "ARG" in ep.args: rel["arg_udef"] = ep.args["ARG"]

            if ep.predicate.startswith("_"):
                # 'Surface' (in ERG parlance) predicates with lexical symbols
                lemma, pos, sense = predicate.split(ep.predicate)

                rel.update({
                    "predicate": lemma,
                    "pos": pos,
                    "sense": sense
                })
            
            else:
                # Reserved, 'abstract' (in ERG parlance) predicates
                if ep.is_quantifier():
                    rel.update({
                        "predicate": ep.predicate.strip("_q"),
                        "pos": "q",
                        "lexical": False
                    })

                else:
                    rel.update({
                        "predicate": ep.predicate,
                        "pos": None
                    })

                    if ep.predicate == "named":
                        rel["carg"] = ep.carg

                    if ep.predicate == "neg":
                        hcons = {hc.hi: hc.lo for hc in parsed.hcons if hc.relation=="qeq"}
                        negated = hcons[ep.args["ARG1"]]
                        negated = [ep for ep in parsed.rels if ep.label == negated][0]

                        rel["neg"] = negated.label
                
            parse["relations"]["by_args"][args].append(rel)
            parse["relations"]["by_id"][ep.id] = rel
            parse["relations"]["by_handle"][ep.label] = rel
        
        parse["relations"]["by_args"] = dict(parse["relations"]["by_args"])
        
        # SF supposedly stands for 'sentential force'
        parse["utt_type"] = parsed.variables[parsed.index]["SF"]

        # Record index (top variable)
        parse["index"] = parsed.index

        return parse

    @staticmethod    
    def asp_translate(parse):
        # First find chains of compound NPs and then appropriately merge them
        chains = {}
        for args in parse["relations"]["by_args"]:
            for rel in parse["relations"]["by_args"][args]:
                if rel["lexical"] == False and rel["predicate"] == "compound":
                    chains[args[2], args[0]] = args[1]
        
        while len(chains) > 0:
            for x in chains:
                if x[0] not in chains.values():
                    # Backtrack by one step to merge
                    p1 = parse["relations"]["by_id"][x[0]]
                    p2 = parse["relations"]["by_id"][chains[x]]
                    merged = p1["predicate"] + p2["predicate"].capitalize()
                    parse["relations"]["by_id"][chains[x]]["predicate"] = merged

                    del parse["relations"]["by_args"][p1["args"]]
                    del parse["relations"]["by_id"][p1["id"]]
                    del parse["relations"]["by_handle"][p1["handle"]]
                    del chains[x]; break

        # Equivalence classes mapping referents to rule variables
        assign_fn = lambda: max(max(ref_map.values())+1, 0)
        ref_map = defaultdict(assign_fn)
        ref_map[parse["index"]] = -float("inf")

        # Negated relations by handle
        negs = [rel["neg"] for rel in parse["relations"]["by_id"].values()
            if rel["predicate"] == "neg"]

        # Traverse the semantic dependency tree represented by parse to collect appropriately
        # structured set of ASP literals
        translation = _traverse_dt(parse, parse["index"], ref_map, set(), negs)

        # Reorganizing ref_map: not entirely necessary, just my OCD
        ref_map_map = defaultdict(lambda: len(ref_map_map))
        for ref in ref_map:
            ref_map[ref] = ref_map_map[ref_map[ref]]

        return translation, dict(ref_map)


def _traverse_dt(parse, rel_id, ref_map, covered, negs):
    """
    Recursive pre-order traversal of semantic dependency tree of referents (obtained from MRS)
    """
    # Collect 'relevant' relations that have rel_id as arg
    rel_rels = {args: rels for args, rels in parse["relations"]["by_args"].items()
        if rel_id in args}

    rel_rels_to_cover = []

    # This node is connected to nodes with ids not registered in ref_map yet
    daughters = set()

    for args, rels in rel_rels.items():
        for a in args:
            if a not in ref_map and a != rel_id: daughters.add(a)

        for rel in rels:
            if rel["id"] not in covered:
                rel_rels_to_cover.append(rel)
                covered.add(rel["id"])
    
    # Then register the remaining referents unless already registered; disregard event
    # referents for stative verb predicates! (TODO?: implement stative thing)
    for id in daughters:
        if id.startswith("x"):
            ref_map[id]
        else:
            ref_map[id] = -float("inf")  # For referents not considered as variables

    # Recursive calls to traversal function for collecting predicates specifying each daughter
    daughters = {id: _traverse_dt(parse, id, ref_map, covered, negs) for id in daughters}

    # Return values of this function are flattened when called recursively
    daughters = {id: tp+fc for id, (tp, fc) in daughters.items()}

    # Return empty list if key not found; no further predicates found (i.e. leaf node)
    daughters = defaultdict(lambda: [], daughters)

    # Build return values; resp. topic & focus
    topic_msgs = []; focus_msgs = []

    for rel in rel_rels_to_cover:

        # Each list entry in topic_msgs and focus_msgs is a tuple of the following form:
        #   ('main' literal, list of condition literals, whether choice rule or not)
        # or a set of such tuples, for representing a negated scope over them. Basically
        # equivalent to conjunction normal form with haphazard formalism... I will have to
        # incorporate some static typing later, if I want to clean up this messy code
        #
        # Each literal is a tuple of the following form:
        #   (predicate, pos, list of arg variables, source rel id)

        negated = rel["handle"] in negs

        # Handle predicates accordingly
        if rel["predicate"] == "be":
            # "be" has different semantics depending on whether the sentence is a generic
            # one or not.

            # Here we test by checking if arg1/arg2 is bare NP; that is, quantified by udef_q.
            # This is of course a very crude test, and may be elaborated later to account for
            # other generic sentence signals: e.g. universal quantifiers.
            bare_arg1 = any([
                rel["pos"] == "q" and rel["predicate"] == "udef"
                for rel in parse["relations"]["by_args"][(rel["args"][1],)]
            ])
            bare_arg2 = any([
                rel["pos"] == "q" and rel["predicate"] == "udef"
                for rel in parse["relations"]["by_args"][(rel["args"][2],)]
            ])
            is_generic = bare_arg1 and bare_arg2

            if is_generic:
                # Predicates describing arg2 of 'be' contribute as rule head literals,
                # while others contribute as rule body literals

                topic_msgs += daughters[rel["args"][1]]

                if negated:
                    focus_msgs.append(
                        (daughters[rel["args"][2]], rel["id"])
                    )
                else:
                    focus_msgs += daughters[rel["args"][2]]

                # Ensure referent identity within the rule at this point
                ref_map[rel["args"][2]] = ref_map[rel["args"][1]]

            else:
                # For non-generic sentences, "be" has semantics of referent identity. Here
                # we add a special reserved predicate named "=", which will be valuated later
                # along with other predicates.
                topic_msgs += daughters[rel["args"][1]]
                focus_msgs += daughters[rel["args"][2]]

                a1a2_vars = [rel["args"][1], rel["args"][2]]
                rel_lit = (("=", "*", a1a2_vars, rel["id"]), [], False)

                if negated:
                    focus_msgs.append(([rel_lit], rel["id"]))
                else:
                    focus_msgs.append(rel_lit)

        elif rel["pos"] == "a":
            # Adjective predicates with args ('event', referent)
            topic_msgs += daughters[rel["args"][1]]

            rel_lit = ((rel["predicate"], "a", [rel["args"][1]], rel["id"]), [], False)

            if negated:
                focus_msgs.append(([rel_lit], rel["id"]))
            else:
                focus_msgs.append(rel_lit)

        elif rel["pos"] == "n":
            # Noun predicates with args (referent[, more optional referents])
            rel_lit = ((rel["predicate"], "n", [rel_id], rel["id"]), [], False)

            if negated:
                focus_msgs.append(([rel_lit], rel["id"]))
            else:
                focus_msgs.append(rel_lit)

        elif rel["predicate"] == "have" or rel["predicate"] == "with":
            # Here we assume the meaning of rel["predicate"] has to do with holonymy/meronymy
            topic_msgs += daughters[rel["args"][1]]

            arg2_head = parse["relations"]["by_id"][rel["args"][2]]

            if len(daughters[rel["args"][2]]) == 1:
                # Just the 'have(e,x1,x2)' is the only and main message here
                a2_lit = daughters[rel["args"][2]][0]
                a1a2_vars = [rel["args"][1], rel["args"][2]]
                cond_lits = [("have", "v", a1a2_vars, rel["id"])]

                rel_lit = (a2_lit[0], cond_lits, True)

                if negated:
                    focus_msgs.append(([rel_lit], rel["id"]))
                else:
                    focus_msgs.append(rel_lit)
            
            else:
                # 'have(e,x1,x2)' itself is presupposed, and modifications of x2 are
                # main messages here
                presup = []; assertions = []

                for a2_lit in daughters[rel["args"][2]]:

                    a1a2_vars = [rel["args"][1], rel["args"][2]]

                    if a2_lit[0][0] == arg2_head["predicate"] and a2_lit[0][1] == arg2_head["pos"]:
                        # Presupposition of the said object having the part
                        cond_lits = [("have", "v", a1a2_vars, rel["id"])]
                        presup.append((a2_lit[0], cond_lits, True))

                    else:
                        # Everything else
                        cond_lits = [
                            (arg2_head["predicate"], arg2_head["pos"], [rel["args"][2]], arg2_head["id"]),
                            ("have", "v", a1a2_vars, rel["id"])
                        ]
                        assertions.append((a2_lit[0], cond_lits, False))

                # Presuppositions may or may not be part of the info conveyed... Perhaps to
                # a weaker extent? E.g. Does "France does not have a bald king" imply France
                # has a king? How about "Does France have a bald king?"? It may serve as a
                # reason to infer so, but not strongly... Let's think about this more later...
                if negated:
                    focus_msgs.append((assertions, rel["id"]))
                else:
                    if parse["utt_type"] != "ques":
                        focus_msgs += presup
                    focus_msgs += assertions

        else:
            # Other general cases
            continue

    return (topic_msgs, focus_msgs)
