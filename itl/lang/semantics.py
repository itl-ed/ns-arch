from collections import defaultdict

from delphin import ace, predicate


class SemanticParser:
    """
    Semantic parser that first processes free-form language inputs into MRS, and then
    translates them into ASP-friendly custom formats
    """
    def __init__(self, grammar, ace_bin):
        self.grammar = grammar
        self.ace_bin = ace_bin

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
        assign_fn = lambda: max(max(var_map.values())+1, 0)
        var_map = defaultdict(assign_fn)
        var_map[parse["index"]] = -float("inf")

        # Negated relations by handle
        negs = [rel["neg"] for rel in parse["relations"]["by_id"].values()
            if rel["predicate"] == "neg"]

        # Traverse the semantic dependency tree represented by parse to collect appropriately
        # structured set of ASP literals
        asp_content = _traverse_dt(parse, parse["index"], var_map, set(), negs)
        
        return asp_content, dict(var_map)


def _traverse_dt(parse, rel_id, var_map, covered, negs):
    """
    Recursive pre-order traversal of semantic dependency tree of referents (obtained from MRS)
    """
    # Collect 'relevant' relations that have rel_id as arg
    rel_rels = {args: rels for args, rels in parse["relations"]["by_args"].items()
        if rel_id in args}

    rel_rels_to_cover = []

    # This node is connected to nodes with ids not registered in var_map yet
    daughters = set()
    id_rels = []  # Track identity relations like "_be_v_id", which need special treatment

    for args, rels in rel_rels.items():
        for a in args:
            if a not in var_map and a != rel_id: daughters.add(a)

        for rel in rels:
            if rel["id"] not in covered:
                rel_rels_to_cover.append(rel)
                covered.add(rel["id"])

            if rel["predicate"] == "be":
                id_rels.append(rel["args"])
    
    # First handle identity relations by assigning them the same variable index
    for args in id_rels:
        assert len(args) >= 3  # (event, ref1, ref2)
        var_map[args[2]] = var_map[args[1]]
    
    # Then register the remaining referents unless already registered; disregard event
    # referents for stative verb predicates! (TODO: implement stative thing)
    for id in daughters:
        if id.startswith("x"):
            var_map[id]
        else:
            var_map[id] = -float("inf")  # For referents not considered as variables

    # Recursive calls to traversal function for collecting predicates specifying each daughter
    daughters = {id: _traverse_dt(parse, id, var_map, covered, negs) for id in daughters}

    # Return values of this function are flattened when called recursively
    daughters = {id: tp+fc for id, (tp, fc) in daughters.items()}

    # Return empty list if key not found; no further predicates found (i.e. leaf node)
    daughters = defaultdict(lambda: [], daughters)

    # Build return values; resp. topic & focus
    topic_lits = []; focus_lits = []

    for rel in rel_rels_to_cover:

        # Each list entry in topic_lits and focus_lits is a tuple of the following form:
        #   ('main' literal, list of condition literals, whether choice rule or not)
        # or a set of such tuples, for representing a negated scope over them. Basically
        # equivalent to conjunction normal form with haphazard formalism... I will have to
        # incorporate some static typing later, if I want to clean up this messy code
        #
        # Each literal is a tuple of the following form:
        #   (predicate, pos, list of arg variables)

        negated = rel["handle"] in negs

        # Handle predicates accordingly
        if rel["predicate"] == "be":
            # Predicates describing arg2 of 'be' contribute as rule head literals,
            # while others contribute as rule body literals
            topic_lits += daughters[rel["args"][1]]

            if negated:
                focus_lits.append(
                    (daughters[rel["args"][2]], "neg")
                )
            else:
                focus_lits += daughters[rel["args"][2]]

        elif rel["pos"] == "a":
            # Adjective predicates with args ('event', referent)
            topic_lits += daughters[rel["args"][1]]

            if negated:
                focus_lits.append(
                    ([((rel["predicate"], "a", [var_map[rel["args"][1]]]), [], False)], "neg")
                )
            else:
                focus_lits += [((rel["predicate"], "a", [var_map[rel["args"][1]]]), [], False)]

        elif rel["pos"] == "n":
            # Noun predicates with args (referent[, more optional referents])
            if negated:
                focus_lits.append(
                    ([((rel["predicate"], "n", [var_map[rel_id]]), [], False)], "neg")
                )
            else:
                focus_lits += [((rel["predicate"], "n", [var_map[rel_id]]), [], False)]

        elif rel["predicate"] == "have" or rel["predicate"] == "with":
            # Here we assume the meaning of rel["predicate"] has to do with holonymy/meronymy
            topic_lits += daughters[rel["args"][1]]

            arg2_head = parse["relations"]["by_id"][rel["args"][2]]

            if len(daughters[rel["args"][2]]) == 1:
                # Just the 'have(e,x1,x2)' is the only and main message here
                a2_lit = daughters[rel["args"][2]][0]
                a2a1_vars = [var_map[rel["args"][2]], var_map[rel["args"][1]]]

                main_lit = (a2_lit[0][0]+"_of", a2_lit[0][1], a2a1_vars)
                cond_lits = [("part_of", "", a2a1_vars)]

                if negated:
                    focus_lits.append(
                        ((main_lit, cond_lits, True), "neg")
                    )
                else:
                    focus_lits.append((main_lit, cond_lits, True))
            
            else:
                # 'have(e,x1,x2)' itself is presupposed, and modifications of x2 are
                # main messages here
                presup = []; assertions = []

                for a2_lit in daughters[rel["args"][2]]:

                    if a2_lit[0][0] == arg2_head["predicate"] and a2_lit[0][1] == arg2_head["pos"]:
                        # Presupposition of the said object having the part

                        # Presuppositions are not about the entities described by the sentence...
                        # Instead, consider hypothetical part entity
                        a2a1_vars = [var_map["hp"], var_map[rel["args"][1]]]

                        main_lit = (a2_lit[0][0]+"_of", a2_lit[0][1], a2a1_vars)
                        cond_lits = [("part_of", "", a2a1_vars)]
                        presup.append((main_lit, cond_lits, True))

                    else:
                        # Everything else
                        a2a1_vars = [var_map[rel["args"][2]], var_map[rel["args"][1]]]

                        main_lit = (a2_lit[0][0], a2_lit[0][1], [var_map[rel["args"][2]]])
                        cond_lits = [(arg2_head["predicate"]+"_of", arg2_head["pos"], a2a1_vars)]
                        assertions.append((main_lit, cond_lits, False))
                
                focus_lits += presup  # Note that the presupposition is not negated

                if negated:
                    focus_lits.append(
                        (assertions, "neg")
                    )
                else:
                    focus_lits += assertions

        else:
            # Other general cases
            continue

    return (topic_lits, focus_lits)
