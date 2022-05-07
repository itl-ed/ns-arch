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

        # Equivalence classes mapping referents to rule variables, while tracking whether
        # they come from referential expressions or not
        ref_map = {}
        ref_map[parse["index"]] = None

        # Negated relations by handle
        negs = [rel["neg"] for rel in parse["relations"]["by_id"].values()
            if rel["predicate"] == "neg"]

        # Traverse the semantic dependency tree represented by parse to collect appropriately
        # structured set of ASP literals
        translation = _traverse_dt(parse, parse["index"], ref_map, set(), negs)

        # We assume bare NPs (underspecified quant.) have universal reading when arg1 of index
        # (and existential reading otherwise)
        is_bare = lambda rf: any([
            r["pos"] == "q" and r["predicate"] == "udef"
            for r in parse["relations"]["by_args"][(rf,)]
        ])
        index_arg1 = parse["relations"]["by_id"][parse["index"]]["args"][1]
        if is_bare(index_arg1):
            ref_map[index_arg1]["is_univ_quantified"] = True

        # Reorganizing ref_map: not entirely necessary, just my OCD
        ref_map_map = defaultdict(lambda: len(ref_map_map))
        for ref in ref_map:
            if ref_map[ref] is not None:
                ref_map[ref] = {
                    "map_id": ref_map_map[ref_map[ref]["map_id"]],
                    "is_referential": ref_map[ref]["is_referential"],
                    "is_univ_quantified": ref_map[ref]["is_univ_quantified"],
                    "is_wh_quantified": ref_map[ref]["is_wh_quantified"]
                }

        return translation, dict(ref_map)


def _traverse_dt(parse, rel_id, ref_map, covered, negs):
    """
    Recursive pre-order traversal of semantic dependency tree of referents (obtained from MRS)
    """
    # Helper method that checks whether a discourse referent is introduced by a referential
    # expression | universally quantified | wh-quantified
    is_referential = lambda rf: any([
        (r["pos"] == "q" and r["predicate"] == "the") \
            or (r["pos"] == "q" and r["predicate"] == "pronoun") \
            or (r["pos"] is None and r["predicate"] == "named") \
            or (r["pos"] == "q" and "sense" in r and r["sense"] == "dem")
        for r in parse["relations"]["by_args"][(rf,)]
    ])
    is_univ_quantified = lambda rf: any([
        r["pos"] == "q" and r["predicate"] in {"all", "every"}
        for r in parse["relations"]["by_args"][(rf,)]
    ])
    is_wh_quantified = lambda rf: any([
        r["pos"] == "q" and r["predicate"] == "which"
        for r in parse["relations"]["by_args"][(rf,)]
    ])

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
            ref_map[id] = {
                "map_id": max(
                    max(
                        [v["map_id"] for v in ref_map.values() if v is not None],
                        default=-1
                    )+1, 0
                ),
                "is_referential": is_referential(id),
                "is_univ_quantified": is_univ_quantified(id),
                "is_wh_quantified": is_wh_quantified(id)
            }
        else:
            # For referents we are not interested about
            ref_map[id] = None

    # Recursive calls to traversal function for collecting predicates specifying each daughter
    daughters = {id: _traverse_dt(parse, id, ref_map, covered, negs) for id in daughters}

    # Return values of this function are flattened when called recursively
    daughters = {id: tp+fc for id, (tp, fc) in daughters.items()}

    # Return empty list if key not found; no further predicates found (i.e. leaf node)
    daughters = defaultdict(lambda: [], daughters)

    # Build return values; resp. topic & focus
    topic_msgs = []; focus_msgs = []

    for rel in rel_rels_to_cover:
        # Each list entry in topic_msgs and focus_msgs is a literal, which is a tuple:
        #   1) predicate
        #   2) part-of-speech
        #   3) list of arg variables (args may be function terms if skolemized)
        # or a list of such literals, for representing a negated scope over them. Basically
        # equivalent to conjunction normal form with haphazard formalism... I will have to
        # incorporate some static typing later, if I want to clean up this messy code        

        # Setting up arg1 (& arg2) variables
        rel_args = rel["args"]
        if len(rel_args) > 1 and rel_args[1].startswith("i"):
            # Let's deal with i-referents later... Just ignore it for now
            rel_args = rel_args[:1]

        if len(rel_args) > 1:
            if len(rel_args) > 2:
                if rel["predicate"] == "be" and is_wh_quantified(rel_args[2]):
                    # MRS flips the arg order when subject is quantified with 'which',
                    # presumably seeing it as wh-movement?
                    arg1 = rel_args[2]
                    arg2 = rel_args[1]
                else:
                    # Default case
                    arg1 = rel_args[1]
                    arg2 = rel_args[2]

                referential_arg1 = ref_map[arg1]["is_referential"]
                referential_arg2 = ref_map[arg2]["is_referential"]

            else:
                # (These won't & shouldn't be referred)
                arg1 = rel_args[1]
                arg2 = None
                referential_arg1 = ref_map[arg1]["is_referential"]
                referential_arg2 = None

            # We assume predicates describing arg1 are topic of the covered constituents
            # (This will do for most cases... May later need to refine depending on voice,
            # discourse context, etc.?)
            topic_msgs += daughters[arg1]
        else:
            # (These won't & shouldn't be referred)
            arg1 = None
            arg2 = None
            referential_arg1 = None
            referential_arg2 = None

        negated = rel["handle"] in negs

        # Handle predicates accordingly
        if rel["pos"] == "a":
            # Adjective predicates with args ('event', referent)
            rel_lit = (rel["predicate"], "a", [arg1])

            if negated:
                focus_msgs.append([rel_lit])
            else:
                focus_msgs.append(rel_lit)

        elif rel["pos"] == "n":
            # Noun predicates with args (referent[, more optional referents])
            rel_lit = (rel["predicate"], "n", [rel_id])

            if negated:
                focus_msgs.append([rel_lit])
            else:
                focus_msgs.append(rel_lit)
        
        elif rel["predicate"] == "be":
            # "be" has different semantics depending on the construction of the clause:
            #   1) Predicational: Non-referential complement
            #   2) Equative: Referential subject, referential complement
            #   3) Specificational: Non-referential subject, referential complement
            # where referential NP is of type <e>, and non-referential NP is of type <e,t>.
            #
            # (* Technically speaking, the dominant taxonomy in the field for copula semantics
            # seems that predicational clauses need referential subjects, and quantificational
            # expressions are allowed as subject as well. But testing these conditions is a bit
            # hairy, and for now we will go with a simplified condition of having non-referential
            # complement only.)

            if not referential_arg2:
                # Predicational; provide predicates as additional info about subject
                if negated:
                    focus_msgs.append(daughters[arg2])
                else:
                    focus_msgs += daughters[arg2]

                # arg1 and arg2 are the same entity
                ref_map[arg2] = ref_map[arg1]

            else:
                if referential_arg1:
                    # Equative; claim referent identity btw. subject and complement
                    topic_msgs += daughters[arg2]

                    a1a2_vars = [arg1, arg2]
                    rel_lit = ("=", "*", a1a2_vars)

                    if negated:
                        focus_msgs.append([rel_lit])
                    else:
                        focus_msgs.append(rel_lit)

                else:
                    # We are not really interested in 3) (or others?), at least for now
                    raise ValueError("Weird sentence with copula")

        elif rel["predicate"] == "have":
            # (Here we assume the meaning of rel["predicate"] has to do with holonymy/meronymy)
            # Two-place predicates have different semantics depending on referentiality of arg2

            if referential_arg2:
                # Simpler case; use constant for arg2, and provided predicates concerning arg2
                # are considered topic message
                topic_msgs += daughters[arg2]

                rel_lit = ("have", "v", [arg1, arg2])

                if negated:
                    focus_msgs.append([rel_lit])
                else:
                    focus_msgs.append(rel_lit)

            else:
                # arg2 is existentially quantified; skolemize the referent (ensuring ASP program
                # safety) and yield conjunction of literals. Note that we are assuming any quantifier
                # scope ambiguity is resolved such that existentials are placed inside: i.e. surface
                # scope reading.
                arg2_sk = (f"f_{arg2}", (arg1,))
                ref_map[arg2_sk] = ref_map[arg2]
                del ref_map[arg2]

                lits = [("have", "v", [arg1, arg2_sk])]
                for a2_lit in daughters[arg2]:
                    # Replace occurrences of arg2 with the skolem term
                    args_sk = [arg2_sk if arg2==a else a for a in a2_lit[2]]
                    a2_lit_sk = a2_lit[:2] + (args_sk,)
                    lits.append(a2_lit_sk)
                
                if negated:
                    focus_msgs.append(lits)
                else:
                    focus_msgs += lits

        else:
            # Other general cases
            continue

    return (topic_msgs, focus_msgs)
