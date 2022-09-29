import os
import re
import string
from collections import defaultdict

from delphin import ace, predicate, codecs


class SemanticParser:
    """
    Semantic parser that first processes free-form language inputs into MRS, and then
    translates them into ASP-friendly custom formats
    """
    def __init__(self, grammar, ace_bin):
        self.grammar = grammar
        self.ace_bin = os.path.join(ace_bin, "ace")

        # For redirecting ACE's stderr (so that messages don't print on console)
        self.null_sink = open(os.devnull,"w")
    
    def __dell__(self):
        """ For closing the null sink before destruction """
        self.null_sink.close()
        super().__del__()

    def nl_parse(self, usr_in):
        parse = {
            "relations": {
                "by_args": defaultdict(lambda: []),
                "by_id": {},
                "by_handle": {}
            },
            "utt_type": {},
            "raw": usr_in,
            "conjunct_raw": {}
        }

        # For now use the top result
        parsed = ace.parse(
            self.grammar, usr_in, executable=self.ace_bin, stderr=self.null_sink
        )
        parsed = parsed.result(0).mrs()

        # Extract essential parse, assuming flat structure (no significant semantic scoping)
        for ep in parsed.rels:

            args = {a: ep.args[a] for a in ep.args if a.startswith("ARG") and len(a)>3}
            args = tuple([ep.args[f"ARG{i}"] for i in range(len(args))])

            rel = {
                "args": args,
                "lexical": ep.predicate.startswith("_"),
                "id": ep.id,
                "handle": ep.label,
                "crange": (ep.cfrom, ep.cto)
            }

            if "ARG" in ep.args: rel["arg_udef"] = ep.args["ARG"]

            if ep.predicate.startswith("_"):
                # 'Surface' (in ERG parlance) predicates with lexical symbols
                lemma, pos, sense = predicate.split(ep.predicate)

                if sense == "unknown":
                    # Predicate not covered by parser lexicon
                    lemma, pos = lemma.split("/")

                    # Multi-clause input occasionally includes '.' after unknown nouns
                    lemma = lemma.strip(".")

                    # Translate the tag obtained from the POS tagger to corresponding
                    # MRS POS code
                    if pos.startswith("n"):
                        pos = "n"
                    elif pos.startswith("j"):
                        pos = "a"
                    elif pos.startswith("v"):
                        pos = "v"
                    else:
                        raise ValueError(f"I don't know how to grammatically process the token '{lemma}'")

                # (R)MRS pos codes to WordNet synset pos tags
                if pos == "p": pos = "r"

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
        
        # SF supposedly stands for 'sentential force' -- handle the utterance types
        # here, along with conjunctions
        def index_conjuncts_with_SF(rel_id, parent_id):
            index_rel = parse["relations"]["by_id"][rel_id]
            if index_rel["predicate"] == "implicit_conj":
                # Recursively process conjuncts
                index_conjuncts_with_SF(index_rel["args"][1], rel_id)
                index_conjuncts_with_SF(index_rel["args"][2], rel_id)
            else:
                parse["utt_type"][rel_id] = parsed.variables[rel_id]["SF"]

                # Recover part of original input corresponding to this conjunct by
                # iteratively expanding set of args covered by this conjunct, then
                # obtaining min/max of the cranges
                covered_args = {rel_id}; fixpoint_reached = False
                while not fixpoint_reached:
                    prev_size = len(covered_args)
                    covered_args = set.union(*[
                        set(args) for args in parse["relations"]["by_args"]
                        if parent_id not in args and any(a in covered_args for a in args)
                    ])
                    fixpoint_reached = len(covered_args) == prev_size
                covered_cranges = [
                    parse["relations"]["by_id"][arg]["crange"]
                    for arg in covered_args if arg in parse["relations"]["by_id"]
                ]
                cj_start = min(cfrom for cfrom, _ in covered_cranges)
                cj_end = max(cto for _, cto in covered_cranges)
                parse["conjunct_raw"][rel_id] = parse["raw"][cj_start:cj_end]
        index_conjuncts_with_SF(parsed.index, None)

        # Record index (top variable)
        parse["index"] = parsed.index

        return parse

    def nl_negate(self, sentence):
        """
        Return the negated version of the sentence by parsing, manipulating MRS, then
        generating with grammar
        """
        # MRS parse
        parsed = ace.parse(
            self.grammar, sentence, executable=self.ace_bin, stderr=self.null_sink
        )
        parsed = parsed.result(0).mrs()

        # pydelphin EP and HCons constructor shoplifted
        EP_Class = type(parsed.rels[0])
        HCons_Class = type(parsed.hcons[0])

        # Find appropriate indices for new variables
        max_var_ind = max(int("".join(filter(str.isdigit, v))) for v in parsed.variables)
        handle_hi = max_var_ind+1
        handle_lo = max_var_ind+2
        ev_var_neg = max_var_ind+3

        # New handle constraint list
        hcons_orig_top = [hc for hc in parsed.hcons if hc.hi == parsed.top][0]
        hcons_kept = [hc for hc in parsed.hcons if hc.hi != parsed.top]
        hcons_hi = HCons_Class(parsed.top, "qeq", f"h{handle_hi}")
        hcons_lo = HCons_Class(f"h{handle_lo}", "qeq", hcons_orig_top.lo)
        new_hcons = hcons_kept + [hcons_hi, hcons_lo]

        if [r for r in parsed.rels if r.label==hcons_orig_top.lo][0].predicate == "neg":
            # Return sentence as-is if it already has negative polarity
            return sentence

        # New EP object for neg predicate
        neg_args = {"ARG0": f"e{ev_var_neg}", "ARG1": f"h{handle_lo}"}
        neg_EP = EP_Class("neg", f"h{handle_hi}", args=neg_args)

        parsed.rels.append(neg_EP)
        parsed.hcons = new_hcons

        # Generate with grammar
        generated = ace.generate(
            self.grammar, codecs.simplemrs.encode(parsed),
            executable=self.ace_bin, stderr=self.null_sink
        )
        try:
            replaced = generated.result(0)["surface"]
        except IndexError:
            # In unfortunate cases the predicate is not in ERG lexicon. Fall back
            # to passing val as named proper noun...
            unk_rels = [r for r in parsed.rels if r.predicate.endswith("unknown")]
            for r in unk_rels:
                r.args["CARG"] = r.predicate.split("/")[0][1:]
                r.predicate = "named"

            generated = ace.generate(
                self.grammar, codecs.simplemrs.encode(parsed),
                executable=self.ace_bin, stderr=self.null_sink
            )
            replaced = generated.result(0)["surface"]

        return replaced

    def nl_change_sf(self, sentence, new_SF):
        """
        Return the version of the sentence with specified sentential force (SF) by parsing,
        manipulating MRS, then generating with grammar
        """
        assert new_SF == "prop" or new_SF == "ques"

        # MRS parse
        parsed = ace.parse(
            self.grammar, sentence, executable=self.ace_bin, stderr=self.null_sink
        )
        parsed = parsed.result(0).mrs()

        # Apply new SF provided
        parsed.variables[parsed.index]["SF"] = new_SF
        
        # Generate with grammar
        generated = ace.generate(
            self.grammar, codecs.simplemrs.encode(parsed),
            executable=self.ace_bin, stderr=self.null_sink
        )
        try:
            replaced = generated.result(0)["surface"]
        except IndexError:
            # In unfortunate cases the predicate is not in ERG lexicon. Fall back
            # to passing val as named proper noun...
            unk_rels = [r for r in parsed.rels if r.predicate.endswith("unknown")]
            for r in unk_rels:
                r.args["CARG"] = r.predicate.split("/")[0][1:]
                r.predicate = "named"

            generated = ace.generate(
                self.grammar, codecs.simplemrs.encode(parsed),
                executable=self.ace_bin, stderr=self.null_sink
            )
            replaced = generated.result(0)["surface"]

        return replaced
    
    def nl_replace_wh(self, sentence, replace_targets, replace_values):
        # MRS parse
        parsed = ace.parse(
            self.grammar, sentence, executable=self.ace_bin, stderr=self.null_sink
        )
        parsed = parsed.result(0).mrs()

        # MRS flips the arg order when subject is quantified with 'which',
        # presumably seeing it as wh-movement? Re-flip, except when the
        # wh-word is 'what' (represented as 'which thing' in MRS)
        index_rel = [r for r in parsed.rels if parsed.index==r.id][0]
        index_sf = parsed.variables[parsed.index]["SF"]
        if index_rel.predicate=="_be_v_id" and index_sf=="ques":
            a2_rels = [r.predicate for r in parsed.rels if index_rel.args["ARG2"]==r.iv]
            a2_wh_quantified = ("which_q" in a2_rels) or ("_which_q" in a2_rels)

            if a2_wh_quantified and ("thing" not in a2_rels):
                index_rel.args["ARG1"], index_rel.args["ARG2"] = \
                    index_rel.args["ARG2"], index_rel.args["ARG1"]

        # pydelphin EP and HCons constructor shoplifted
        EP_Class = type(parsed.rels[0])
        HCons_Class = type(parsed.hcons[0])

        hcons_h2l = {h.hi: h.lo for h in parsed.hcons}

        # Find appropriate indices for new variables
        max_var_ind = max(int("".join(filter(str.isdigit, v))) for v in parsed.variables)

        for tgt, (val, is_named) in zip(replace_targets, replace_values):
            # Recover target referent by id
            tgt = [
                r for r in parsed.rels
                if (r.cfrom, r.cto)==tgt and r.id[0]=="x"
            ][0].id

            # The wh-quantifying relation
            wh_i, wh_rel = [
                (i, r) for i, r in enumerate(parsed.rels)
                if r.iv==tgt and r.predicate.endswith("which_q")
            ][0]

            # Find relations & hcons to be replaced in the new MRS, sweeping down the
            # MRS starting from wh_rel down to all children
            rels_to_replace = {wh_i}
            encountered_referents = {tgt}
            encountered_handles = {wh_rel.args["RSTR"], hcons_h2l[wh_rel.args["RSTR"]]}
            sweep_frontier = [hcons_h2l[wh_rel.args["RSTR"]]]

            while len(sweep_frontier) > 0:
                h = sweep_frontier.pop()

                # Relations with the handle being explored
                h_rels = {i: r for i, r in enumerate(parsed.rels) if r.label==h}

                # Update relations to replace
                rels_to_replace |= set(h_rels.keys())

                # Newly encountered referents, and update set of encountered ones
                new_referents = set.union(*[
                    {a for n, a in r.args.items() if n.startswith("ARG")}
                    for r in h_rels.values()
                ]) - encountered_referents
                encountered_referents |= new_referents

                # Expand frontier with labels of rels that have newly encountered
                # referents as argument
                sweep_frontier += [
                    hcons_h2l[r.label] for r in parsed.rels
                    if len(new_referents & set(r.args.values())) > 0 and r.label!=h
                ]
                encountered_handles |= set(sweep_frontier)
            
            hcons_to_replace = {i: {h.hi, h.lo} for i, h in enumerate(parsed.hcons)}
            hcons_to_replace = {
                i for i, hs in hcons_to_replace.items()
                if len(encountered_handles & hs) > 0
            }

            # Filter out rels/hcons to be replaced
            for i in reversed(range(len(parsed.rels))):
                if i in rels_to_replace: del parsed.rels[i]
            for i in reversed(range(len(parsed.hcons))):
                if i in hcons_to_replace: del parsed.hcons[i]

            # 'Seed' relations to be added, no matter how long the camelCased split is
            handle_lbl = f"h{max_var_ind+1}"
            handle_rstr = f"h{max_var_ind+2}"
            handle_body = f"h{max_var_ind+3}"
            handle_n = f"h{max_var_ind+4}"
            max_var_ind += 4

            q_args = {
                "ARG0": tgt, "RSTR": handle_rstr, "BODY": handle_body
            }
            if val is None:
                # Nothing is answer
                q_EP = EP_Class("_no_q", handle_lbl, args=q_args)
                n_EP = EP_Class("thing", handle_n, args={"ARG0": tgt})

                splits = []
            else:
                if is_named:
                    # Named referent quantified by proper_q
                    q_EP = EP_Class("proper_q", handle_lbl, args=q_args)

                    # Short name, no camelCase splits
                    splits = [val]
                else:
                    # Common noun referent quantified by _a_q
                    # (It will attach the indefinte 'a' to uncountable nouns as well,
                    # let's keep it this way for now)
                    q_EP = EP_Class("_a_q", handle_lbl, args=q_args)

                    # Check if string val is camelCased and needs to be split
                    splits = re.findall(r"(?:^|[A-Z])(?:[a-z]+|[A-Z]*(?=[A-Z]|$))", val)
                    splits = [w[0].lower()+w[1:] for w in splits]
                
                n_args = {"ARG0": tgt, "CARG": splits[-1]}
                n_EP = EP_Class("named", handle_n, args=n_args)

            hcons_add = HCons_Class(handle_rstr, "qeq", handle_n)

            parsed.rels.extend([q_EP, n_EP])
            parsed.hcons.append(hcons_add)

            # Handle compound nouns by adding appropriate relations & hcons in order
            prev_ref_n = tgt; prev_handle_n = handle_n
            for w in splits[:-1]:
                ref_c = f"e{max_var_ind+1}"
                ref_n = f"x{max_var_ind+2}"
                handle_lbl = f"h{max_var_ind+3}"
                handle_rstr = f"h{max_var_ind+4}"
                handle_body = f"h{max_var_ind+5}"
                handle_n = f"h{max_var_ind+6}"
                max_var_ind += 6

                q_args = {
                    "ARG0": ref_n, "RSTR": handle_rstr, "BODY": handle_body
                }
                q_EP = EP_Class("proper_q", handle_lbl, args=q_args)

                n_args = {"ARG0": ref_n, "CARG": w}
                n_EP = EP_Class("named", handle_n, args=n_args)

                c_args = {"ARG0": ref_c, "ARG1": prev_ref_n, "ARG2": ref_n}
                c_EP = EP_Class("compound", prev_handle_n, args=c_args)

                hcons_add = HCons_Class(handle_rstr, "qeq", handle_n)

                parsed.rels.extend([q_EP, n_EP, c_EP])
                parsed.hcons.append(hcons_add)

                prev_ref_n = ref_n
                prev_handle_n = handle_n

            # Generate with grammar
            generated = ace.generate(
                self.grammar, codecs.simplemrs.encode(parsed),
                executable=self.ace_bin, stderr=self.null_sink
            )
            replaced = generated.result(0)["surface"]

            # Weird behaviour of ACE ERG generator; for some reason, 'indefinite
            # named' nouns in a non-trivially manipulated MRS often get surface
            # form trailing "an" even if it doesn't start with a vowel... Patch
            # this by manual replacement
            if len(splits) > 0:
                if splits[0][0] not in 'aeiou':
                    replaced = replaced.replace(f"an {splits[0]}", f"a {splits[0]}")
                if splits[0][0] in 'aeiou':
                    replaced = replaced.replace(f"a {splits[0]}", f"an {splits[0]}")

        return replaced

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

                    args_to_del = []
                    for args, rels in parse["relations"]["by_args"].items():
                        if x[0] in args:
                            args_to_del.append(args)
                            for r in rels:
                                del parse["relations"]["by_id"][r["id"]]
                                del parse["relations"]["by_handle"][r["handle"]]
                    for args in args_to_del:
                        del parse["relations"]["by_args"][args]

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

        # We assume bare NPs (underspecified quant.) have universal reading when they are arg1
        # of index (and existential reading otherwise)
        is_bare = lambda rf: any([
            r["pos"] == "q" and r["predicate"] == "udef"
            for r in parse["relations"]["by_args"][(rf,)]
        ])
        for ev_id in translation:
            ev_arg1 = parse["relations"]["by_id"][ev_id]["args"][1]
            if is_bare(ev_arg1):
                ref_map[ev_arg1]["is_univ_quantified"] = True
        for ref in ref_map:
            # Handle function terms as well
            if isinstance(ref, tuple):
                # If all function term args are universally quantified
                if all(ref_map[fa]["is_univ_quantified"] for fa in ref[1]):
                    ref_map[ref]["is_univ_quantified"] = True

        # Reorganizing ref_map: not entirely necessary, just my OCD
        ref_map_map = defaultdict(lambda: len(ref_map_map))
        for ref in ref_map:
            if ref_map[ref] is not None:
                ref_map[ref] = {
                    "map_id": ref_map_map[ref_map[ref]["map_id"]],
                    "provenance": ref_map[ref]["provenance"],
                    "is_referential": ref_map[ref]["is_referential"],
                    "is_univ_quantified": ref_map[ref]["is_univ_quantified"],
                    "is_wh_quantified": ref_map[ref]["is_wh_quantified"],
                    "is_pred": ref_map[ref]["is_pred"]
                }

        return translation, dict(ref_map)


def _traverse_dt(parse, rel_id, ref_map, covered, negs):
    """
    Recursive pre-order traversal of semantic dependency tree of referents (obtained from MRS)
    """
    # Handle conjunction
    if rel_id in parse["relations"]["by_id"]:
        if parse["relations"]["by_id"][rel_id]["predicate"] == "implicit_conj":
            # Conjunction arg1 & arg2 id
            conj_args = parse["relations"]["by_id"][rel_id]["args"]

            # Make sure parts of arg1/arg2 are each processed only by the first/second pass
            ref_map[conj_args[1]] = None
            ref_map[conj_args[2]] = None
            covered.add(rel_id)
            
            a1_out = _traverse_dt(parse, conj_args[1], ref_map, covered, negs)
            a2_out = _traverse_dt(parse, conj_args[2], ref_map, covered, negs)

            return {**a1_out, **a2_out}

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
                "provenance": parse["relations"]["by_id"][id]["crange"],
                    # Referent's source expression in original utterance, represented as char range
                "is_referential": is_referential(id),
                "is_univ_quantified": is_univ_quantified(id),
                "is_wh_quantified": is_wh_quantified(id),
                "is_pred": False
                    # Whether it is a predicate (False; init default) or individual (True)
            }
        else:
            # For referents we are not interested about
            ref_map[id] = None

    # Recursive calls to traversal function for collecting predicates specifying each daughter
    daughters = {id: _traverse_dt(parse, id, ref_map, covered, negs) for id in daughters}

    # Return values of this function are flattened when called recursively
    daughters = {
        id: sum([tp+fc for tp, fc in outs.values()], []) for id, outs in daughters.items()
    }

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
            if len(rel_args) > 2 and rel_args[2].startswith("i"):
                # Let's deal with i-referents later... Just ignore it for now
                rel_args = rel_args[:2]

            if len(rel_args) > 2:
                if rel["predicate"] == "be" and is_wh_quantified(rel_args[2]):
                    # MRS flips the arg order when subject is quantified with 'which',
                    # presumably seeing it as wh-movement? Re-flip, except when the
                    # wh-word is 'what' (represented as 'which thing' in MRS)
                    if parse["relations"]["by_id"][rel_args[2]]["predicate"] == "thing":
                        arg1 = rel_args[1]
                        arg2 = rel_args[2]
                    else:
                        arg1 = rel_args[2]
                        arg2 = rel_args[1]

                    referential_arg1 = ref_map[arg1]["is_referential"]
                    referential_arg2 = ref_map[arg2]["is_referential"]

                elif ref_map[rel_args[1]] is None:
                    # Strong indication of passive participial adjective (e.g. flared)
                    arg1 = rel_args[2]
                    arg2 = None
                    referential_arg1 = ref_map[arg1]["is_referential"]
                    referential_arg2 = None
                    rel["pos"] = "a"
                    rel["predicate"] = parse["raw"][rel["crange"][0]:rel["crange"][1]]
                    rel["predicate"] = rel["predicate"].translate(
                        str.maketrans("", "", string.punctuation)
                    )       # There may be trailing punctuations, remove them

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

        negate_focus = rel["handle"] in negs

        # For negative polar (yes/no) question we don't negate focus message
        if negate_focus and rel_id==parse["index"] and parse["utt_type"]=="ques":
            is_polar_q = not any(
                v["is_wh_quantified"] for v in ref_map.values() if v is not None
            )
            if is_polar_q:
                negate_focus = False

        # Handle predicates accordingly
        if rel["pos"] == "a":
            # Adjective predicates with args ('event', referent)

            # (Many) Adjectives require consideration of the nominal 'class' of the
            # object they modify for proper interpretation of their semantics (c.f.
            # small elephant vs. big flea). For now, we will attach such nominal
            # predicate names after (all) adjectival predicate names; practically,
            # this allows the agent to search for exemplars within its exemplar-base
            # with added info.
            pred_name = rel["predicate"]
            modified_ent = parse["relations"]["by_id"][arg1]
            if modified_ent["lexical"]:
                pred_name += "/" + modified_ent["predicate"]
            rel_lit = (pred_name, "a", [arg1])

            if negate_focus:
                focus_msgs.append([rel_lit])
            else:
                focus_msgs.append(rel_lit)

        elif rel["pos"] == "n":
            # Noun predicates with args (referent[, more optional referents])
            rel_lit = (rel["predicate"], "n", [rel_id])

            if negate_focus:
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
                if parse["utt_type"][rel_id]=="ques" and parse["relations"]["by_id"][arg2]["predicate"]=="thing":
                    # "What is X?" type of question; attach a special reserved predicate
                    # to focus_msgs, which signals that which predicate arg2 belongs to is
                    # under question
                    rel_lit = ("?", "*", [arg2, arg1])
                    if negate_focus:
                        focus_msgs.append([rel_lit])
                    else:
                        focus_msgs.append(rel_lit)

                    # Any restrictor of the wh-quantified predicate
                    focus_msgs += daughters[arg2]

                    # Mark arg2 refers to a predicate, not an entity
                    ref_map[arg2]["is_pred"] = True

                else:
                    # Predicational; provide predicates as additional info about subject
                    if negate_focus:
                        focus_msgs.append(daughters[arg2])
                    else:
                        focus_msgs += daughters[arg2]

                    # arg2 is the same entity as arg1
                    ref_map[arg2] = ref_map[arg1]

            else:
                if referential_arg1:
                    # Equative; claim referent identity btw. subject and complement
                    topic_msgs += daughters[arg2]

                    a1a2_vars = [arg1, arg2]
                    rel_lit = ("=", "*", a1a2_vars)

                    if negate_focus:
                        focus_msgs.append([rel_lit])
                    else:
                        focus_msgs.append(rel_lit)

                else:
                    # We are not really interested in 3) (or others?), at least for now
                    raise ValueError("Weird sentence with copula")

        elif arg2 is not None:
            # Two-place predicates have different semantics depending on referentiality of arg2
            # (or whether arg2 is wh-quantified)

            if referential_arg2 or ref_map[arg2]["is_wh_quantified"]:
                # Simpler case; use constant for arg2, and provided predicates concerning arg2
                # are considered topic message
                topic_msgs += daughters[arg2]

                rel_lit = (rel["predicate"], rel["pos"], [arg1, arg2])

                if negate_focus:
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

                lits = [(rel["predicate"], rel["pos"], [arg1, arg2_sk])]
                for a2_lit in daughters[arg2]:
                    # Replace occurrences of arg2 with the skolem term
                    args_sk = [arg2_sk if arg2==a else a for a in a2_lit[2]]
                    a2_lit_sk = a2_lit[:2] + (args_sk,)
                    lits.append(a2_lit_sk)
                
                if negate_focus:
                    focus_msgs.append(lits)
                else:
                    focus_msgs += lits

        else:
            # Other general cases
            continue

    return {rel_id: (topic_msgs, focus_msgs)}
