"""
Language processing module API that exposes only the high-level functionalities
required by the ITL agent: situate the embodied agent in a physical environment,
understand & generate language input in the context of the dialogue
"""
import copy
from collections import defaultdict

from .semantics import SemanticParser
from .dialogue import DialogueManager


class LanguageModule:

    def __init__(self, cfg):
        """
        Args:
            opts: argparse.Namespace, from parse_argument()
        """
        self.semantic = SemanticParser(
            cfg.lang.paths.grammar_image, cfg.lang.paths.ace_binary
        )
        self.dialogue = DialogueManager()

        self.vis_raw = None

        self.unresolved_neologisms = set()

        # New language input buffer
        self.new_input = None

    def situate(self, vis_raw, vis_scene):
        """
        Put entities in the physical environment into domain of discourse
        """
        # No-op if no new visual input
        if vis_scene is None:
            return

        # Start a dialogue information state anew
        self.dialogue.refresh()

        # Store raw visual perception so that it can be used during 'pointing' gesture
        self.vis_raw = vis_raw

        # Incorporate parsed scene graph into dialogue context
        for oi, obj in vis_scene.items():
            bbox = obj["pred_box"]
            self.dialogue.referents["env"][oi] = {
                "bbox": bbox,
                "area": (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
            }
            self.dialogue.referent_names[oi] = oi
        
        # Register these indices as names, for starters
        self.dialogue.referent_names = {i: i for i in self.dialogue.referents["env"]}

    def understand(self, parses, vis_raw, pointing=None):
        """
        Parse language input into MRS, process into ASP-compatible form, and then
        update dialogue state.

        pointing (optional) is a dict summarizing the 'gesture' made along with the
        utterance, indicating the reference (represented as bbox) made by the n'th
        occurrence of linguistic token. Mostly for programmed experiments.
        """
        ti = len(self.dialogue.record)      # New dialogue turn index
        new_record = []                     # New dialogue record for the turn

        # Processing natural language into appropriate logical form
        asp_contents, ref_maps = self.semantic.asp_translate(parses)

        # For indexing clauses in dialogue turn
        se2i = defaultdict(lambda: len(se2i))

        per_sentence = enumerate(zip(parses, asp_contents, ref_maps))
        for si, (parse, asp_content, ref_map) in per_sentence:
            # For indexing referents within individual clauses
            reind_per_src = {
                ei: defaultdict(lambda: len(reind_per_src[ei]))
                for ei in asp_content
            }
            r2i = {
                rf: reind_per_src[v["source_ind"]][v["map_id"]]
                for rf, v in ref_map.items() if v is not None
            }

            # Add to the list of discourse referents
            for rf, v in ref_map.items():
                if v is not None:
                    term_char = "p" if ref_map[rf]["is_pred"] else "x"
                    turn_clause_tag = f"t{ti}c{se2i[(si,v['source_ind'])]}"

                    if type(rf) == tuple:
                        # Function term
                        f_args = tuple(
                            f"{term_char.upper()}{r2i[a]}{turn_clause_tag}"
                                if ref_map[a]["is_univ_quantified"] or ref_map[a]["is_wh_quantified"]
                                else f"{term_char}{r2i[a]}{turn_clause_tag}"
                            for a in rf[1]
                        )
                        rf = (rf[0], f_args)

                        self.dialogue.referents["dis"][rf] = {
                            "provenance": v["provenance"],
                            "is_referential": v["is_referential"],
                            "is_univ_quantified": v["is_univ_quantified"],
                            "is_wh_quantified": v["is_wh_quantified"]
                        }
                    else:
                        assert type(rf) == str
                        if v["is_univ_quantified"] or v["is_wh_quantified"]:
                            rf = f"{term_char.upper()}{r2i[rf]}{turn_clause_tag}"
                        else:
                            rf = f"{term_char}{r2i[rf]}{turn_clause_tag}"

                        self.dialogue.referents["dis"][rf] = {
                            "provenance": v["provenance"],
                            "is_referential": v["is_referential"],
                            "is_univ_quantified": v["is_univ_quantified"],
                            "is_wh_quantified": v["is_wh_quantified"]
                        }

            # Fetch arg1 of index (i.e. sentence 'subject' referent)
            index = parse["relations"]["by_id"][parse["index"]]
            if not len(index["args"]) > 1:
                raise ValueError("Input is not a sentence")

            # Handle certain hard assignments
            for rel in parse["relations"]["by_id"].values():

                if rel["pos"] == "q":
                    # Demonstratives need pointing
                    if "sense" in rel and rel["sense"] == "dem":
                        rels_with_pred = sorted([
                            r for r in parse["relations"]["by_id"].values()
                            if r["predicate"]==rel["predicate"]
                        ], key=lambda r: r["crange"][0])

                        point_target_ind = [
                            r["handle"] for r in rels_with_pred
                        ].index(rel["handle"])

                        pointing_not_provided = pointing is None \
                            or (rel["predicate"] not in pointing) \
                            or (pointing[rel["predicate"]] is None) \
                            or (pointing[rel["predicate"]][point_target_ind] is None)

                        if pointing_not_provided:
                            dem_bbox = None
                        else:
                            dem_bbox = pointing[rel["predicate"]][point_target_ind]

                        pointed = self.dialogue.dem_point(
                            vis_raw, rel["predicate"], dem_bbox=dem_bbox
                        )

                        if pointing is not None:
                            # Update token-to-bbox map of pointing if needed
                            if rel["predicate"] in pointing:
                                bboxes = pointing[rel["predicate"]]
                            else:
                                bboxes = [None] * len(rels_with_pred)

                            bboxes[point_target_ind] = \
                                self.dialogue.referents["env"][pointed]["bbox"]
                            pointing[rel["predicate"]] = bboxes

                        ri = r2i[rel["args"][0]]
                        clause_tag = se2i[(si, ref_map[rel["args"][0]]["source_ind"])]
                        self.dialogue.assignment_hard[f"x{ri}t{ti}c{clause_tag}"] = pointed

                if rel["predicate"] == "named":
                    # Provided symbolic name
                    if rel["carg"] not in self.dialogue.referent_names:
                        # Couldn't resolve the name; explicitly ask again for name resolution
                        rels_with_pred = sorted([
                            r for r in parse["relations"]["by_id"].values()
                            if r["predicate"]=="named" and r["carg"]==rel['carg']
                        ], key=lambda r: r["crange"][0])

                        point_target_ind = [
                            r["handle"] for r in rels_with_pred
                        ].index(rel["handle"])

                        pointing_not_provided = pointing is None \
                            or (rel["predicate"] not in pointing) \
                            or (pointing[rel["predicate"]] is None) \
                            or (pointing[rel["predicate"]][point_target_ind] is None)

                        if pointing_not_provided:
                            dem_bbox = None
                        else:
                            dem_bbox = pointing[rel["carg"]][point_target_ind]

                        pointed = self.dialogue.dem_point(
                            vis_raw, rel["carg"], dem_bbox=dem_bbox
                        )
                        self.dialogue.referent_names[rel["carg"]] = pointed

                        if pointing is not None:
                            # Update token-to-bbox map of pointing if needed
                            if rel["carg"] in pointing:
                                bboxes = pointing[rel["carg"]]
                            else:
                                bboxes = [None] * len(rels_with_pred)

                            bboxes[point_target_ind] = \
                                self.dialogue.referents["env"][pointed]["bbox"]
                            pointing[rel["carg"]] = bboxes

                    ri = r2i[rel["args"][0]]
                    clause_tag = se2i[(si, ref_map[rel["args"][0]]["source_ind"])]
                    self.dialogue.assignment_hard[f"x{ri}t{ti}c{clause_tag}"] = \
                        self.dialogue.referent_names[rel["carg"]]

            # ASP-compatible translation
            for ev_id, (topic_msgs, focus_msgs) in asp_content.items():
                head = []; body = []

                # Process topic messages
                for m in topic_msgs:
                    if type(m) == tuple:
                        # Non-negated message
                        occurring_args = sum([a[1] if type(a)==tuple else (a,) for a in m[2]], ())
                        occurring_args = tuple(set(occurring_args))

                        var_free = not any([ref_map[a]["is_univ_quantified"] for a in occurring_args])

                        if var_free:
                            # Add the grounded literal to head
                            head.append(m[:2]+(tuple(m[2]),False))
                        else:
                            # Add the non-grounded literal to body
                            body.append(m[:2]+(tuple(m[2]),False))
                    else:
                        # Negation of conjunction
                        occurring_args = sum([
                            sum([a[1] if type(a)==tuple else (a,) for a in l[2]], ()) for l in m
                        ], ())
                        occurring_args = tuple(set(occurring_args))

                        var_free = not any([ref_map[a]["is_univ_quantified"] for a in occurring_args])

                        conj = [l[:2]+(tuple(l[2]),False) for l in m]
                        conj = list(set(conj))      # Remove duplicate literals

                        if var_free:
                            # Add the non-grounded conjunction to head
                            head.append(conj)
                        else:
                            # Add the non-grounded conjunction to body
                            body.append(conj)

                # Process focus messages
                for m in focus_msgs:
                    if type(m) == tuple:
                        # Non-negated message
                        head.append(m[:2] + (tuple(m[2]), False))
                    else:
                        # Negation of conjunction
                        conj = [l[:2]+(tuple(l[2]),False) for l in m]
                        conj = list(set(conj))      # Remove duplicate literals
                        head.append(conj)

                if parse["utt_type"][ev_id] == "prop":
                    # Indicatives
                    prop = _map_and_format((head, body), ref_map, ti, si, r2i, se2i)

                    new_record.append(((prop, None), parse["conjunct_raw"][ev_id]))

                elif parse["utt_type"][ev_id] == "ques":
                    # Interrogatives

                    # Determine type of question: Y/N or wh-
                    wh_refs = {
                        rf for rf, v in ref_map.items()
                        if v is not None and v["is_wh_quantified"]
                    }

                    if len(wh_refs) == 0:
                        # Y/N question with no wh-quantified entities
                        q_vars = None
                        raise NotImplemented
                    else:
                        # wh- question with wh-quantified entities
                        
                        # Pull out any literals containing wh-quantified referents
                        # and build into question. Remaining var-free literals are
                        # considered as presupposed statements (all grounded facts;
                        # currently I cannot think of any universally quantified
                        # presuppositions that can be conveyed via questions -- at
                        # least we don't have such cases in our use scenarios).
                        head = [l for l in head if len(wh_refs & set(l[2])) > 0]
                        body = [l for l in body if len(wh_refs & set(l[2])) > 0]
                        presup = [
                            l for l in head if len(wh_refs & set(l[2])) == 0
                        ] + [
                            l for l in body if len(wh_refs & set(l[2])) == 0
                        ]

                        q_vars = wh_refs & set.union(*[set(l[2]) for l in head+body])

                    question = (q_vars, (head, body))

                    if len(presup) > 0:
                        presup = _map_and_format((presup, []), ref_map, ti, si, r2i, se2i)
                    else:
                        presup = None
                    question = _map_and_format(question, ref_map, ti, si, r2i, se2i)

                    new_record.append(((presup, question), parse["conjunct_raw"][ev_id]))
                    self.dialogue.unanswered_Q.add((ti, si))

                elif parse["utt_type"][ev_id] == "comm":
                    # Imperatives
                    raise NotImplementedError
                
                else:
                    # Ambiguous SF
                    raise NotImplementedError
        
        self.dialogue.record.append(("U", new_record))    # Add new record

    def acknowledge(self):
        """ Push an acknowledging utterance to generation buffer """
        self.dialogue.to_generate.append((None, "OK."))

    def generate(self):
        """ Flush the buffer of utterances prepared """
        if len(self.dialogue.to_generate) > 0:
            return_val = []

            new_record = []
            for logical_forms, surface_form in self.dialogue.to_generate:
                if logical_forms is None:
                    logical_forms = (None, None)

                new_record.append((logical_forms, surface_form))

                # NL utterance to log/print
                return_val.append(("generate", surface_form))

            self.dialogue.record.append(("A", new_record))

            self.dialogue.to_generate = []

            return return_val
        else:
            return


def _map_and_format(data, ref_map, ti, si, r2i, se2i):
    # Map MRS referents to ASP terms and format

    def fmt(rf):
        if type(rf) == tuple:
            return (rf[0], tuple(fmt(a) for a in rf[1]))
        else:
            assert type(rf) == str
            is_var = ref_map[rf]['is_univ_quantified'] or ref_map[rf]['is_wh_quantified']

            term_char = "p" if ref_map[rf]["is_pred"] else "x"
            term_char = term_char.upper() if is_var else term_char
            turn_clause_tag = f"t{ti}c{se2i[(si, ref_map[rf]['source_ind'])]}"

            return f"{term_char}{r2i[rf]}{turn_clause_tag}"

    process_conjuncts = lambda conjs: tuple(
        (cnj[0], cnj[1], tuple(fmt(a) for a in cnj[2]), cnj[3])
            if isinstance(cnj, tuple) else list(process_conjuncts(cnj))
        for cnj in conjs
    )

    if isinstance(data[0], list):
        # Proposition
        head, body = data
        return (process_conjuncts(head), process_conjuncts(body))

    elif isinstance(data[0], set):
        # Question
        q_vars, (head, body) = data

        # Annotate whether the variable is zeroth-order (entity) or first-order
        # (predicate)
        new_q_vars = tuple(
            (fmt(e), ref_map[e]["is_pred"]) for e in q_vars
        ) if q_vars is not None else None

        return (new_q_vars, (process_conjuncts(head), process_conjuncts(body)))

    else:
        raise ValueError("Can't _map_and_format this")
