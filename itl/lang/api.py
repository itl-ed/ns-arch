"""
Language processing module API that exposes only the high-level functionalities
requtt_idred by the ITL agent: situate the embodied agent in a physical environment,
understand & generate language input in the context of the dialogue
"""
import copy

from .semantics import SemanticParser
from .dialogue import DialogueManager


class LanguageModule:

    def __init__(self, opts):
        """
        Args:
            opts: argparse.Namespace, from parse_argument()
        """
        self.opts = opts
        self.semantic = SemanticParser(opts.grammar_image_path, opts.ace_binary_path)
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
        if vis_raw is None and vis_scene is None:
            return

        # Start a dialogue information state anew
        self.dialogue.refresh()

        # Store raw visual perception so that it can be used during 'pointing' gesture
        self.vis_raw = vis_raw

        # Incorporate parsed scene graph into dialogue context
        for oi, obj in vis_scene.items():
            bbox = obj["pred_boxes"]
            self.dialogue.referents["env"][oi] = {
                "bbox": bbox,
                "area": (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
            }
        
        # Register these indices as names, for starters
        self.dialogue.referent_names = {i: i for i in self.dialogue.referents["env"]}

    def understand(self, parse, vis_raw, pointing=None):
        """
        Parse language input into MRS, process into ASP-compatible form, and then
        update dialogue state. Also return any new agenda items.

        pointing (optional) is a dict summarizing the 'gesture' made along with the
        utterance, indicating the reference (represented as bbox) made by the n'th
        occurrence of linguistic token. Mostly for programmed experiments.
        """
        ui = len(self.dialogue.record)  # Utterance index
        agenda = []

        # Processing natural language into appropriate logical form
        asp_content, ref_map = self.semantic.asp_translate(parse)

        # Add to the list of discourse referents
        for rf, v in ref_map.items():
            if v is not None:
                term_char = "p" if ref_map[rf]["is_pred"] else "x"

                if type(rf) == tuple:
                    # Function term
                    f_args = tuple(
                        f"{term_char.upper()}{ref_map[a]['map_id']}u{ui}"
                            if ref_map[a]["is_univ_quantified"] or ref_map[a]["is_wh_quantified"]
                            else f"{term_char}{ref_map[a]['map_id']}u{ui}"
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
                        rf = f"{term_char.upper()}{v['map_id']}u{ui}"
                    else:
                        rf = f"{term_char}{v['map_id']}u{ui}"

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

                    referent = ref_map[rel["args"][0]]["map_id"]
                    self.dialogue.assignment_hard[f"x{referent}u{ui}"] = pointed

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

                referent = ref_map[rel["args"][0]]["map_id"]
                self.dialogue.assignment_hard[f"x{referent}u{ui}"] = \
                    self.dialogue.referent_names[rel["carg"]]

        # ASP-compatible translation
        for ev_id, (topic_msgs, focus_msgs) in asp_content.items():

            info = []; info_aux = []

            # Process topic messages
            body_lits = []
            for m in topic_msgs:
                if type(m) == tuple:
                    # Non-negated message
                    occurring_args = sum([a[1] if type(a)==tuple else (a,) for a in m[2]], ())
                    occurring_args = tuple(set(occurring_args))

                    var_free = not any([ref_map[a]["is_univ_quantified"] for a in occurring_args])

                    if var_free:
                        # Add the grounded literal as body-less fact
                        info.append(([m[:2]+(tuple(m[2]),False)], None, None))
                    else:
                        # Add the non-grounded literal as rule body literal
                        body_lits.append(m[:2]+(tuple(m[2]),False))
                else:
                    # Negation of conjunction
                    occurring_args = sum([
                        sum([a[1] if type(a)==tuple else (a,) for a in l[2]], ()) for l in m
                    ], ())
                    occurring_args = tuple(set(occurring_args))

                    var_free = not any([ref_map[a]["is_univ_quantified"] for a in occurring_args])

                    if var_free:
                        # Add a constraint (head-less rule) having the conjunction as body
                        constr_body = [l[:2]+(tuple(l[2]),False) for l in m]
                        constr_body = list(set(constr_body))  # Remove duplicate literals

                        info.append((None, constr_body, None))
                    else:
                        if len(m) > 1:
                            # Introduce new aux literal which is satisfied when the conjunction
                            # is True, then add negated aux literal to rule body
                            aux_pred_name = f"aux{len(info_aux)}"

                            aux_head = (aux_pred_name, "*", occurring_args, False)
                            aux_head_neg = (aux_pred_name, "*", occurring_args, True)
                            aux_body = [l[:2]+(tuple(l[2]),False) for l in m]
                            aux_body = list(set(aux_body))  # Remove duplicate literals

                            info_aux.append(([aux_head], aux_body, None))
                            body_lits.append(aux_head_neg)
                        else:
                            # Simpler case; add negated literal straight to rule body
                            assert len(m) == 1
                            body_lits.append(m[0][:2]+(tuple(m[0][2]),True))
            
            # Remove duplicates in body literals
            body_lits = list(set(body_lits))

            if parse["utt_type"][ev_id] == "prop":
                # Indicatives

                # Process focus messages
                for m in focus_msgs:
                    rule_body = copy.deepcopy(body_lits)

                    if type(m) == tuple:
                        # Non-negated message
                        rule_head = [m[:2] + (tuple(m[2]), False)]
                    else:
                        # Negation of conjunction
                        rule_head = None
                        rule_body += [l[:2]+(tuple(l[2]),False) for l in m]

                    if len(rule_body) == 0:
                        rule_body = None

                    info.append((rule_head, rule_body, None))
                
                info = _map_and_format(info, ref_map, f"u{ui}")
                info_aux = _map_and_format(info_aux, ref_map, f"u{ui}")

                new_record = ("U", (info+info_aux, None), parse["raw"])
                self.dialogue.record.append(new_record)    # Add new record

            elif parse["utt_type"][ev_id] == "ques":
                # Interrogatives

                # Determine type of question: Y/N or wh-
                wh_refs = {rf for rf, v in ref_map.items() if v is not None and v["is_wh_quantified"]}

                if len(wh_refs) == 0:
                    # Y/N question with no wh-quantified entities; add built rules as
                    # queried formulas

                    # Process focus messages
                    q_rules = []

                    for m in focus_msgs:
                        rule_body = copy.deepcopy(body_lits)

                        if type(m) == tuple:
                            # Non-negated message
                            rule_head = [m[:2] + (tuple(m[2]), False)]
                        else:
                            # Negation of conjunction
                            rule_head = None
                            rule_body += [l[:2]+(tuple(l[2]),False) for l in m]
                        
                        if len(rule_body) == 0:
                            rule_body = None
                        
                        q_rules.append((rule_head, rule_body, None))
                    
                    query = (None, q_rules)

                else:
                    # wh- question with wh-quantified entities; first add built rules to
                    # info, then extract any rules containing wh-entities as queried rules

                    # Process focus messages
                    for m in focus_msgs:
                        rule_body = copy.deepcopy(body_lits)

                        if type(m) == tuple:
                            # Non-negated message
                            rule_head = [m[:2] + (tuple(m[2]), False)]
                        else:
                            # Negation of conjunction
                            rule_head = None
                            rule_body += [l[:2]+(tuple(l[2]),False) for l in m]

                        if len(rule_body) == 0:
                            rule_body = None
                        
                        info.append((rule_head, rule_body, None))
                    
                    # Then pull out any rules containing wh-quantified referents
                    rule_lits = [
                        (r[0] if r[0] is not None else []) + (r[1] if r[1] is not None else [])
                        for r in info
                    ]
                    rule_args = [
                        set(sum([l[2] for l in rls], ())) for rls in rule_lits
                    ]
                    q_rules = [info[i] for i, args in enumerate(rule_args) if len(wh_refs & args) > 0]
                    info = [info[i] for i, args in enumerate(rule_args) if len(wh_refs & args) == 0]
                    
                    q_vars = set.union(*rule_args) & wh_refs
                    query = (q_vars, q_rules)

                if len(info) > 0:
                    info = _map_and_format(info, ref_map, f"u{ui}")
                else:
                    info = None
                query = _map_and_format(query, ref_map, f"u{ui}")

                new_record = ("U", (info, query), parse["raw"])
                self.dialogue.record.append(new_record)    # Add new record
                self.dialogue.unanswered_Q.add(ui)

            elif parse["utt_type"][ev_id] == "comm":
                # Imperatives
                raise NotImplementedError
            
            else:
                # Ambiguous SF
                raise NotImplementedError

        return agenda

    def acknowledge(self):
        """ Push an acknowledging utterance to generation buffer """
        self.dialogue.to_generate.append("OK.")

    def generate(self):
        """ Flush the buffer of utterances prepared """
        if len(self.dialogue.to_generate) > 0:
            utt = " ".join(self.dialogue.to_generate)
            print(f"A> {utt}")

            self.dialogue.to_generate = []
            
            return ("generate", utt)
        else:
            return


def _map_and_format(data, ref_map, tail):
    # Map MRS referents to ASP terms and format

    def _fmt(rf):
        if type(rf) == tuple:
            return (rf[0], tuple(_fmt(a) for a in rf[1]))
        else:
            assert type(rf) == str
            is_var = ref_map[rf]['is_univ_quantified'] or ref_map[rf]['is_wh_quantified']

            term_char = "p" if ref_map[rf]["is_pred"] else "x"
            term_char = term_char.upper() if is_var else term_char

            return f"{term_char}{ref_map[rf]['map_id']}{tail}"

    def _process_rules(entries):
        formatted = []
        for entry in entries:
            h, b, c = entry

            if h is None:
                new_h = None
            else:
                new_h = [
                    (l[0], l[1], tuple(_fmt(a) for a in l[2]), l[3]) for l in h
                ]

            if b is None:
                new_b = None
            else:
                new_b = [
                    (l[0], l[1], tuple(_fmt(a) for a in l[2]), l[3]) for l in b
                ]
            
            formatted.append((new_h, new_b, c))
        
        return formatted

    if type(data) == list:
        # ASP-compatible rules
        return _process_rules(data)

    elif type(data) == tuple:
        # Queries to make on computed ASP models
        q_vars, q_rules = data

        # Annotate whether the variable is zeroth-order (entity) or first-order (predicate)
        # (May generalize this to handle second-order (gen. quantifiers) and beyond? Though
        # I don't think that would happen in near future...)
        new_q_vars = tuple(
            (_fmt(e), ref_map[e]["is_pred"]) for e in q_vars
        ) if q_vars is not None else None

        return (new_q_vars, _process_rules(q_rules))

    else:
        raise ValueError("Can't _map_and_format this")
