import copy

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector


class DialogueManager:
    """Maintain dialogue state and handle NLU, NLG in context"""

    def __init__(self):

        self.referents = {
            "env": {},  # Sensed via physical perception
            "dis": {}   # Introduced by dialogue
        }

        self.assignment_hard = {}  # Store fixed assignment by demonstrative+pointing, names, etc.
        self.referent_names = {}   # Store mapping from symbolic name to entity

        # Each record is a 4-tuple of:
        #   1) speaker: user ("U") or agent ("A"),
        #   2) utterance type: proposition ("|"), question ("?") or command ("!")
        #   3) logical form of utterance content
        #   4) original user input string
        self.record = []

        # Buffer of utterances to generate
        self.to_generate = []

    def refresh(self):
        """Clear the current dialogue state to start fresh in a new situation"""
        self.__init__()
    
    def export_as_dict(self):
        """ Export the current dialogue information state as a dict """
        return vars(self)

    def _dem_point(self, vis_raw):
        """
        Simple pointing interface for entities quantified by demonstratives
        """
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_title("Press 't' to toggle bounding box selector")
        ax.imshow(vis_raw)

        # Bounding boxes for recognized entities
        rects = {}
        for name, ent in self.referents["env"].items():
            x1, y1, x2, y2 = ent["bbox"]
            r = Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=1, edgecolor=(0.5, 0.8, 1, 0.6),
                facecolor=(0.5, 0.8, 1, 0.2)
            )

            ax.add_patch(r)
            r.set_visible(True)

            rects[name] = r
        
        # Tracking UI states
        ui_status = {
            "hovered": {e: False for e in self.referents["env"]},
            "focus": None,
            "clicked": None,
            "choice": None
        }

        # Rectangle selector for drawing new bounding box
        def bbox_draw_callback(ev_click, ev_release):
            x1, y1 = ev_click.xdata, ev_click.ydata
            x2, y2 = ev_release.xdata, ev_release.ydata

            ui_status["choice"] = (x1, y1, x2, y2)
            plt.close()
        selector = RectangleSelector(
            ax, bbox_draw_callback,
            minspanx=1, minspany=1,
            spancoords="pixels",
            useblit=True,
            interactive=True
        )
        selector.set_active(False)

        fig = plt.gcf()

        # Event handlers
        def hover(ev):
            if ev.inaxes != ax: return

            update = {}
            for name, rect in rects.items():

                if rect.get_visible():
                    # Toggle active
                    cont, _ = rect.contains(ev)

                    # Mouseenter
                    if cont and not ui_status["hovered"][name]:
                        update[name] = True

                    # Mouseleave
                    if not cont and ui_status["hovered"][name]:
                        update[name] = False

                else:
                    # Toggle inactive
                    if ui_status["hovered"][name]:
                        update[name] = False
            
            if len(update) == 0: return  # Early exit

            ui_status["hovered"].update(update)

            hovered = [name for name, over in ui_status["hovered"].items() if over]
            hovered.sort(key=lambda n: self.referents["env"][n]["area"])

            new_focus = hovered[0] if len(hovered) > 0 else None

            if new_focus != ui_status["focus"]:
                ui_status["focus"] = new_focus

                for name, rect in rects.items():
                    c = rect.get_facecolor()
                    alpha = 0.6 if name == ui_status["focus"] else 0.2
                    rect.set_facecolor((c[0], c[1], c[2], alpha))
            
            fig.canvas.draw_idle()
        
        def mouse_press(ev):
            # Ignore when selecting by drawing a new bbox
            if selector.get_active():
                return
            
            # Ignore when no focus
            if ui_status["focus"] is None:
                return

            ui_status["clicked"] = ui_status["focus"]

            rect = rects[ui_status["clicked"]]
            rect.set_facecolor((0.8, 0.5, 1, 0.6))

            fig.canvas.draw_idle()

        def mouse_release(ev):
            # Ignore when selecting by drawing a new bbox
            if selector.get_active():
                return

            # Ignore when not clicked on any bbox
            if ui_status["clicked"] is None:
                return

            rect = rects[ui_status["clicked"]]

            cont, _ = rect.contains(ev)

            if cont:
                ui_status["choice"] = ui_status["clicked"]
                plt.close()

            else:
                alpha = 0.6 if ui_status["clicked"] == ui_status["focus"] else 0.2
                rect.set_facecolor((0.5, 0.8, 1, alpha))

                ui_status["clicked"] = None

                fig.canvas.draw_idle()

        def key_press(ev):
            if ev.key == "t":
                is_active = selector.get_active()
                selector.set_active(not is_active)

                for r in rects.values():
                    r.set_visible(is_active)

            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)
        fig.canvas.mpl_connect("button_press_event", mouse_press)
        fig.canvas.mpl_connect("button_release_event", mouse_release)
        fig.canvas.mpl_connect("key_press_event", key_press)
        plt.show(block=True)

        # If the choice is a newly drawn bounding box and doesn't overlap with 
        # any other box with high IoU, register this as new entity and return
        is_drawn = type(ui_status["choice"]) is tuple
        if is_drawn:
            # First check if there's any existing high-IoU bounding box; by 'high'
            # we refer to some arbitrary threshold -- let's use 0.8 here
            drawn_bbox = np.array(ui_status["choice"])
            env_ref_bboxes = torch.stack(
                [torch.tensor(e["bbox"]) for e in self.referents["env"].values()]
            )

            iou_thresh = 0.8
            ious = torchvision.ops.box_iou(
                torch.tensor(drawn_bbox)[None,:], env_ref_bboxes
            )
            best_match = ious.max(dim=-1)

            if best_match.values.item() > iou_thresh:
                # Assume the 'pointed' entity is actually this one
                matched_ent = list(self.referents["env"].keys())[2]
                print(f"A> I'm going to assume you pointed at '{matched_ent}'...")

                ui_status["choice"] = matched_ent
            else:
                # Register the entity as a novel environment referent
                new_ent = f"o{len(env_ref_bboxes)}"
                
                print(f"A> I wasn't aware of this as an entity.")
                print(f"A> Registering as new object '{new_ent}'...")
                self.referents["env"][new_ent] = {
                    "bbox": drawn_bbox,
                    "area": (drawn_bbox[2]-drawn_bbox[0]) * (drawn_bbox[3]-drawn_bbox[1])
                }
                self.referent_names[new_ent] = new_ent

                ui_status["choice"] = new_ent

        return ui_status["choice"]

    def understand(self, usr_in, parser, vis_raw):
        """
        Parse language input into MRS, process into ASP-compatible form, and then
        update dialogue state. Also return any new agenda items.
        """
        ui = len(self.record)  # Utterance index
        agenda = []

        # Processing natural language into appropriate logical form
        parse = parser.nl_parse(usr_in)
        asp_content, ref_map = parser.asp_translate(parse)

        # Add to the list of discourse referents
        for rf, v in ref_map.items():
            if type(rf) == tuple:
                # Function term
                if v is not None:
                    f_args = tuple(
                        f"X{ref_map[a]['map_id']}u{ui}"
                            if ref_map[a]["is_univ_quantified"] or ref_map[a]["is_wh_quantified"]
                            else f"x{ref_map[a]['map_id']}u{ui}"
                        for a in rf[1]
                    )
                    rf = (rf[0], f_args)

                    self.referents["dis"][rf] = {
                        "is_referential": v["is_referential"],
                        "is_univ_quantified": v["is_univ_quantified"],
                        "is_wh_quantified": v["is_wh_quantified"]
                    }
            else:
                assert type(rf) == str
                if v is not None:
                    if v["is_univ_quantified"] or v["is_wh_quantified"]:
                        rf = f"X{v['map_id']}u{ui}"
                    else:
                        rf = f"x{v['map_id']}u{ui}"

                    self.referents["dis"][rf] = {
                        "is_referential": v["is_referential"],
                        "is_univ_quantified": v["is_univ_quantified"],
                        "is_wh_quantified": v["is_wh_quantified"]
                    }

        # Fetch arg1 of index (i.e. sentence 'subject' referent)
        index = parse["relations"]["by_id"][parse["index"]]
        if not len(index["args"]) > 1:
            raise ValueError("Input is not a sentence")

        # ASP-compatible translation
        topic_msgs, focus_msgs = asp_content

        # Handle certain hard assignments
        for rel in parse["relations"]["by_id"].values():

            if rel["pos"] == "q":
                # Demonstratives need pointing
                if "sense" in rel and rel["sense"] == "dem":
                    print(f"Sys> '{rel['predicate']}' needs pointing")

                    referent = ref_map[rel["args"][0]]["map_id"]
                    self.assignment_hard[f"x{referent}u{ui}"] = self._dem_point(vis_raw)
            
            if rel["predicate"] == "named":
                # Provided symbolic name
                if rel["carg"] not in self.referent_names:
                    # Couldn't resolve the name; explicitly ask again for name resolution
                    print(f"A> What were you referring to by '{rel['carg']}'?")

                    self.referent_names[rel["carg"]] = self._dem_point(vis_raw)

                referent = ref_map[rel["args"][0]]["map_id"]
                self.assignment_hard[f"x{referent}u{ui}"] = self.referent_names[rel["carg"]]

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

        if parse["utt_type"] == "prop":
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

            self.record.append(("U", "|", (info+info_aux, None), usr_in))

        elif parse["utt_type"] == "ques":
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

            self.record.append(("U", "?", (info, query), usr_in))

            agenda.append(("unanswered_Q", ui))

        elif parse["utt_type"] == "comm":
            # Imperatives
            raise NotImplementedError
        
        else:
            # Ambiguous SF
            raise NotImplementedError

        return agenda


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
