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
            "dis": set()   # Introduced by dialogue
        }
        self.assignment_hard = {}  # Store fixed assignment by demonstrative+pointing, names, etc.
        self.referent_names = {}   # Store mapping from symbolic name to entity

        self.record = []
        self.unanswered_Q = []
    
    def refresh(self):
        """Clear the current dialogue state to start fresh in a new situation"""
        self.__init__()
    
    def export_state(self):
        """Export the current dialogue information state as a dict"""
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

            ui_status["clicked"] = ui_status["focus"]

            rect = rects[ui_status["clicked"]]
            rect.set_facecolor((0.8, 0.5, 1, 0.6))

            fig.canvas.draw_idle()

        def mouse_release(ev):
            # Ignore when selecting by drawing a new bbox
            if selector.get_active():
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
        plt.show()

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

    def understand(self, usr_in, parser, lex, vis_raw, agenda):
        """
        Parse language input into MRS, process into ASP-compatible form, and then
        update dialogue state. Also add new agenda items to the provided list.
        """

        ui = len(self.record)  # Utterance index

        # Processing natural language into appropriate logical form
        parse = parser.nl_parse(usr_in)
        asp_content, var_map = parser.asp_translate(parse)

        # Fetch arg1 of index (i.e. sentence 'subject' referent)
        index = parse["relations"]["by_id"][parse["index"]]
        if not len(index["args"]) > 1:
            raise ValueError("Input is not a sentence")
        index_arg1 = index["args"][1]

        # Whether arg1 is bare NP (signalling generic statement)        
        bare_arg1 = False
        for rel in parse["relations"]["by_args"][(index_arg1,)]:
            if rel["pos"] == "q" and rel["predicate"] == "udef":
                bare_arg1 = True

        # ASP-compatible translation
        topic_lits, focus_lits = asp_content

        # Referents ever mentioned in grounded facts are (ASP) constants, others are
        # (ASP) variables
        const_ents = set()

        # Handle certain hard assignments
        for rel in parse["relations"]["by_id"].values():

            if rel["pos"] == "q":
                # Demonstratives need pointing
                if "sense" in rel and rel["sense"] == "dem":
                    print(f"Sys> '{rel['predicate']}' needs pointing")

                    referent = var_map[rel["args"][0]]
                    const_ents.add(referent)

                    self.assignment_hard[f"x{referent}u{ui}"] = self._dem_point(vis_raw)
            
            if rel["predicate"] == "named":
                # Provided symbolic name
                if rel["carg"] not in self.referent_names:
                    # Couldn't resolve the name; explicitly ask again for name resolution
                    print(f"A> What were you referring to by '{rel['carg']}'?")

                    self.referent_names[rel["carg"]] = self._dem_point(vis_raw)

                referent = var_map[rel["args"][0]]
                const_ents.add(referent)

                self.assignment_hard[f"x{referent}u{ui}"] = self.referent_names[rel["carg"]]

        if parse["utt_type"] == "prop":
            # Indicatives

            if bare_arg1:
                # Consider the intensional meaning of the generic statement as message
                rules = []

                # Collapse conditions in topic_lits
                topic_lits_coll = []
                for l in topic_lits:
                    topic_lits_coll.append(l[0])
                    topic_lits_coll += l[1]

                # Compose rules, appropriately processing consequent literals
                for l in focus_lits:

                    if len(l) == 3:
                        # Rule with non-negated head
                        rule_head = l[0]

                        additional_conds = l[1]

                        rule_body = topic_lits_coll + additional_conds

                        choice = l[2]

                    else:
                        # len(l) == 2; rule with negated head
                        neg_lits = l[0]

                        rule_head = None  # No rule head ~ Integretiy constraint

                        additional_conds = sum([nl[1] for nl in neg_lits], [])
                        additional_mains = [nl[0] for nl in neg_lits]

                        rule_body = topic_lits_coll + additional_conds + additional_mains

                        choice = False
                    
                    rule_body = [(bl[0], bl[1], tuple(bl[2])) for bl in rule_body]
                    rule_body = list(set(rule_body))  # Remove duplicate literals

                    rule = (rule_head, rule_body, choice)
                    rules.append(rule)
                
                rules = _varnames_format(rules, set(), f"u{ui}")

                self.record.append(("U", "|", rules))

                agenda.append(("kb_add", ui))  # Add new knowledge to KB

            else:
                # Otherwise, regard propositions as grounded facts about individuals
                constraints = []; facts = []

                # Collapse all entries in topic_lits into a bag of facts (body-less rules)
                for l in topic_lits:
                    facts.append(l[0])
                    facts += l[1]
                    
                    for t in l[0][2]:
                        const_ents.add(t)
                    for cl in l[1]:
                        for t in cl[2]:
                            const_ents.add(t)

                # Non-negated entries in focus_lits also become facts, while negated entries
                # become (partially) grounded integrity constraints
                for l in focus_lits:

                    if len(l) == 3:
                        # Non-negated facts
                        facts.append(l[0])
                        facts += l[1]

                        for t in l[0][2]:
                            const_ents.add(t)
                        for cl in l[1]:
                            for t in cl[2]:
                                const_ents.add(t)

                    else:
                        # len(l) == 2; negated facts
                        neg_lits = l[0]

                        additional_conds = sum([nl[1] for nl in neg_lits], [])
                        additional_mains = [nl[0] for nl in neg_lits]

                        constr_body = additional_conds + additional_mains
                        constr_body = [(bl[0], bl[1], tuple(bl[2])) for bl in constr_body]
                        constr_body = list(set(constr_body))  # Remove duplicate literals

                        constr = (None, constr_body, False)
                        constraints.append(constr)
                
                facts = [(l[0], l[1], tuple(l[2])) for l in facts]
                facts = list(set(facts))

                # Facts with no body literals (keep formats consistent for rules and facts)
                facts = [(l, None, False) for l in facts]

                rules = facts + constraints

                rules = _varnames_format(rules, const_ents, f"u{ui}")

                self.record.append(("U", "|", rules))

        elif parse["utt_type"] == "ques":
            # Interrogatives

            ## TODO: Move these to 'generate (answer)' part? ##
            # Determine type of question
            q_type = None

            for rel in parse["relations"]["by_id"].values():
                if rel["predicate"] == "which":
                    bound = parse["relations"]["by_id"][rel["args"][0]]
                    
                    if bound["predicate"] == "thing":
                        q_type = "What"
                    elif bound["predicate"] == "reason":
                        q_type = "Why"
                    else:
                        q_type = "Which"
            
            if q_type is None: q_type = "Y/N"

            self.record.append(("U", "?", None))
            self.unanswered_Q.append(ui)

            agenda.append(("answer", ui))

        elif parse["utt_type"] == "comm":
            # Imperatives
            raise NotImplementedError
        
        else:
            # Ambiguous SF
            raise NotImplementedError
        
        # Add to the list of discourse referents
        for ref in const_ents:
            self.referents["dis"].add(f"x{ref}u{ui}")
        
        # Handle neologisms
        # for rel in parse["relations"]["by_id"].values():
        #     if not rel["lexical"]: continue

        #     term = (rel["pos"], rel["predicate"])

        #     # Some reserved terms
        #     if term[0] == "q" or \
        #         term == ("v", "be") or \
        #         term == ("v", "have"): continue

        #     if term not in lex.s2d:
        #         agenda.append(("resolve_neologism", term))


def _varnames_format(rules, const_ents, tail):

    formatted = []
    fmt = lambda ri: f"{'x' if ri in const_ents else 'X'}{ri}{tail}"

    for r in rules:
        h, b, c = r

        if h is None:
            new_h = None
        else:
            new_h = (h[0], h[1], tuple(fmt(ri) for ri in h[2]))

        if b is None:
            new_b = None
        else:
            new_b = [
                (l[0], l[1], tuple(fmt(ri) for ri in l[2]))
            for l in b]

        formatted.append((new_h, new_b, c))

    return formatted
