import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


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
        ax.imshow(vis_raw)

        rects = {}
        for name, ent in self.referents["env"].items():
            x1, y1, x2, y2 = ent["bbox"]
            r = Rectangle((x1, y1), x2-x1, y2-y1, facecolor=(0.5, 0.8, 1, 0.2))

            ax.add_patch(r)
            r.set_visible(name.startswith("o"))  # Toggle: object visible, part not

            rects[name] = r
        
        ui_status = {
            "hovered": {e: False for e in self.referents["env"]},
            "focus": None,
            "clicked": None,
            "choice": None
        }

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
            ui_status["clicked"] = ui_status["focus"]

            rect = rects[ui_status["clicked"]]
            rect.set_facecolor((0.8, 0.5, 1, 0.6))

            fig.canvas.draw_idle()

        def mouse_release(ev):
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
            if ev.key == "shift":
                for r in rects.values():
                    visible = r.get_visible()
                    r.set_visible(not visible)
                
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)
        fig.canvas.mpl_connect("button_press_event", mouse_press)
        fig.canvas.mpl_connect("button_release_event", mouse_release)
        fig.canvas.mpl_connect("key_press_event", key_press)
        plt.show()

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
                    entity = self._dem_point(vis_raw)

                    self.referent_names[rel["carg"]] = entity

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
                    rule_body = _filter_implied(rule_body)

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
                        constr_body = _filter_implied(constr_body)

                        constr = (None, constr_body, False)
                        constraints.append(constr)
                
                facts = [(l[0], l[1], tuple(l[2])) for l in facts]
                facts = list(set(facts))
                facts = _filter_implied(facts)

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


def _filter_implied(literals):
    """ Remove literals if already implied by others; for now only consider '_part_of' """

    filtered = []

    for l1 in literals:

        implied = False
        for l2 in literals:
            if l1[0] == "part_of" and l1[0] != l2[0] and \
                    l2[0].endswith("_of") and l1[2] == l2[2]:
                implied = True; break

        if not implied: filtered.append(l1)

    return filtered
