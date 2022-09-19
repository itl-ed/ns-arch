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

        # Each record is a 3-tuple of:
        #   1) speaker: user ("U") or agent ("A")
        #   2) logical form of utterance content
        #   3) original user input string
        self.record = []

        self.unanswered_Q = set()

        # Buffer of utterances to generate
        self.to_generate = []

    def refresh(self):
        """ Clear the current dialogue state to start fresh in a new situation """
        self.__init__()
    
    def export_as_dict(self):
        """ Export the current dialogue information state as a dict """
        return vars(self)

    def dem_point(self, vis_raw, label_target, dem_bbox=None):
        """
        Simple pointing interface for entities quantified by demonstratives.

        dem_bbox (optional), if provided, replaces the click/drag UI and can be
        directly used as desired bbox specification. Mostly used for programmed
        experiments.
        """
        if dem_bbox is not None:
            drawn_bbox = dem_bbox
        else:
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            ax.set_title(
                f"'{label_target}' needs pointing\n"
                "(Press 't' to toggle bounding box selector)"
            )
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

            is_drawn = type(ui_status["choice"]) is tuple
            if is_drawn:
                drawn_bbox = np.array(ui_status["choice"])
            else:
                drawn_bbox = None

        if drawn_bbox is not None:
            # If the choice is a newly drawn bounding box and doesn't overlap with 
            # any other box with high IoU, register this as new entity and return

            # First check if there's any existing high-IoU bounding box; by 'high'
            # we refer to some arbitrary threshold -- let's use 0.8 here
            env_ref_bboxes = torch.stack(
                [torch.tensor(e["bbox"]) for e in self.referents["env"].values()]
            )

            iou_thresh = 0.7
            ious = torchvision.ops.box_iou(
                torch.tensor(drawn_bbox)[None,:], env_ref_bboxes
            )
            best_match = ious.max(dim=-1)

            if best_match.values.item() > iou_thresh:
                # Assume the 'pointed' entity is actually this one
                max_ind = best_match.indices.item()
                pointed = list(self.referents["env"].keys())[max_ind]
            else:
                # Register the entity as a novel environment referent
                pointed = f"o{len(env_ref_bboxes)}"

                self.referents["env"][pointed] = {
                    "bbox": drawn_bbox,
                    "area": (drawn_bbox[2]-drawn_bbox[0]) * (drawn_bbox[3]-drawn_bbox[1])
                }
                self.referent_names[pointed] = pointed
        else:
            # Clicked on one of the options
            pointed = ui_status["choice"]

        assert pointed is not None

        return pointed
