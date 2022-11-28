"""
For visualizing predictions from scene graph generation models, using detectron2
visualization toolkits.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider


def visualize_sg_predictions(img, scene, lexicon=None):
    """
    Args:
        img: numpy.array; RGB image data
        scene: dict; a parsed scene graph, obtained from visual module output
    """
    if lexicon is None: lexicon = { "cls": {}, "att": {} }

    # Prepare labels for class/attribute/relation predictions; show only the categories
    # with the highest scores
    cls_argmaxs = [obj["pred_classes"].argmax(axis=-1) for obj in scene.values()]
    cls_maxs = [obj["pred_classes"].max(axis=-1) for obj in scene.values()]
    cls_labels = {
        oi: (lexicon["cls"].get(i, f"cls_{i}"), v)
        for oi, i, v in zip(scene, cls_argmaxs, cls_maxs)
    }

    att_argmaxs = [obj["pred_attributes"].argmax(axis=-1) for obj in scene.values()]
    att_maxs = [obj["pred_attributes"].max(axis=-1) for obj in scene.values()]
    att_labels = {
        oi: (lexicon["att"].get(i, f"att_{i}"), v)
        for oi, i, v in zip(scene, att_argmaxs, att_maxs)
    }

    # rel_argmaxs = [
    #     { obj2: rels.argmax(axis=-1) for obj2, rels in obj["pred_relations"].items() }
    #     for obj in scene.values()
    # ]
    # rel_maxs = [
    #     { obj2: rels.max(axis=-1) for obj2, rels in obj["pred_relations"].items() }
    #     for obj in scene.values()
    # ]
    # rel_labels = {
    #     oi: { obj: (indices[obj], values[obj]) for obj in indices }
    #     for oi, indices, values in zip(scene, rel_argmaxs, rel_maxs)
    # }

    # rel_colors = defaultdict(lambda: random_color(rgb=True, maximum=1))

    # Show in pop-up window
    fig = plt.gcf()

    # thresholds = { "obj": 0, "rel": 0 }
    thresholds = { "obj": 0.1 }

    # def render(obj_thresh, rel_thresh):
    def render(obj_thresh):
        ax = fig.add_subplot(1,1,1)
        ax.imshow(img)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        # Filter predictions to visualize by objectness threshold
        # objs_filtered = sorted(
        #     {
        #         oi for oi, obj in scene.items()
        #         if obj["pred_objectness"] > obj_thresh
        #     },
        #     key=lambda t: int(t.strip("o"))
        # )
        objs_filtered = scene

        # Boxes and classes
        for oi in objs_filtered:
            x1, y1, x2, y2 = scene[oi]["pred_box"]
            r = Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor="r", facecolor="none"
            )
            ax.add_patch(r)

            # text_label = f"{oi}: {float(scene[oi]['pred_objectness']):.2f}:"
            text_label = f"{oi}:"
            text_label += f" {cls_labels[oi][0]} ({float(cls_labels[oi][1]):.2f})"
            text_label += f" / {att_labels[oi][0]} ({float(att_labels[oi][1]):.2f})"
            text_label = ax.text(x1, y1, text_label, color="w")
            text_label.set_bbox({ "facecolor": "r", "alpha": 0.5, "edgecolor": "r" })

        # # Relations; show only those between objects with high objectness scores
        # occurred_rels = []
        # for oi in objs_filtered:
        #     for oj in objs_filtered:
        #         if oi==oj: continue

        #         pred, score = rel_labels[oi][oj]

        #         if score > rel_thresh:
        #             occurred_rels.append(pred)

        #             box1 = scene[oi]["pred_box"]
        #             box2 = scene[oj]["pred_box"]

        #             v_pred.draw_line(
        #                 [float(box1[0]+10), float(box2[0]+10)],
        #                 [float(box1[1]+10), float(box2[1]+10)],
        #                 color=rel_colors[pred],
        #                 linewidth=((score*1.5)**3)
        #             )

        # pred_img = v_pred.output

        # # Relation legend
        # pred_img.ax.legend(
        #     handles=[
        #         Patch(color=rel_colors[r], label=r) for r in set(occurred_rels)
        #     ]
        # )

        fig.canvas.draw_idle()
    
    def objs_render(val):
        thresholds["obj"] = val
        # render(thresholds["obj"], thresholds["rel"])
        render(thresholds["obj"])
    # def rels_render(val):
    #     thresholds["rel"] = val
    #     render(thresholds["obj"], thresholds["rel"])
    
    # render(thresholds["obj"], thresholds["rel"])
    render(thresholds["obj"])

    obj_slider = Slider(
        plt.axes([0.25, 0.05, 0.6, 0.03]),
        "Obj. score", 0.0, 1.0, valinit=thresholds["obj"]
    )
    obj_slider.on_changed(objs_render)
    # rel_slider = Slider(
    #     plt.axes([0.25, 0.02, 0.6, 0.03]),
    #     "Rel. score", 0.0, 1.0, valinit=thresholds["rel"]
    # )
    # rel_slider.on_changed(rels_render)

    plt.show()

    return fig
