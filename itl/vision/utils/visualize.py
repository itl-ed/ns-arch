"""
For visualizing predictions from scene graph generation models, using detectron2
visualization toolkits.
"""
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.widgets import Slider
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.colormap import random_color


def visualize_sg_predictions(img, scene, predicates):
    """
    Args:
        img: numpy.array; RGB image data
        scene: dict; a parsed scene graph, obtained from visual module output
    """
    # Prepare labels for class/attribute/relation predictions; show only the categories
    # with the highest scores
    if len(predicates["cls"]) > 0:
        cls_argmaxs = [obj["pred_classes"].argmax(axis=-1) for obj in scene.values()]
        cls_maxs = [obj["pred_classes"].max(axis=-1) for obj in scene.values()]
        cls_labels = {
            oi: (predicates['cls'][i], v)
            for oi, i, v in zip(scene, cls_argmaxs, cls_maxs)
        }
    else:
        cls_labels = {oi: ("n/a", float("nan")) for oi in scene}

    if len(predicates["att"]) > 0:
        att_argmaxs = [obj["pred_attributes"].argmax(axis=-1) for obj in scene.values()]
        att_maxs = [obj["pred_attributes"].max(axis=-1) for obj in scene.values()]
        att_labels = {
            oi: (predicates['att'][i], v)
            for oi, i, v in zip(scene, att_argmaxs, att_maxs)
        }
    else:
        att_labels = {oi: ("n/a", float("nan")) for oi in scene}

    if len(predicates["rel"]) > 0:
        rel_argmaxs = [
            { obj2: rels.argmax(axis=-1) for obj2, rels in obj["pred_relations"].items() }
            for obj in scene.values()
        ]
        rel_maxs = [
            { obj2: rels.max(axis=-1) for obj2, rels in obj["pred_relations"].items() }
            for obj in scene.values()
        ]
        rel_labels = {
            oi: { obj: (predicates['rel'][indices[obj]], values[obj]) for obj in indices }
            for oi, indices, values in zip(scene, rel_argmaxs, rel_maxs)
        }
    else:
        rel_labels = {
            oi: { obj2: ("n/a", float("nan")) for obj2 in scene if oi != obj2 }
            for oi in scene
        }

    rel_colors = defaultdict(lambda: random_color(rgb=True, maximum=1))

    # Show in pop-up window
    fig = plt.gcf()
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    thresholds = { "obj": 0, "rel": 0 }

    def render(obj_thresh, rel_thresh):
        # Filter predictions to visualize by objectness threshold
        objs_filtered = sorted(
            {
                oi for oi, obj in scene.items()
                if obj["pred_objectness"] > obj_thresh
            },
            key=lambda t: int(t.strip("o"))
        )

        # Boxes and classes
        v_pred = Visualizer(img, None)
        v_pred.overlay_instances(
            boxes=np.stack([scene[oi]["pred_boxes"] for oi in objs_filtered]),
            labels=[
                f"{oi} ({float(scene[oi]['pred_objectness']):.2f}): "
                f"{cls_labels[oi][0]} ({cls_labels[oi][1]:.2f}) / "
                f"{att_labels[oi][0]} ({att_labels[oi][1]:.2f})"
                for oi in objs_filtered
            ]
        )

        # Relations; show only those between objects with high objectness scores
        occurred_rels = []
        for oi in objs_filtered:
            for oj in objs_filtered:
                if oi==oj: continue

                pred, score = rel_labels[oi][oj]

                if score > rel_thresh:
                    occurred_rels.append(pred)

                    box1 = scene[oi]["pred_boxes"]
                    box2 = scene[oj]["pred_boxes"]

                    v_pred.draw_line(
                        [float(box1[0]+10), float(box2[0]+10)],
                        [float(box1[1]+10), float(box2[1]+10)],
                        color=rel_colors[pred],
                        linewidth=((score*1.5)**3)
                    )

        pred_img = v_pred.output

        # Relation legend
        pred_img.ax.legend(
            handles=[
                Patch(color=rel_colors[r], label=r) for r in set(occurred_rels)
            ]
        )

        ax.imshow(pred_img.get_image())

        fig.canvas.draw_idle()
    
    def objs_render(val):
        thresholds["obj"] = val
        render(thresholds["obj"], thresholds["rel"])
    def rels_render(val):
        thresholds["rel"] = val
        render(thresholds["obj"], thresholds["rel"])
    
    render(thresholds["obj"], thresholds["rel"])

    obj_slider = Slider(
        plt.axes([0.25, 0.05, 0.6, 0.03]),
        "Obj. score", 0.0, 1.0, valinit=thresholds["obj"]
    )
    obj_slider.on_changed(objs_render)
    rel_slider = Slider(
        plt.axes([0.25, 0.02, 0.6, 0.03]),
        "Rel. score", 0.0, 1.0, valinit=thresholds["rel"]
    )
    rel_slider.on_changed(rels_render)

    plt.show()

    return fig
