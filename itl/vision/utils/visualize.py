"""
For visualizing predictions from scene graph generation models, using detectron2
visualization toolkits.
"""
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.widgets import Slider
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.colormap import random_color
from detectron2.data.detection_utils import convert_image_to_rgb


def visualize_sg_predictions(inputs, predictions, predicates):
    """
    Args:
        inputs (list): a list that contains input to the model.
        predictions (list): a list that contains final predictions from scene graph
            generation models for the input. Should have the same length as inputs.
    """
    for inp, pred in zip(inputs, predictions):
        img = inp["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), "BGR")
        img = cv2.resize(img, dsize=(inp["width"], inp["height"]))

        cls_maxs = pred.pred_classes.max(dim=1)
        cls_maxs = [
            f"{predicates['cls'][i]} ({v:.2f})"
            for i, v in zip(cls_maxs.indices, cls_maxs.values)
        ]
        att_maxs = pred.pred_attributes.max(dim=1)
        att_maxs = [
            f"{predicates['att'][i]} ({v:.2f})"
            for i, v in zip(att_maxs.indices, att_maxs.values)
        ]
        rel_maxs = pred.pred_relations.max(dim=2)

        rel_colors = [
            random_color(rgb=True, maximum=1) for _ in range(len(predicates['rel']))
        ]

        # Show in pop-up window
        fig = plt.gcf()
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        thresholds = { "obj": 0.7, "rel": 0.7 }

        def render(obj_thresh, rel_thresh):
            thresh = obj_thresh
            thresh_filter = (pred.pred_objectness > thresh).view(-1)
            thresh_topk = int(thresh_filter.sum())

            # Boxes and classes
            v_pred = Visualizer(img, None)
            v_pred.overlay_instances(
                boxes=pred.pred_boxes[thresh_filter].tensor.cpu().numpy(),
                labels=[f"o{oi} ({float(obj_label):.2f}): {cls_label} / {att_label}"
                    for oi, (obj_label, cls_label, att_label)
                    in enumerate(zip(
                        pred.pred_objectness[thresh_filter],
                        cls_maxs[:thresh_topk],
                        att_maxs[:thresh_topk]
                    ))
                ]
            )

            # Relations; show only those between objects with high objectness scores
            occurred_rels = []
            for i in range(thresh_topk):
                for j in range(thresh_topk):
                    if i==j: continue

                    rel_ind = int(rel_maxs.indices[i,j])
                    score = float(rel_maxs.values[i,j])

                    if score > rel_thresh:
                        occurred_rels.append(rel_ind)

                        obj1 = pred.pred_boxes[i].tensor[0]
                        obj2 = pred.pred_boxes[j].tensor[0]
                        v_pred.draw_line(
                            [float(obj1[0]+10), float(obj2[0]+10)],
                            [float(obj1[1]+10), float(obj2[1]+10)],
                            color=rel_colors[rel_ind],
                            linewidth=((score*2)**4)
                        )

            pred_img = v_pred.output

            # Relation legend
            pred_img.ax.legend(
                handles=[
                    Patch(color=rel_colors[r], label=predicates['rel'][r])
                    for r in set(occurred_rels)
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
