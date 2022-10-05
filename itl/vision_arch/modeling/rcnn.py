"""
Implement custom modules by extending detectron2-provided defaults, to be plugged
into our scene graph generation model
"""
import torch
import numpy as np
import torch.nn.functional as F
from detectron2.modeling import (
    GeneralizedRCNN,
    META_ARCH_REGISTRY
)
from detectron2.structures import Boxes, Instances
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.utils.events import get_event_storage
from detectron2.modeling.proposal_generator.proposal_utils import (
    find_top_rpn_proposals
)


@META_ARCH_REGISTRY.register()
class DualModeRCNN(GeneralizedRCNN):
    """
    GeneralizedRCNN extended to support two additional prediction modes:
        1) Accept inputs with already prepared proposal boxes; for such inputs
            self.proposal_generator is suppressed, and RoI heads are applied
            on the provided proposals only, in effect solving classification
            problem
        2) Accept search specifications for finding object(s) that best satisfy
            the conditions; (TODO: describe how this is implemented when it's done)
    """
    def inference(
        self, batched_inputs,
        detected_instances=None, do_postprocess=True, exs_cached=None, search_specs=None
    ):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        if exs_cached is None:
            features = self.backbone(images.tensor)
        else:
            # Use cached backbone output
            features = exs_cached["backbone_output"]

        if detected_instances is None:
            if "proposals" in batched_inputs[0]:
                # When "proposals" already present in input, treat them as ground-truth;
                # run roi_heads on the boxes with boxes_provided=True to prevent any of them
                # getting filtered out
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
                results, f_vecs, inc_out = self.roi_heads(
                    images, features, proposals, targets=None,
                    exs_cached=exs_cached, boxes_provided=True
                )
            else:
                # Otherwise, default behavior of the parent class
                assert self.proposal_generator is not None
                if search_specs is None:
                    # Vanilla ensemble prediction
                    proposals, _ = self.proposal_generator(images, features, None)
                    results, f_vecs, inc_out = self.roi_heads(
                        images, features, proposals, targets=None,
                        exs_cached=exs_cached, boxes_provided=False
                    )
                else:
                    # Test unfiltered sets of proposals by search specifications, then
                    # return selected proposals
                    proposals_unf, objectness_logits = self.proposals_unfiltered(
                        images, features
                    )
                    prelim_search_scores = self.search_prelim_test_scores(
                        images, features, proposals_unf, exs_cached, search_specs
                    )
                    prelim_proposals = self.search_prelim_filtering(
                        images, proposals_unf, prelim_search_scores, objectness_logits
                    )
                    final_choices = self.search_final_filtering(
                        images, features, prelim_proposals, exs_cached, search_specs
                    )
                    final_choices = Instances.cat(final_choices)

                    results, f_vecs, inc_out = self.roi_heads(
                        images, features, [final_choices], targets=None,
                        exs_cached=exs_cached, boxes_provided=True
                    )
        else:
            # No use case handled by this snippet yet...
            raise NotImplementedError
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
            f_vecs = None

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return self._postprocess(results, batched_inputs, images.image_sizes), f_vecs, inc_out
        else:
            return results, f_vecs, inc_out
    
    def proposals_unfiltered(self, images, features):
        """
        Use self.proposal_generator's components to return proposals corresponding
        to all anchors before NMS by 'objectness', so that we can use features other
        than objectness to select from them; code pilfered and modified mostly from
        detectron2.modeling.proposal_generator.rpn
        """
        pg = self.proposal_generator      # Shortcut

        features = [features[f] for f in pg.in_features]
        anchors = pg.anchor_generator(features)

        pred_objectness_logits, pred_anchor_deltas = pg.rpn_head(features)
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, pg.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        proposals = pg._decode_proposals(
            anchors, pred_anchor_deltas
        )
        proposals = {
            f: Boxes(props[0]) for f, props in zip(pg.in_features, proposals)
        }
        for props in proposals.values():
            props.clip(images.image_sizes[0])

        return proposals, pred_objectness_logits
    
    def search_prelim_test_scores(
            self, images, features, proposals, exs_cached, search_specs
        ):
        """
        Use self.roi_heads' components to test proposals with respect to provided
        search specifications. Return corresponding 'compatibility scores' for the
        (unfiltered) proposals -- one set of scores for each search spec. Also return
        intermediate feature vector outputs, so that they can be reused for testing
        finalists later.
        
        Test by prototype vectors (means of reference vectors) instead of weighted
        votes from all vectors, due to large size of unfiltered proposal sets
        """
        # Shortcuts
        rh = self.roi_heads
        dev = features[rh.box_in_features[0]].device

        search_by, search_conds = search_specs

        # Incremental relation predictions/f_vectors may be needed for computing
        # compatibility scores in certain scenarios; will need to compute them
        # in advance in such cases
        need_inc_out = any(
            any(
                cat_type=="rel" and sum(a[0]=="v" for a in arg_handles)==1
                for cat_type, arg_handles, _ in descr
            )
            for _, descr in search_conds
        )

        # In case any search spec requires relation search, with >=2 variable args?
        # Of course it's absurd to check every single proposal pair (from more than
        # ~100k unfiltered proposals? no thanks). Will need some clever way to narrow
        # down candidates -- How? we don't need to think about that for the current
        # scope of research, keep it unsupported
        need_within_rels = any(
            any(
                cat_type=="rel" and sum(a[0]=="v" for a in arg_handles)>1
                for cat_type, arg_handles, _ in descr
            )
            for _, descr in search_conds
        )
        if need_within_rels: raise NotImplementedError

        features = [features[f] for f in rh.box_in_features]
        all_boxes = Boxes.cat([proposals[f] for f in rh.box_in_features])

        # Since the size of unfiltered sets of proposals can be humongous, process
        # chunks of proposals with fixed size iteratively
        comp_scores_all = [[] for _ in range(len(search_conds))]
        CHUNK_SIZE = 8000
        for ci in range(len(all_boxes) // CHUNK_SIZE + 1):
            chunk_start = ci * CHUNK_SIZE
            chunk_end = ci * CHUNK_SIZE + CHUNK_SIZE

            boxes = all_boxes[chunk_start:chunk_end]
            proposals_objs = Instances(images.image_sizes[0])
            proposals_objs.proposal_boxes = boxes
            proposals_objs = [proposals_objs]

            box_features = rh.box_pooler(features, [boxes])
            box_features = rh.box_head(box_features)

            _, f_vecs = rh.box_predictor(
                box_features, None, (proposals_objs, None)
            )

            # Compute those incremental outputs with respect to existing predictions
            # if needed
            if need_inc_out:
                inc_out = rh._forward_box_inc_rels(
                    features, box_features, proposals_objs, f_vecs[3], exs_cached
                )[0]
            else:
                inc_out = None

            for si, (s_vars_num, descr) in enumerate(search_conds):
                # Keeping aggregate 'compatibility scores', which are boosted
                # according to how well each entity satisfies the provided search
                # conditions
                comp_scores_cond = [[] for _ in range(s_vars_num)]

                if search_by == "model":
                    raise NotImplementedError

                elif search_by == "exemplar":
                    for cat_type, arg_handles, ref_vecs in descr:
                        # Should contain at least one variable arg (for the search
                        # to make any sense)
                        arg_is_vars = [a[0]=="v" for a in arg_handles]
                        assert any(arg_is_vars)

                        # Exemplar feature vector & label vector
                        pos_exs = ref_vecs[0]; neg_exs = ref_vecs[1]

                        # Mean prototype from the reference vectors
                        pos_proto = pos_exs.mean(dim=0) if len(pos_exs) > 0 else None
                        neg_proto = neg_exs.mean(dim=0) if len(neg_exs) > 0 else None

                        if pos_proto is not None and neg_proto is not None:
                            if cat_type == "cls" or cat_type == "att":
                                # Compatibility scores by cls/att feature vector
                                assert len(arg_handles) == 1
                                cat_f_vecs = f_vecs[0] if cat_type == "cls" else f_vecs[1]
                            else:
                                # Compatibility scores by rel feature vector
                                assert cat_type == "rel"
                                assert len(arg_handles) == 2

                                v_args_num = sum(arg_is_vars)
                                if v_args_num == 1:
                                    # Recover incrementally obtained relation feature vectors,
                                    # i.e. top right and bottom left parts of 2-by-2 partitioning
                                    assert inc_out is not None
                                    N_E = len(exs_cached["detections"]); N_N = len(proposals_objs[0])
                                    D = inc_out[1].shape[-1]
                                    inc_rel_f_vecs = inc_out[1].view(-1, 2, D)

                                    v_ind = arg_is_vars.index(True); v_arg = arg_handles[v_ind]
                                    e_ind = arg_is_vars.index(False); e_arg = arg_handles[e_ind]
                                    if e_ind < v_ind:
                                        # Case (e,v): Consult 'top right' of the partition
                                        rel_f_vecs_top_right = inc_rel_f_vecs[:,0,:].view(N_E,N_N,-1)
                                        cat_f_vecs = rel_f_vecs_top_right[e_arg[1],:]
                                    else:
                                        # Case (v,e): Consult 'bottom left' of the partition
                                        assert v_ind < e_ind
                                        rel_f_vecs_bottom_left = inc_rel_f_vecs[:,1,:].view(N_N,N_E,-1)
                                        cat_f_vecs = rel_f_vecs_bottom_left[:,e_arg[1]]
                                else:
                                    assert v_args_num == 2
                                    raise NotImplementedError

                            # Softmax of negative euclidean distances between the reference
                            # vector and proposal feature vectors; for space concern, use
                            # prototype (mean) vector for search (as opposed to whole exemplar
                            # vectors as in few-shot predictions)
                            pos_dists = torch.linalg.norm(pos_proto[None,:]-cat_f_vecs, dim=-1)
                            neg_dists = torch.linalg.norm(neg_proto[None,:]-cat_f_vecs, dim=-1)
                            comp_scores = torch.stack([-pos_dists, -neg_dists], dim=-1)
                            comp_scores = F.softmax(comp_scores)[:,0]
                        else:
                            if pos_proto is None:
                                # No positive exemplars for reference, just assign very
                                # low (zero for now) probabilities for all
                                comp_scores = torch.zeros(len(box_features), device=dev)
                            else:
                                # No negative exemplars for reference, just assign very
                                # high (one for now) probabilities for all
                                assert neg_proto is None
                                comp_scores = torch.ones(len(box_features), device=dev)

                        if cat_type == "cls" or cat_type == "att":
                            comp_scores_cond[arg_handles[0][1]].append(comp_scores)
                        else:
                            comp_scores_cond[v_arg[1]].append(comp_scores)

                else:
                    raise ValueError        # Shouldn't happen

                comp_scores_all[si].append(comp_scores_cond)

        # Aggregate and flatten scores by variable; aggregate scores by multiplication
        # (in place of continuous logical connective "AND")
        scores = torch.cat([
            # comp_scores_all is organized by the order: search_spec -> chunk ->
            # search_var -> search_description
            torch.stack([
                torch.cat(per_var) for per_var in
                zip(*[
                    [torch.stack(per_var, dim=-1).prod(dim=-1) for per_var in per_chunk]
                    for per_chunk in per_spec
                ])
            ])
            for per_spec in comp_scores_all
        ])          # Now this is not a reader-friendly script...

        return scores

    def search_prelim_filtering(
        self, images, proposals_unfiltered, search_scores, objectness_logits
    ):
        """
        Use self.proposal_generator's components to filter proposals with NMS by
        search compatibility scores; return proposals corresponding
        to all anchors before NMS by 'objectness', so that we can use features other
        than objectness to select from them; code pilfered and modified mostly from
        detectron2.modeling.proposal_generator.rpn
        """
        pg = self.proposal_generator      # Shortcut

        proposals_unfiltered = [
            proposals_unfiltered[f].tensor[None] for f in pg.in_features
        ]

        offsets = [props.shape[1] for props in proposals_unfiltered]
        offsets = np.cumsum([0] + offsets)
        search_scores = [
            [per_cond[offsets[oi]:offsets[oi+1]][None] for oi in range(len(offsets)-1)]
            for per_cond in search_scores
        ]

        OBJ_REG = 0.1   # Ratio with which NMS scores (from search compatibility) will
                        # be regularized with objectness scores

        filtered_all = []
        for search_scores_per_cond in search_scores:
            f_level_nonempty = [per_f.shape[-1] > 0 for per_f in search_scores_per_cond]
            proposals_unfiltered_nonempty = [
                props for i, props in enumerate(proposals_unfiltered)
                if f_level_nonempty[i]
            ]
            search_scores_per_cond = [
                scores for i, scores in enumerate(search_scores_per_cond)
                if f_level_nonempty[i]
            ]
            objectness_scores = [
                torch.sigmoid(logits[0])[None] for i, logits in enumerate(objectness_logits)
                if f_level_nonempty[i]
            ]
            final_nms_scores = [
                ss * (1-OBJ_REG) + os * OBJ_REG
                for ss, os in zip(search_scores_per_cond, objectness_scores)
            ]

            filtered_per_cond = []
            for si in range(len(search_scores_per_cond[0])):
                # For each search spec, top-k (k=80 default) proposals that best
                # match the spec
                filtered = find_top_rpn_proposals(
                    proposals_unfiltered_nonempty,
                    [per_f_level[si][None] for per_f_level in final_nms_scores],
                    images.image_sizes,
                    pg.nms_thresh,
                    pg.pre_nms_topk[pg.training],
                    pg.post_nms_topk[pg.training],
                    pg.min_box_size,
                    pg.training,
                )[0]
                filtered.pred_objectness = filtered.objectness_logits
                filtered_per_cond.append(filtered)
            filtered_all.append(filtered_per_cond)

        return filtered_all
    
    def search_final_filtering(
            self, images, features, proposals_filtered, exs_cached, search_specs
        ):
        """
        Use self.roi_heads' components to test proposals with respect to provided
        search specifications, again with higher-accuracy test. Return final choice
        for each search spec.

        Now that we have much more manageable sizes of proposals, (re)compute the
        compatibility scores with weighted vote from all reference vectors.
        """
        # Shortcuts
        rh = self.roi_heads
        dev = features[rh.box_in_features[0]].device

        search_by, search_conds = search_specs

        # Incremental relation predictions/f_vectors may be needed for computing
        # compatibility scores in certain scenarios; will need to compute them
        # in advance in such cases
        need_inc_out = any(
            any(
                cat_type=="rel" and sum(a[0]=="v" for a in arg_handles)==1
                for cat_type, arg_handles, _ in descr
            )
            for _, descr in search_conds
        )

        # In case any search spec requires relation search, with >=2 variable args?
        # Of course it's absurd to check every single proposal pair (from more than
        # ~100k unfiltered proposals? no thanks). Will need some clever way to narrow
        # down candidates -- How? we don't need to think about that for the current
        # scope of research, keep it unsupported
        need_within_rels = any(
            any(
                cat_type=="rel" and sum(a[0]=="v" for a in arg_handles)>1
                for cat_type, arg_handles, _ in descr
            )
            for _, descr in search_conds
        )
        if need_within_rels: raise NotImplementedError

        features = [features[f] for f in rh.box_in_features]

        final_choices = []       # Return value; final choices, flattened

        # For each set of 'finalists', corresponding to each search spec...
        assert len(proposals_filtered) == len(search_conds)
        for (s_vars_num, descr), props_set in zip(search_conds, proposals_filtered):
            assert len(props_set) == s_vars_num
            comp_scores_cond = [[] for _ in range(s_vars_num)]

            for props in props_set:
                box_features = rh.box_pooler(features, [props.proposal_boxes])
                box_features = rh.box_head(box_features)

                _, f_vecs = rh.box_predictor(
                    box_features, None, ([props], None)
                )

                # Compute those incremental outputs with respect to existing predictions
                # if needed
                if need_inc_out:
                    inc_out = rh._forward_box_inc_rels(
                        features, box_features, [props], f_vecs[3], exs_cached
                    )[0]
                else:
                    inc_out = None
                
                if search_by == "model":
                    raise NotImplementedError

                elif search_by == "exemplar":
                    for cat_type, arg_handles, ref_vecs in descr:
                        # Should contain at least one variable arg (for the search
                        # to make any sense)
                        arg_is_vars = [a[0]=="v" for a in arg_handles]
                        assert any(arg_is_vars)

                        # Exemplar feature vector & label vector
                        pos_exs = ref_vecs[0]; neg_exs = ref_vecs[1]

                        if len(pos_exs) > 0 and len(neg_exs) > 0:
                            if cat_type == "cls" or cat_type == "att":
                                # Compatibility scores by cls/att feature vector
                                assert len(arg_handles) == 1
                                cat_f_vecs = f_vecs[0] if cat_type == "cls" else f_vecs[1]
                            else:
                                # Compatibility scores by rel feature vector
                                assert cat_type == "rel"
                                assert len(arg_handles) == 2

                                v_args_num = sum(arg_is_vars)
                                if v_args_num == 1:
                                    # Recover incrementally obtained relation feature vectors,
                                    # i.e. top right and bottom left parts of 2-by-2 partitioning
                                    assert inc_out is not None
                                    N_E = len(exs_cached["detections"]); N_N = len(props)
                                    D = inc_out[1].shape[-1]
                                    inc_rel_f_vecs = inc_out[1].view(-1, 2, D)

                                    v_ind = arg_is_vars.index(True); v_arg = arg_handles[v_ind]
                                    e_ind = arg_is_vars.index(False); e_arg = arg_handles[e_ind]
                                    if e_ind < v_ind:
                                        # Case (e,v): Consult 'top right' of the partition
                                        rel_f_vecs_top_right = inc_rel_f_vecs[:,0,:].view(N_E,N_N,-1)
                                        cat_f_vecs = rel_f_vecs_top_right[e_arg[1],:]
                                    else:
                                        # Case (e,v): Consult 'bottom left' of the partition
                                        assert v_ind < e_ind
                                        rel_f_vecs_bottom_left = inc_rel_f_vecs[:,1,:].view(N_N,N_E,-1)
                                        cat_f_vecs = rel_f_vecs_bottom_left[:,e_arg[1]]
                                else:
                                    assert v_args_num == 2
                                    raise NotImplementedError
                                
                            # Softmax of negative euclidean distances between the reference
                            # vector and proposal feature vectors; for space concern, use
                            # prototype (mean) vector for search (as opposed to whole exemplar
                            # vectors as in few-shot predictions)
                            pos_dists = torch.linalg.norm(
                                pos_exs[None,:,:]-cat_f_vecs[:,None,:], dim=-1
                            )
                            neg_dists = torch.linalg.norm(
                                neg_exs[None,:,:]-cat_f_vecs[:,None,:], dim=-1
                            )
                            comp_scores = torch.cat([-pos_dists, -neg_dists], dim=-1)
                            comp_scores = F.softmax(comp_scores, dim=-1)
                            comp_scores = comp_scores[:,:len(pos_exs)].sum(dim=-1)
                        else:
                            if len(pos_exs) == 0:
                                # No positive exemplars for reference, just assign very
                                # low (zero for now) probabilities for all
                                comp_scores = torch.zeros(len(cat_f_vecs), device=dev)
                            else:
                                # No negative exemplars for reference, just assign very
                                # high (one for now) probabilities for all
                                assert len(neg_exs) == 0
                                comp_scores = torch.ones(len(cat_f_vecs), device=dev)
                        
                        if cat_type == "cls" or cat_type == "att":
                            comp_scores_cond[arg_handles[0][1]].append(comp_scores)
                        else:
                            comp_scores_cond[v_arg[1]].append(comp_scores)

                else:
                    raise ValueError        # Shouldn't happen

            comp_scores_agg = [
                torch.stack(per_var, dim=-1).prod(dim=-1)
                for per_var in comp_scores_cond
            ]
            final_choices += [
                props[per_var.max(dim=0).indices.item()]
                for per_var in comp_scores_agg
            ]
        
        return final_choices

    def visualize_training(self, batched_inputs, proposals):
        """
        Minimal modification for moving input["instances"].gt_boxes to cpu device
        before v_gt.overlay_instances(), and *NOT* transposing image dimensions
        """
        from detectron2.utils import comm
        from detectron2.utils.visualizer import Visualizer

        if not comm.is_main_process():
            return

        storage = get_event_storage()
        max_vis_prop = 30

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes.tensor.cpu())
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch
