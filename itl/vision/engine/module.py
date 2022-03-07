"""
Wrap the custom extension of detectron2 GeneralizedRCNN for scene graph generation
as PyTorch Lightning LightningModule, implementing training & inference logics.
Taken partially and extended from tools/lightning_train_net.py at detectron2 main repo.
"""
import os
import time
import uuid
import logging
import weakref
from collections import defaultdict, OrderedDict

import wandb
import numpy as np
import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage, CommonMetricPrinter
from detectron2.engine import (
    SimpleTrainer,
    hooks
)
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.evaluation import print_csv_format
from detectron2.evaluation.testing import flatten_results_dict
from pytorch_lightning import LightningModule

from .evaluator import VGDetEvaluator, VGClfEvaluator, VGFewShotEvaluator
from .writer import WandbWriter, TBXWriter
from ..modeling import MetaLearner
from ..utils.visualize import visualize_sg_predictions


logger = logging.getLogger("vision.module.model")
logger.setLevel(logging.INFO)

class SceneGraphGenerator(LightningModule):
    """
    Implement LightningModule hooks calling appropriate detectron2 methods
    """
    def __init__(self, cfg, offline=False):
        super().__init__()

        self.cfg = cfg
        self.offline = offline

        self.storage = None

        # Register our custom modules
        if self.cfg.MODEL.META_ARCHITECTURE == "DualModeRCNN":
            from ..modeling.rcnn import DualModeRCNN
        if self.cfg.MODEL.ROI_HEADS.NAME == "SceneGraphROIHeads":
            from ..modeling.roi_heads import SceneGraphROIHeads

        # Base batch-learned SGG model
        self.base_model = build_model(self.cfg)
        self.base_model.backbone.bottom_up.requires_grad_(False)   # Freeze ResNet layers

        # Few-shot-learned meta learder
        self.meta = MetaLearner(
            code_size=self.base_model.roi_heads.box_predictor.CODE_SIZE,
            loss_type=cfg.MODEL.META.FEW_SHOT_LOSS_TYPE
        )

        self.start_iter = 0
        self.max_iter = self.cfg.SOLVER.MAX_ITER

        self.wandb_id = None
        self.few_shot = None
    
    # exp_name property, whose setter randomly initializes and sets value if called
    # with 'invalid' arg (i.e. None or empty string)
    @property
    def exp_name(self):
        return self._exp_name
    @exp_name.setter
    def exp_name(self, en):
        if not en:
            en = str(uuid.uuid4().hex)[:12]
        self._exp_name = en

    def on_save_checkpoint(self, ckpt):
        # Delete any existing checkpoint files starting with "last-"
        ckpt_dir = os.path.join(self.cfg.OUTPUT_DIR, self.exp_name, "checkpoints")
        if comm.is_main_process() and os.path.isdir(ckpt_dir):
            for f in os.listdir(ckpt_dir):
                if f.startswith("last-"):
                    os.remove(os.path.join(ckpt_dir, f))

        ckpt["iteration"] = self.storage.iter

        md = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        ckpt["predicates"] = {
            "cls": md.classes,
            "att": md.attributes,
            "rel": md.relations
        }

    def on_load_checkpoint(self, ckpt):
        self.start_iter = ckpt["iteration"]

    def setup(self, stage):
        self.iteration_timer = hooks.IterationTimer()
        self.iteration_timer.before_train()
        self.data_start = time.perf_counter()
        self.writers = None

        if self.few_shot:
            # Freeze base model (except box classifiers......?)
            self.base_model.requires_grad_(False)
            # self.base_model.roi_heads.box_predictor.cls_codes.requires_grad_(True)
            # self.base_model.roi_heads.box_predictor.att_codes.requires_grad_(True)
            # self.base_model.roi_heads.box_predictor.rel_codes.requires_grad_(True)

    def training_step(self, batch, batch_idx):
        data_time = time.perf_counter() - self.data_start
        # Need to manually enter/exit since trainer may launch processes
        # This ideally belongs in setup, but setup seems to run before processes are spawned
        if self.storage is None:
            self.storage = EventStorage()
            self.storage.__enter__()
            self.storage.iter = self.start_iter
            self.iteration_timer.trainer = weakref.proxy(self)
            self.iteration_timer.before_step()

        if self.writers is None:
            if comm.is_main_process():
                self.writers = [
                    CommonMetricPrinter(self.max_iter),
                ]

                if not self.offline:
                    wr = WandbWriter(
                        self, window_size=50, resume_id=self.wandb_id
                    )
                else:
                    wr = TBXWriter(
                        self, window_size=50
                    )

                self.writers.append(wr)
            else:
                self.writers = {}

        if self.few_shot:
            loss_dict = self.meta(batch, self.base_model)
        else:
            loss_dict = self.base_model(batch)

        loss_dict = {f"loss/{k}": v for k, v in loss_dict.items()}
        SimpleTrainer.write_metrics(loss_dict, data_time)

        opt = self.optimizers()
        self.storage.put_scalar(
            "lr", opt.param_groups[self._best_param_group_id]["lr"], smoothing_hint=False
        )
        self.iteration_timer.after_step()
        self.storage.step()
        # A little odd to put before step here, but it's the best way to get a proper timing
        self.iteration_timer.before_step()

        if self.storage.iter % 50 == 0:
            for writer in self.writers:
                writer.write()
        return sum(loss_dict.values())

    def training_step_end(self, step_output):
        self.data_start = time.perf_counter()
        return step_output

    def training_epoch_end(self, epoch_outputs):
        self.iteration_timer.after_train()
        for writer in self.writers:
            writer.write()
            writer.close()
        self.storage.__exit__(None, None, None)
    
    def on_train_end(self) -> None:
        exp_dir_path = os.path.join(self.cfg.OUTPUT_DIR, self.exp_name)
        ckpt_dir_path = os.path.join(exp_dir_path, 'checkpoints')

        if comm.is_main_process():
            if not self.offline:
                wandb.save(f"{ckpt_dir_path}/*", base_path=exp_dir_path)
                wandb.finish()
                self.wandb_id = None    # Reset

    def on_validation_epoch_start(self):
        self.on_test_epoch_start()    # Identical as testing

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.test_step(batch, batch_idx, dataloader_idx)    # Identical as testing

    def validation_epoch_end(self, epoch_outputs):
        # This part corresponds to _process_dataset_evaluation_results() method in the
        # original file
        results = OrderedDict()
        for dataset_name in self.cfg.DATASETS.TEST:
            if self.few_shot:
                results[dataset_name] = self._evaluators[0].evaluate()
            else:
                # Only aggregate stats are used
                _, det_res_avg = self._evaluators[0].evaluate()
                _, clf_res_avg = self._evaluators[1].evaluate()
                det_res_avg = defaultdict(dict, det_res_avg)
                clf_res_avg = defaultdict(dict, clf_res_avg)
                results[dataset_name] = {
                    task: {**det_res_avg[task], **clf_res_avg[task]}
                    for task in set(det_res_avg)|set(clf_res_avg)
                }    # Set union on tasks maybe an overkill, but better be safe than sorry...

            if comm.is_main_process():
                print_csv_format(results[dataset_name])
        # ... said part end

        if len(results) == 1:
            results = list(results.values())[0]
        
        flattened_results = flatten_results_dict(results)
        for k, v in flattened_results.items():
            try:
                v = float(v)
            except Exception as e:
                raise ValueError(
                    "[EvalHook] eval_function should return a nested dict of float. "
                    "Got '{}: {}' instead.".format(k, v)
                ) from e

        if not self.trainer.sanity_checking:
            self.storage.put_scalars(**flattened_results, smoothing_hint=False)
            for writer in self.writers:
                writer.write()
            
            self.log("val_acc_mean", sum(flattened_results.values()))
    
    def on_test_epoch_start(self):
        # This part corresponds to _reset_dataset_evaluators() method in the original file
        self._evaluators = []
        for dataset_name in self.cfg.DATASETS.TEST:
            if self.few_shot:
                fs_evaluator = VGFewShotEvaluator()
                fs_evaluator.reset()
                self._evaluators = [fs_evaluator]
            else:
                # Here, we use two custom evaluators (detection vs. classification mode)
                det_evaluator = VGDetEvaluator(dataset_name, self.cfg.OUTPUT_DIR)
                det_evaluator.reset()
                self._evaluators.append(det_evaluator)

                clf_evaluator = VGClfEvaluator(dataset_name, self.cfg.OUTPUT_DIR)
                clf_evaluator.reset()
                self._evaluators.append(clf_evaluator)
        # ... said part end

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if not isinstance(batch, list):
            batch = [batch]

        if self.few_shot:
            outputs = self.meta(batch, self.base_model)
        else:
            outputs = self.base_model(batch)

        self._evaluators[dataloader_idx].process(batch, outputs)

    def test_epoch_end(self, epoch_outputs):
        # (Maybe this should be factored out)
        # This part corresponds to _process_dataset_evaluation_results() method in the
        # original file
        results_per_cat = OrderedDict()
        results_avg = OrderedDict()
        for dataset_name in self.cfg.DATASETS.TEST:
            if self.few_shot:
                results_avg[dataset_name] = self._evaluators[0].evaluate()
            else:
                det_res_per_cat, det_res_avg = self._evaluators[0].evaluate()
                clf_res_per_cat, clf_res_avg = self._evaluators[1].evaluate()

                det_res_per_cat = defaultdict(dict, det_res_per_cat)
                clf_res_per_cat = defaultdict(dict, clf_res_per_cat)
                results_per_cat[dataset_name] = {
                    f"{task}_test": {**det_res_per_cat[task], **clf_res_per_cat[task]}
                    for task in set(det_res_per_cat)|set(clf_res_per_cat)
                }

                det_res_avg = defaultdict(dict, det_res_avg)
                clf_res_avg = defaultdict(dict, clf_res_avg)
                results_avg[dataset_name] = {
                    f"{task}_test": {**det_res_avg[task], **clf_res_avg[task]}
                    for task in set(det_res_avg)|set(clf_res_avg)
                }
        # ... said part end

        if len(results_avg) == 1:
            results_per_cat = list(results_per_cat.values())[0]
            results_avg = list(results_avg.values())[0]
        
        flattened_results_per_cat = flatten_results_dict(results_per_cat)
        flattened_results_avg = flatten_results_dict(results_avg)
        for k, v in flattened_results_avg.items():
            try:
                v = float(v)
            except Exception as e:
                raise ValueError(
                    "[EvalHook] eval_function should return a nested dict of float. "
                    "Got '{}: {}' instead.".format(k, v)
                ) from e
        
        md = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])

        # Categories sorted by occurrence freq
        cls_sort_inds = np.argsort(-np.array(md.classes_counts))
        att_sort_inds = np.argsort(-np.array(md.attributes_counts))
        rel_sort_inds = np.argsort(-np.array(md.relations_counts))

        # String names & frequencies by sorted indices
        cls_names_sorted = [md.classes[i] for i in cls_sort_inds]
        att_names_sorted = [md.attributes[i] for i in att_sort_inds]
        rel_names_sorted = [md.relations[i] for i in rel_sort_inds]
        cls_counts_sorted = [md.classes_counts[i] for i in cls_sort_inds]
        att_counts_sorted = [md.attributes_counts[i] for i in att_sort_inds]
        rel_counts_sorted = [md.relations_counts[i] for i in rel_sort_inds]

        # Collect all metrics by sorted indices
        cls_res_sorted = {
            k.split("_")[-1]: np.array(v)[cls_sort_inds]
            for k, v in flattened_results_per_cat.items() if "cls" in k
        }
        att_res_sorted = {
            k.split("_")[-1]: np.array(v)[att_sort_inds]
            for k, v in flattened_results_per_cat.items() if "att" in k
        }
        rel_res_sorted = {
            k.split("_")[-1]: np.array(v)[rel_sort_inds]
            for k, v in flattened_results_per_cat.items() if "rel" in k
        }

        # Evaluation metric tables to log
        cls_table = wandb.Table(
            data=list(zip(cls_names_sorted, cls_counts_sorted, *cls_res_sorted.values())),
            columns=["Name", "Frequency"]+list(cls_res_sorted.keys())
        )
        att_table = wandb.Table(
            data=list(zip(att_names_sorted, att_counts_sorted, *att_res_sorted.values())),
            columns=["Name", "Frequency"]+list(att_res_sorted.keys())
        )
        rel_table = wandb.Table(
            data=list(zip(rel_names_sorted, rel_counts_sorted, *rel_res_sorted.values())),
            columns=["Name", "Frequency"]+list(rel_res_sorted.keys())
        )

        # Category embedding tables to log
        bp = self.base_model.roi_heads.box_predictor
        cls_embs = wandb.Table(
            data=[
                [md.classes[i]]+c.tolist()
                for i, c in enumerate(bp.cls_codes.weight.detach())
            ],
            columns=["Name"]+[f"D{d}" for d in range(bp.CODE_SIZE)]
        )
        att_embs = wandb.Table(
            data=[
                [md.attributes[i]]+a.tolist()
                for i, a in enumerate(bp.att_codes.weight.detach())
            ],
            columns=["Name"]+[f"D{d}" for d in range(bp.CODE_SIZE)]
        )
        rel_embs = wandb.Table(
            data=[
                [md.relations[i]]+r.tolist()
                for i, r in enumerate(bp.rel_codes.weight.detach())
            ],
            columns=["Name"]+[f"D{d}" for d in range(bp.CODE_SIZE)]
        )
        
        if comm.is_main_process():
            if not self.offline and self.wandb_id:
                # Writing evaluation metrics and some analysis results on test data
                # to the loaded W&B run
                from dotenv import load_dotenv
                load_dotenv("wandb.env")

                wandb.init(resume="must", id=self.wandb_id)

                for k, v in flattened_results_avg.items():
                    wandb.run.summary[k] = v
                
                wandb.log({
                    "cls_table": cls_table,
                    "att_table": att_table,
                    "rel_table": rel_table,

                    "cls_mAP@0.5": wandb.plot.scatter(
                        cls_table, "Frequency", "mAP@0.5", title="Class mAP@0.5"
                    ),
                    "cls_recall@100": wandb.plot.scatter(
                        cls_table, "Frequency", "recall@100", title="Class recall@100"
                    ),
                    "att_mAP@0.5": wandb.plot.scatter(
                        att_table, "Frequency", "mAP@0.5", title="Attribute mAP@0.5"
                    ),
                    "att_recall@100": wandb.plot.scatter(
                        att_table, "Frequency", "recall@100", title="Attribute recall@100"
                    ),
                    "rel_recall@100": wandb.plot.scatter(
                        rel_table, "Frequency", "recall@100", title="Relation recall@100"
                    ),

                    "cls_embs": cls_embs,
                    "att_embs": att_embs,
                    "rel_embs": rel_embs
                })

                wandb.finish()
    
    def predict_step(self, batch, batch_idx):
        assert hasattr(self, "predicates"), "Need predicate name info for predict_step()"

        if not isinstance(batch, list):
            batch = [batch]
        outputs = self.base_model.inference(batch, do_postprocess=False)
        visualize_sg_predictions(batch, outputs, self.predicates)

    def configure_optimizers(self):
        optimizer = build_optimizer(self.cfg, self)
        self._best_param_group_id = hooks.LRScheduler.get_best_param_group_id(optimizer)
        scheduler = build_lr_scheduler(self.cfg, optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)
