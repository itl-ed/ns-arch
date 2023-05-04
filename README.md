# A Neurosymbolic Architecture for Interactive Symbol Grounding

Codebase, datasets and trained vision model weights for the paper "Interactive Acquisition of Fine-grained Visual Concepts by Exploiting Semantics of Generic Characterizations in Discourse"; accepted to the 15th International Conference on Computational Semantics (IWCS 2023).

## Datasets and models

- Tabletop domain datasets, images + annotations: [Google drive link](https://drive.google.com/file/d/1BOHNoiCFCmLYRTPkzdneDkjoKwc_IgJo/view?usp=share_link)
- Model weights for the custom extension modules added to Deformable DETR: [Google drive link](https://drive.google.com/file/d/1oWxQwfcx9GzxHox28q5qcPuQ_EUWe7Nt/view?usp=share_link)
- (Datasets for training the custom feature extractor module, i.e. Visual Genome, not directly uploaded. Refer to `tools/vision/prepare_data.py` script for starts if interested in training the extension module from scratch with VG data.)

## Some important command-line arguments

(Arguments are configured with `hydra`; see `itl/configs` directory for how they are set up if you are familiar with `hydra`)
- `+vision.model.fs_model={PATH_TO_MODEL_CKPT}`: path to the feature extractor module weights
- `+agent.model_path={PATH_TO_MODEL_CKPT}`: path to the agent model with part/attribute concepts injected with `tools/exp1/inject_concepts.py` script
- `seed={N}`: integer random seed
- `exp1.strat_feedback=[minHelp/medHelp/maxHelp]`: Teacher's strategy for providing feedback upon student's incorrect answers to episode-initial probing questions
- `agent.strat_generic=[semOnly/semNeg/semNegScal]`: Student's strategy for interpreting generic characterizations in discourse context

## Checklist for running experiments

(Checklist items not ordered)
- Run `pip install -r requirements.txt` to install Python packages.
- Download tabletop domain datasets and put them in `{PROJECT_ROOT}/datasets/tabletop` directory.
- Download model weights for the custom extension modules for few-shot feature extraction (added to Deformable DETR) and put them in `{PROJECT_ROOT}/assets/vision_models` directory.
- Run `bash tools/lang/get_grammar.sh` to download ERG grammar image and ACE parser software binary prior to any experiments involving `maxHelp` teacher strategy config.
- Run `python tools/exp1/inject_concepts.py` for injection of part & attribute concepts prior to experiments involving `maxHelp` teacher strategy config.

## Citation

(To be updated)
