import argparse

def parse_arguments():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="ITL agent engine")

    ## Options for training/testing vision module ##
    # Arguments that are fed to VisionModule
    parser.add_argument("-dp", "--data_dir_path",
        type=str,
        default="./datasets/visual_genome",
        help="Path to directory where data is downloaded and stored (Default: ./datasets/visual_genome)")
    parser.add_argument("-cp", "--config_file_path",
        type=str,
        default="./itl/vision/configs/SGG-RCNN-FPN-R_50_20ep.yaml",
        help="Path to detectron2 configuration file (Default: ./itl/vision/configs/SGG-RCNN-FPN-R_50_20ep.yaml")
    parser.add_argument("-op", "--output_dir_path",
        type=str,
        default="./output",
        help="Path to store output files: including checkpoints, temp files, etc. (Default: ./output)")
    parser.add_argument("-lp", "--load_checkpoint_path",
        type=str,
        help="Path to checkpoint (dir, URL, ...) from which to load model")
    parser.add_argument("-of", "--offline",
        action="store_true",
        help="Run offline mode; use Tensorboard instead of W&B for logging, checkpoint loading from URL disabled")
    parser.add_argument("-ng", "--num_gpus",
        type=int,
        default=1,
        help="Number of GPU workers to use (Default: 1)")
    parser.add_argument("-nd", "--num_dataloader_workers",
        type=int,
        default=1,
        help="Number of total (across GPUs) DataLoader workers to use (Default: 1)")
    parser.add_argument("-ni", "--num_imgs",
        type=int,
        default=10807,
        help="Number of VG images to download and use (Default: 10807; 10%% of VG))")
    parser.add_argument("-mi", "--max_iter",
        type=int,
        default=270000,
        help="Max training iterations (Default: 270000)")

    # Hyperparameters that control positive weight distribution in batch training
    parser.add_argument("-we", "--weight_exponent",
        type=float,
        default=0,
        help="Exponent that controls variance of distributions of inverse-frequency of categories; in effect, values "
            "lower than 1 smooth the distribution by suppressing/augmenting larger/smaller weights (Default: 0)")
    parser.add_argument("-wm", "--weight_target_mean",
        type=float,
        default=200,
        help="Target mean of the weight distributions; scaling factors are computed such that the mean of the (smoothed) "
            "distribution is adjusted to the provided value (Default: 200)")

    # Arguments that are fed to training scripts
    parser.add_argument("-en", "--exp_name",
        type=str,
        help="Training experiment name")
    parser.add_argument("-rt", "--resume_training",
        action="store_true",
        help="Whether to resume training")

    # Options primarily for 'in vivo' use, with no further batch training happening
    # (only inference and incremental few-shot registration)
    parser.add_argument("-ic", "--initialize_categories",
        action="store_true",
        help="If true, bomb the category prediction heads, leaving only the feature "
            "extractor backbone and objectness & bbox dimension prediction heads")


    ## Options for language module
    parser.add_argument("-gp", "--grammar_image_path",
        type=str,
        default="./assets/grammars/erg-2018-x86-64-0.9.34.dat",
        help="Path to pre-compiled grammar image file (Default: ./assets/grammars/erg-2018-x86-64-0.9.34.dat)")
    parser.add_argument("-ap", "--ace_binary_path",
        type=str,
        default="./assets/binaries/ace-0.9.34",
        help="Path to directory containing ACE binary file (Default: ./output)")


    ## Options that specify learner's strategy
    parser.add_argument("-sm", "--strat_mismatch",
        type=str,
        default="zero_init",
        choices=["zero_init", "request_exmp", "request_expl"],
        help="Learner's strategy on how to address recognition mismatch:\n"
            "1) zero_init: Zero initiative from learner whatsoever at mismatches\n"
            "2) request_exmp: Always request new exemplars of imperfect concepts\n"
            "3) request_expl: Request info in such a way that allows linguistic description\n"
            "(Default: zero_init)")
    parser.add_argument("-sg", "--strat_generic",
        type=str,
        default="sem_only",
        choices=["sem_only", "scalar_impl"],
        help="Learner's strategy on how to interpret linguistically provided generic rules:\n"
            "1) sem_only: Read and incorporate face-value semantics of generic rules only\n"
            "2) scalar_impl: Exploit discourse context to extract scalar implicature on similarity/difference as well\n"
            "(Default: sem_only)")


    return parser.parse_args()
