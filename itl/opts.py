import argparse

def parse_arguments():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="ITL agent engine")
    
    ## General agent-level options ##
    parser.add_argument("-dp", "--data_dir_path",
        type=str,
        default="./datasets",
        help="Path to directory where data is downloaded and stored (Default: ./datasets)")
    parser.add_argument("-ap", "--agent_model_path",
        type=str,
        help="Path to checkpoint (dir, URL, ...) from which to load agent model")
    parser.add_argument("-op", "--output_dir_path",
        type=str,
        default="./output",
        help="Path to store output files: including checkpoints, temp files, etc. (Default: ./output)")

    ## Options for vision module ##
    parser.add_argument("-vp", "--vision_model_path",
        type=str,
        default="./itl/vision/configs/SGG-RCNN-FPN-R_50_20ep.yaml",
        help="Path to pre-trained OwL-ViT model to load as vision module (Default: google/owlvit-base-patch32)")

    ## Options for language module
    parser.add_argument("-gp", "--grammar_image_path",
        type=str,
        default="./assets/grammars/erg-2018-x86-64-0.9.34.dat",
        help="Path to pre-compiled grammar image file (Default: ./assets/grammars/erg-2018-x86-64-0.9.34.dat)")
    parser.add_argument("-bp", "--ace_binary_path",
        type=str,
        default="./assets/binaries/ace-0.9.34",
        help="Path to directory containing ACE binary file (Default: ./output)")


    ## Options that specify learner's strategy
    parser.add_argument("-sm", "--strat_mismatch",
        type=str,
        default="zeroInit",
        choices=["zeroInit", "request_exmp", "request_expl"],
        help="Learner's strategy on how to address recognition mismatch:\n"
            "1) zeroInit: Zero initiative from learner whatsoever at mismatches\n"
            "2) request_exmp: Always request new exemplars of imperfect concepts\n"
            "3) request_expl: Request info in such a way that allows linguistic description\n"
            "(Default: zeroInit)")
    parser.add_argument("-sg", "--strat_generic",
        type=str,
        default="sem_only",
        choices=["sem_only", "scalar_impl"],
        help="Learner's strategy on how to interpret linguistically provided generic rules:\n"
            "1) sem_only: Read and incorporate face-value semantics of generic rules only\n"
            "2) scalar_impl: Exploit discourse context to extract scalar implicature on similarity/difference as well\n"
            "(Default: sem_only)")


    ## Options specific to experiment 1
    parser.add_argument("-x1tf", "--exp1_strat_feedback",
        type=str,
        default="min",
        choices=["min", "med", "max"],
        help="Teacher's strategy on how to provide feedback upon learner's incorrect answer:\n"
            "1) min: Absolutely minimal feedback, telling that the answer was incorrect\n"
            "2) med: 'Medium' feedback, providing correct label in addition to 'min'\n"
            "3) max: 'Maximum' feedback, providing linguistic explanation in addition to 'med'\n"
            "(Default: min)")
    parser.add_argument("-x1df", "--exp1_difficulty",
        type=str,
        default="base",
        choices=["base", "easy", "hard"],
        help="Difficulty of classification test suite:\n"
            "1) base: Baseline difficulty of 3-way none-fine-grained recognition\n"
            "2) easy: Easier 3-way fine-grained recognition\n"
            "3) hard: Harder fine-grained recognition on all types"
            "(Default: base)")
    parser.add_argument("-x1ne", "--exp1_num_episodes",
        type=int,
        default=50,
        help="Number of teaching episodes per concept, each starting with one exemplar")
    parser.add_argument("-x1ts", "--exp1_test_set_size",
        type=int,
        default=30,
        help="Number of exemplars per concept to reserve as test set for 'final exam'")
    parser.add_argument("-x1rs", "--exp1_random_seed",
        type=int,
        default=42,
        help="Random seed, which would determine how the exemplar sequence is shuffled (Default: 42)")


    return parser.parse_args()
