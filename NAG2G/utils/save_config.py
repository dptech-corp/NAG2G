import configparser

list_ = [
    "batch_size",
    "batch_size_valid",
    "data",
    "tensorboard_logdir",
    "bf16",
    "num_workers",
    "required_batch_size_multiple",
    "valid_subset",
    "label_prob",
    "mid_prob",
    "mid_upper",
    "mid_lower",
    "plddt_loss_weight",
    "pos_loss_weight",
    "shufflegraph",
    "infer_save_name",
    "decoder_attn_from_loader",
    "infer_step",
    "config_file",
    "path",
    "results_path",
    "beam_size",
    "search_strategies",
    "len_penalty",
    "temperature",
    "beam_size_second",
    "beam_head_second",
    "nprocs_per_node",
    "data_buffer_size",
    "distributed_rank",
    "distributed_port",
    "distributed_world_size",
    "distributed_backend",
    "distributed_init_method",
    "distributed_no_spawn",
    "lr_shrink"
]

def add_config_save_args(parser):
    parser.add_argument(
        "--config_file",
        type=str,
        default="",
        help="Path to configuration file",
    )


def save_config(args):
    save_path = args.config_file
    if save_path == "":
        return
    args_dict = vars(args)
    args_dict = {k: str(v) for k, v in args_dict.items()}
    config = configparser.ConfigParser()
    config.read_dict({"DEFAULT": args_dict})
    with open(save_path, "w") as f:
        config.write(f)


def read_config(args):
    if args.config_file != "":
        config = configparser.ConfigParser()
        config.read(args.config_file)

        for arg in vars(args):
            if arg in list_:
                continue
            value = config["DEFAULT"].get(arg)
            if arg == "noise_scale":
                value = 0.0
            type_arg = type(getattr(args, arg))
            if type_arg == type(None):
                if arg in [
                    "log_format",
                    "distributed_init_method",
                    "path",
                    "results_path",
                ]:
                    type_arg = type("abc")
                elif arg in [
                    "fp16_scale_window",
                    "batch_size",
                    "fixed_validation_seed",
                    "batch_size_valid",
                    "max_valid_steps",
                    "force_anneal",
                ]:
                    type_arg = type(123)
                elif arg in ["threshold_loss_scale"]:
                    type_arg = type(1.2)
                else:
                    raise
            if value is not None and value != "None":
                # print(arg, type_arg, value)
                if type_arg == type(True):
                    if value == "True":
                        setattr(args, arg, True)
                    elif value == "False":
                        setattr(args, arg, False)
                else:
                    setattr(args, arg, type_arg(value))

    return args
