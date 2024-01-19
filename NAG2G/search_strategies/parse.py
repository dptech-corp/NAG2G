def add_search_strategies_args(parser, train=False, gen=False):
    group = parser.add_argument_group("beam search")
    group.add_argument(
        "--beam-size", default=10, type=int, metavar="N", help="beam size for inference"
    )
    group.add_argument(
        "--search_strategies",
        type=str,
        default="SequenceGeneratorBeamSearch",
        help="beam size for inference",
    )
    group.add_argument(
        "--len-penalty",
        default=1.0,
        type=float,
        metavar="N",
        help="Length penalty in beam search for inference",
    )
    group.add_argument(
        "--temperature",
        default=1.0,
        type=float,
        metavar="N",
        help="Temperature in beam search for inference",
    )

    # for two stage
    group.add_argument(
        "--beam-size-second", default=5, type=int, metavar="N", help="beam size for second stage inference"
    )
    group.add_argument(
        "--beam-head-second", default=3, type=int, metavar="N", help="beam head for second stage inference"
    )
    return group
