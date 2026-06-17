import argparse
from pathlib import Path

from sima_lmm.mole.modalixboard import ModalixBoard
from sima_lmm.mole.mole import perf_bench, prepare_mole, run

def main():
    parser = argparse.ArgumentParser(description="LLM benchmark tool")

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("model_id", type=str, help="Model ID")
    common_parser.add_argument(
        "-n", "--num_samples", type=int,
        help="Number of samples to run. Run the full set if not specified"
    )
    common_parser.add_argument(
        "--max_num_tokens", type=int, default=2048, help="Max number of tokens"
    )
    common_parser.add_argument("-o", "--output", type=Path, help="Output directory")
    common_parser.add_argument("--board_ip", type=str, help="Board IP or host name")
    common_parser.add_argument("--board_port", type=int, default=8002, help="Board port")
    common_parser.add_argument(
        "--board_start_server", type=bool, action=argparse.BooleanOptionalAction, default=True,
        help="Start board server"
    )
    common_parser.add_argument("--board_venv_path", type=str, help="Python venv path on the board")
    common_parser.add_argument("--board_model", type=str, help="Model ID or path on the board")
    common_parser.add_argument(
        "--board_ssh_user", type=str, default="sima", help="Board user name for SSH connection"
    )
    common_parser.add_argument(
        "--board_ssh_pass", type=str, default=None, help="Board user password for SSH connection."
    )
    common_parser.add_argument(
        "--modalix_group_prefill", action=argparse.BooleanOptionalAction, default=False,
        help="Use grouped prefix prefill for Modalix loglikelihood scoring"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    acc_parser = subparsers.add_parser(
        "accuracy", parents=[common_parser], formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Benchmark accuracy"
    )
    acc_parser.set_defaults(max_new_tokens=0)
    acc_parser.add_argument(
        "-b", "--backend", type=str, choices=["hf", "modalix"], required=True,
        help="Model evaluation backend"
    )
    acc_parser.add_argument(
        "-t", "--task", type=str, nargs="+", required=True, help="Tasks to be done"
    )

    perf_parser = subparsers.add_parser(
        "perf", parents=[common_parser], formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Benchmark performance"
    )
    perf_parser.set_defaults(backend="modalix", task=[])
    perf_parser.add_argument(
        "--max_new_tokens", type=int, default=256, help="Max number of new tokens"
    )
    args = parser.parse_args()

    # Create and validate mole config.
    board = None
    if args.command == "perf" or args.backend == "modalix":
        board = ModalixBoard(
            address=args.board_ip, port=args.board_port, model=args.board_model,
            ssh_user=args.board_ssh_user, ssh_password=args.board_ssh_pass,
            venv_path=args.board_venv_path
        )

    config = prepare_mole(
        model_id=args.model_id,
        backend=args.backend,
        tasks=args.task,
        output_path=args.output,
        board=board,
        limit=args.num_samples,
        do_start_server=args.board_start_server,
        do_perf=args.command == "perf",
        max_num_tokens=args.max_num_tokens,
        max_new_tokens=args.max_new_tokens,
        modalix_group_prefill=args.modalix_group_prefill
    )
    if config is None:
        parser.error("Invalid argument combinations")

    if config.do_perf:
        # Benchmark performance.
        perf_bench(config=config)
    else:
        # Benchmark accuracy.
        run(config=config)


if __name__ == "__main__":
    main()
