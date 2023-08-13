import argparse

from matese import predict
from matese.utils import utils


root_dir = utils.get_root_dir()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        help="The path to a file containing source sentences, one per line",
        type=str,
        default=str(root_dir.joinpath("data", "sources.txt")),
    )
    parser.add_argument(
        "--candidates",
        help="The path to a file containing candidate translations, one per line",
        type=str,
        default=str(root_dir.joinpath("data", "candidates.txt")),
    )
    parser.add_argument(
        "--references",
        help="The path to a file containing reference translations, one per line",
        type=str,
        default=str(root_dir.joinpath("data", "references.txt")),
    )
    parser.add_argument(
        "--output",
        help="The path to a file that will be used as prefix to create the path of spans and scores files",
        type=str,
        default=str(root_dir.joinpath("data", "output"))
    )
    parser.add_argument(
        "--metric",
        help="Name of the type of metric you want to use (has to be in ['matese', 'matese-qe', 'matese-en'])",
        type=str,
        default='matese',
    )
    parser.add_argument(
        "--cpu",
        help="Perform inference on CPU",
        action="store_true",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    predict.main(args)


if __name__ == "__main__":
    main()
