import argparse

from gdigs_low_pipe.pipeline import eq2gal


def parse_args():
    """
    Argument parser.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=str, nargs="+",
                        help="The SDFITS files to modify.")
    
    return parser.parse_args()


def main():
    """
    Entry point for script.
    """

    args = parse_args()

    for f in args.files:
        eq2gal.eq2gal(f)


if __name__ == "__main__":

    main()

