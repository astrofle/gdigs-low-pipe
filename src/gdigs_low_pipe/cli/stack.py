"""
"""

import argparse

from gdigs_low_pipe.pipeline import stack


def parse_args():
    """
    Argument parser.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=str, nargs="+", 
                        help="The SDFITS files to stack.")
    parser.add_argument("-ll", type=str, required=True, dest="line_list",
                        help="File with the lines to be used for the stack.")
    parser.add_argument("-o", type=str, required=True, dest="output",
                        help="Output filename. Full path without the .fits")
    parser.add_argument("-l", "--chan_left", type=int, default=400,
                        help="Line free channels to estimate the rms at the start of the spectra.")
    parser.add_argument("-r", "--chan_right", type=int, default=400,
                        help="Line free channels to estimate the rms at the end of the spectra.")
    parser.add_argument("-p", "--poly_order", type=int, default=1,
                        help="Polynomial order to remove continuum.")

    return parser.parse_args()


def main():
    """
    Entry point for stacking script.
    """

    args = parse_args()

    stack.stack(args.files, args.line_list, args.output,
                poly_order=args.poly_order, 
                left=args.chan_left, right=args.chan_right)


if __name__ == "__main__":

    main()
