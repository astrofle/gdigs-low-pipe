"""
Command line interface for the pipeline calibration step.
"""

import argparse

from pathlib import Path
from gdigs_low_pipe.utils import get_vegas_sdfits_files
from gdigs_low_pipe.pipeline import cal_pipe


def parse_args():
    """
    Argument parser.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="SDFITS file path to calibrate", type=Path)
    parser.add_argument("output_path", help="Where to save the data products", type=Path)
    parser.add_argument("--ifnums", help="IF numbers to process", default=None)
    parser.add_argument("--plnums", help="Polarizations to process", default=None)
    parser.add_argument("--nchan", help="Number of channels", default=None)
    parser.add_argument("--nd_win", help="Size of the median rolling window for the noise diode temperature", default=512, type=int)
    parser.add_argument("--ref_win", help="Size of the rolling window for the off-source", default=128, type=int)
    parser.add_argument("--sig_win", help="Size of the rolling window for the on-source", default=32, type=int)
    parser.add_argument("--ch_edge", help="Fraction of the channel edges to ignore", default=0.1, type=float)
    parser.add_argument("--overwrite", action="store_true", default=False)

    return parser.parse_args()


def main():
    """
    Entry point for the GDIGS-Low calibration.

    Parameters
    ----------
    path : str
        Directory with the data to be calibrated.

    """

    args = parse_args()
    sdfitsfiles = get_vegas_sdfits_files(args.path)

    for sdfitsfile in sdfitsfiles:
        cal_pipe(sdfitsfile, args.output_path, 
                 args.ifnums, args.plnums, 
                 args.nchan, args.ch_edge,
                 args.nd_win, args.ref_win, args.sig_win, 
                 args.overwrite)


if __name__ == "__main__":
    main()
