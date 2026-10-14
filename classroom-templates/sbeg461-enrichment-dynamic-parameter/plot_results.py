"""Plot a CSV file written by the C++ track (sbeg::write_csv): every column against the first one.

    python plot_results.py results.csv                 # all columns against column 1 (e.g. t)
    python plot_results.py results.csv q_1 qf_1        # only the named columns
    python plot_results.py results.csv --out fig.png   # save instead of showing

Given helper for the C++ track (the Python track plots directly with matplotlib).
"""
import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("columns", nargs="*", help="columns to plot (default: all but the first)")
    ap.add_argument("--out", help="save the figure to this file instead of showing it")
    args = ap.parse_args()
    with open(args.csv, newline="") as fh:
        header = [h.strip() for h in next(csv.reader(fh))]
    data = np.loadtxt(args.csv, delimiter=",", skiprows=1, ndmin=2)
    names = args.columns or header[1:]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in names:
        ax.plot(data[:, 0], data[:, header.index(name)], label=name)
    ax.set_xlabel(header[0])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()
