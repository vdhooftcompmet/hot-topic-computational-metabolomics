#!/usr/bin/env python3
"""
Author:         David Meijer
Description:    Create a clamshell plot of your counts data.
Usage:          Run './clamshell_plot.py -h' for help.
"""
import typing as ty 
import argparse
import math

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


Color = ty.Tuple[float, float, float]


GREY = (220 / 256, 220 / 256, 220 / 256)
BLUE = (0 / 256, 0 / 256, 128 / 256)


def cli () -> argparse.Namespace:
    """
    Create command line interface for script.
    
    Returns
    -------
    argparse.Namespace: parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create a clamshell plot of your counts data."
    )

    # Required parameters.
    parser.add_argument(
        "-i", "--input", 
        required=True,
        help="Input file containing a list of items to count and visualize."
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output file to save plot to."
    )
    parser.add_argument(
        "-nb", "--nbins",
        type=int,
        help="Number of bins to use."
    )

    # Optional parameters.
    parser.add_argument(
        "-t", "--transparent", 
        action="store_true", 
        help="Make plot background transparent."
    )
    parser.add_argument(
        "-ct", "--counts_threshold",
        type=int,
        default=0,
        help="Low counts threshold."
    )
    parser.add_argument(
        "-l", "--log",
        default=None,
        help="Path to log file. No log is saved if no path is supplied."
    )

    return parser.parse_args()


def parse_input(path: str) -> ty.Tuple[ty.List[str], ty.List[int]]:
    """
    Parse input file.

    Parameters
    ----------
    path (str): Path to input file.

    Returns
    -------
    ty.List[str]: List of labels.
    ty.List[int]: List of counts.
    """
    with open(path, "r") as handle:
        handle.readline()  # Skip header.
        lines = [line.strip().split("\t") for line in handle]
    items = list(map(lambda x: (x[0], int(x[1])), lines))
    labels, counts = zip(*items)
    return labels, counts


def make_color_map(
    source_color: Color,
    target_color: Color,
    nbins: int = 256
) -> LinearSegmentedColormap:
    """
    Create a color map.

    Parameters
    ----------
    source_color (Color): Source color.
    target_color (Color): Target color.
    nbins (int): Number of bins.

    Returns
    -------
    LinearSegmentedColormap: Color map.

    Note: red-green-blue items of color are floats in range [0, 1].
    """
    return LinearSegmentedColormap.from_list(
        name="custom",
        colors=[source_color, target_color],
        N=nbins
    )


def clamshell_plot(
    counts: ty.List[int], 
    labels: ty.List[str],
    nbins: int,
    output: str,
    transparent: bool = False,
    color_range: ty.Tuple[Color, Color] = (GREY, BLUE),
    counts_threshold: int = 0,
    log: str = None
) -> None:
    """
    Draw a clamshell plot.

    Arguments
    ---------
    counts (ty.List[int]): List of counts.
    labels (ty.List[str]): List of labels.
    nbins (int): Number of bins.
    output (str): Path to output file.
    transparent (bool): Whether to make the plot background transparent.
    color_range (ty.Tuple[Color, Color]): Color range.
    counts_threshold (int): Low counts threshold.
    log (str): Path to log file.

    Note: counts cannot be <0.
    """
    if log: log_handle = open(log, "w")

    fig, axs = plt.subplots()
    axs.set_aspect(1)
    cmap = make_color_map(color_range[0], color_range[1])

    # Filter out classes with low counts and display them as smallest class 
    # (circle with radius near-0).
    if counts_threshold > 0:
        all_items = list(zip(labels, counts))
        filtered_items = filter(lambda x: x[1] < counts_threshold, all_items)
        other_items = filter(lambda x: x[1] >= counts_threshold, all_items)
        filtered_items, other_items = list(filtered_items), list(other_items)
    else:
        filtered_items, other_items = [], list(zip(labels, counts))

    # Calculate bin thresholds in counts.
    thresholds = [
        ((max(counts)) / nbins) * bin
        for bin in range(1, nbins + 1)
    ][::-1] + [0]
    if log: log_handle.write(f"Thresholds: {thresholds}\n")

    for i in range(len(thresholds) - 1):
        max_bin = thresholds[i]
        min_bin = thresholds[i + 1]

        # Select items from counts based on previous threshold and new max
        # threshold.
        items = filter(
            lambda x: (x[1] > min_bin) & (x[1] <= max_bin),
            other_items
        )

        # Convert counts to surface areas with same ratio as to the max count.
        max_surface = math.pi  # Circle of max count has radius 1. 
        count_ratio = lambda v: v / max(counts)
        radius_from_surface = lambda a: math.sqrt(a / math.pi)

        items = map(lambda x: (x[0], x[1], count_ratio(x[1])), items)
        items = map(lambda x: (x[0], x[1], x[2] * max_surface), items)
        items = map(lambda x: (x[0], x[1], radius_from_surface(x[2])), items)
        items = list(items)
        if log: log_handle.write(f"Items ({len(items)}): {items}\n")

        # Draw clamshell plot section.
        if len(items) > 0:
            radius = max(list(map(lambda v: v[2], items)))

            # Draw circle.
            circle = plt.Circle(
                (0.0, radius), 
                radius, 
                color=cmap(radius),
                fill=True                
            )
            axs.add_artist(circle)

            # Draw edges circles separately since edgecolor is overwritten by
            # color if fill is True.
            circle = plt.Circle(
                (0.0, radius), 
                radius + 0.002, 
                fill=False,
                linewidth=0.6,
                color="k"
            )
            axs.add_artist(circle)

            # Draw annotation.
            axs.hlines(
                y=radius * 2 + 0.002, 
                xmin=0, 
                xmax=1.5, 
                linewidth=0.6, 
                color="k"
            )


    # Draw low counts point at in clamshell plot for filtered out classes. 
    if len(filtered_items) > 0:
        if log: 
            log_handle.write(
                f"Filtered out {len(filtered_items)} classes:\n"
                f"{filtered_items}\n"
            )

        radius = 0.025

        # Draw circle.
        draw_circle = plt.Circle(
            (0.0, radius), 
            radius, 
            fill=True, 
            linewidth=0.5, 
            color="k"
        )
        axs.add_artist(draw_circle)

        # Draw annotation.
        axs.hlines(
            y=radius * 2, 
            xmin=0, 
            xmax=1.5, 
            linewidth=0.5, 
            color="k"
        )

    plt.axis("off")
    plt.xlim([-1.01, 2.1])
    plt.ylim([-0.01, 2.1])
    plt.savefig(output, bbox_inches="tight", transparent=transparent, dpi=900)

    if log: log_handle.close()


def main() -> None:
    """
    Driver code.
    """
    args = cli()
    labels, counts = parse_input(args.input)
    clamshell_plot(
        counts=counts, 
        labels=labels, 
        nbins=args.nbins,
        output=args.output,
        transparent=args.transparent,
        counts_threshold=args.counts_threshold,
        log=args.log,
    )


if __name__ == "__main__":
    main()
    
