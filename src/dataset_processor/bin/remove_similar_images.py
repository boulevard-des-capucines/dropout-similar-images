#!/usr/bin/env python

import logging
import argparse

from dataset_processor.cleaner import DatasetCleaner


def main():

    cli = argparse.ArgumentParser(
        usage="The program removes similar images from a dataset and saves "
              "selected images to an output directory."
    )

    cli.add_argument(
        "--dataset-path", required=True,
        help="A path to a directory with dataset to be cleaned."
    )
    cli.add_argument(
        "--image-file-extensions", nargs='+', default=[".png"],
        help="Image file extensions to be processed."
    )
    cli.add_argument(
        "--min-contour-area-diff", type=int, required=True,
        help="Minimum difference between total contour areas in two images. "
             "If difference is smaller than that, the images are supposed "
             "to be similar."

    )
    cli.add_argument(
        "--output-dir-path", required=True,
        help="An directory path to save selected images to."
    )
    cli.add_argument(
        "--min-imsize-percentile", type=int, required=True,
        help="To estimate minimum allowable images size, the app calculates "
             "q-th percentile of all image sizes and multiplies it "
             "by scale (defined by '--min-imsize-scale'). "
             "Should be in the integer range [0, 100]"
    )
    cli.add_argument(
        "--min-imsize-scale", type=float, required=True,
        help="To estimate minimum allowable images size, the app calculates "
             "q-th percentile of all image sizes and multiplies it "
             "by scale defined by this parameter. Should be a float number."
    )
    cli.add_argument(
        "--min-contour-area", type=int, required=True,
        help="Minimum contour area to be considered when sismilarity estimating."
    )
    cli.add_argument(
        "--gaussian-blur-radii", nargs='+', type=int, required=True,
        help="A list of gaussian blur radii to pre-process input images and "
             "make contour estimation more robust. They should be odd integers."
    )
    cli.add_argument(
        "--black-mask", nargs='+', type=int, required=True,
        help="Mask to crop a some interior part of an image and make black "
             "the other (border) part. Should be a list of 4 integers: "
             "[xmin, ymin, xmax, ymax] in percents of the image size."
    )
    cli.add_argument(
        "--save-data-analysis-plots-to", default=None,
        help="A directory path to save dataset statistics to."
    )

    cli.add_argument(
        "--log-level", choices=("info", "warning", "error"), default="info",
        help="A log level to print in a terminal."
    )

    args = cli.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    data_proc = DatasetCleaner(
        args.dataset_path, args.image_file_extensions
    )

    is_successful = data_proc.run(
        args.min_contour_area_diff,
        args.min_imsize_percentile,
        args.min_imsize_scale,
        args.min_contour_area,
        args.gaussian_blur_radii,
        args.black_mask,
        args.output_dir_path,
        args.save_data_analysis_plots_to
    )

    if is_successful:
        logging.info("Success! Exit.")
        return
    else:
        logging.error("Data cleaning was NOT successful. Exit.")
        exit(1)


if __name__ == "__main__":
    main()
