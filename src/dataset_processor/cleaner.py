import os
import shutil
import logging

from datetime import datetime
import collections as cols
import json

import typing

import pandas as pd
import numpy as np
import cv2

from .image_comparators import \
    preprocess_image_change_detection, \
    compare_frames_change_detection


class DatasetCleaner:

    def __init__(self, dir_path: str, extensions: list) -> None:

        self._dir = dir_path
        self._extensions = extensions

        self._fnames = self._get_time_sorted_and_grouped_by_camera_fnames()

    def run(self,
            min_contour_area_diff: int,
            min_imsize_percentile: int,
            min_imsize_scale: float,
            min_contour_area: int,
            gaussian_blur_radiuses: list,
            black_mask: list,
            save_images_to: str,
            save_data_analysis_plots_to: str):

        is_processing_successful = False

        blur_radiuses = self._check_and_correct(gaussian_blur_radiuses)
        if not blur_radiuses:
            logging.info("No correct gaussian blur radiuses.")
            return is_processing_successful

        logging.info(f"Reading the dataset (direcotry: '{self._dir}')...")
        df = self._get_data(
            min_imsize_percentile, min_imsize_scale,
            min_contour_area, blur_radiuses, black_mask
        )
        if df.empty:
            logging.info("No data is read.")
            return is_processing_successful

        params = {
            "min_contour_area_diff": min_contour_area_diff,
            "min_imsize_percentile": min_imsize_percentile,
            "min_imsize_scale": min_imsize_scale,
            "min_contour_area": min_contour_area,
            "gaussian_blur_radiuses": gaussian_blur_radiuses,
            "black_mask": black_mask
        }
        self._save_analytics(df, params, save_data_analysis_plots_to)

        df = df[df["score"] > min_contour_area_diff]

        if self._save_images(df, save_images_to):
            logging.info(
                f"The images have been saved to the directory: "
                f"'{save_images_to}'"
            )
            is_processing_successful = True
        else:
            logging.error(
                f"The images have NOT been saved to the directory: "
                f"'{save_images_to}'"
            )
            is_processing_successful = False

        return is_processing_successful

    def _get_data(self,
                  min_imsize_percentile: int,
                  min_imsize_scale: float,
                  min_contour_area: int,
                  gaussian_blur_radiuses: list,
                  black_mask: list) -> pd.DataFrame:

        min_imsizes = self._get_min_imsize(min_imsize_percentile)

        data = {"camera_id": [], "timestamp": [], "score": [], "fname": []}
        prev_img = None

        for cam_id in self._fnames.keys():

            min_size = min_imsizes[cam_id]

            for timestamp, fname in self._fnames[cam_id].items():

                current_img = cv2.imread(os.path.join(self._dir, fname), cv2.IMREAD_ANYCOLOR)
                if current_img is None:
                    logging.warning(
                        f"The file '{fname}' is NOT read. Continue..."
                    )
                    continue

                if (current_img.shape[:2] < min_imsize_scale * min_size[:2]).any():
                    logging.warning(
                        f"The image '{fname}' is too small. "
                        f"Image size: {current_img.shape[:2]}. "
                        f"Minimum image size for the camera "
                        f"'{cam_id}': {(min_size.astype(tuple))}."
                        f"Try to configure the 'min_imsize_percentile' parameter. "
                        f"Continue..."
                    )
                    continue

                if prev_img is None:
                    prev_img = np.zeros_like(current_img)

                is_calculated, score = self._get_score(
                    prev_img, current_img,
                    min_contour_area, gaussian_blur_radiuses, black_mask
                )
                if not is_calculated:
                    logging.warning(
                        f"Can't calculate the score for '{fname}'. "
                        f"Image size: {current_img.shape}. "
                        f"Continue..."
                    )
                    continue

                data["camera_id"].append(cam_id)
                data["timestamp"].append(timestamp)
                data["score"].append(score)
                data["fname"].append(fname)

                prev_img = current_img

        return pd.DataFrame(data)

    def _save_analytics(self,
                        df: pd.DataFrame,
                        algo_params: dict,
                        dir_path: typing.Optional[str]=None):

        is_dir_created = self._create_directory(dir_path)
        if not is_dir_created:
            return

        params = {k: v for k, v in algo_params.items()}
        with open(os.path.join(dir_path, "algorithm-params.json"), "w") as f:
            json.dump(params, f, indent=4)

        ax = df["score"].hist(bins=100, legend=True)
        ax.get_figure().savefig(os.path.join(dir_path, "score-hist.png"))
        ax.clear()

        with open(os.path.join(dir_path, "df-stat-summary.txt"), "w") as f:
            f.write(str(df.describe()))

    def _get_score(self,
                   prev_img: np.array,
                   next_img: np.array,
                   min_contour_area: int,
                   gaussian_blur_radiuses: list,
                   black_mask: list) -> (bool, typing.Optional[int]):

        success, score = False, None

        if prev_img.shape != next_img.shape:
            prev_img_scaled = cv2.resize(
                prev_img, dsize=next_img.shape[:2][::-1],
                interpolation=cv2.INTER_NEAREST
            )
        else:
            prev_img_scaled = prev_img

        prepared_prev = preprocess_image_change_detection(
            prev_img_scaled, gaussian_blur_radiuses, black_mask
        )
        prepared_next = preprocess_image_change_detection(
            next_img, gaussian_blur_radiuses, black_mask
        )

        if prepared_prev.shape == prepared_next.shape:

            score, _, _ = compare_frames_change_detection(
                prepared_prev, prepared_next, min_contour_area
            )
            success = True

        return success, score

    def _get_min_imsize(self, percentile: int) -> dict:

        min_sizes = {}

        for cam_id in self._fnames.keys():

            shapes = []

            for timestamp, fname in self._fnames[cam_id].items():

                img = cv2.imread(os.path.join(self._dir, fname), cv2.IMREAD_ANYCOLOR)
                if img is None:
                    continue

                shapes.append(img.shape)

            min_sizes[cam_id] = np.percentile(shapes, percentile, axis=0).astype(int)

        return min_sizes

    def _check_and_correct(self, gaussian_blur_radiuses) -> list:

        if any([r <= 0 for r in gaussian_blur_radiuses]):
            logging.warning(
                f"The gaussian blur radius should be positive. "
                f"Some of the passed values are negative or zero. "
                f"All negative or zero values won't be used. Continue..."
            )

        positive_blur_radiuses = [r for r in gaussian_blur_radiuses if r > 0]

        for i, r in enumerate(positive_blur_radiuses):

            if r % 2 == 0:
                logging.warning(
                    f"The gaussian blur radius should be odd. "
                    f"The value '{r}' will be replaced with '{r + 1}'. "
                    f"Continue..."
                )
                positive_blur_radiuses[i] += 1

        return positive_blur_radiuses

    def _save_images(self, df: pd.DataFrame, dir_path: str):

        if not self._create_directory(dir_path):
            return False

        try:
            for idx, fname in df["fname"].items():
                shutil.copy2(os.path.join(self._dir, fname), dir_path)
        except Exception as e:
            logging.error(
                f"Can't copy selected images to dir '{dir_path}'. "
                f"Error: '{e}'."
            )
            return False

        return True

    def _create_directory(self, path: str) -> bool:

        is_dir_created = False
        if not path:
            return is_dir_created

        os.makedirs(path, exist_ok=True)

        if os.path.exists(path):
            is_dir_created = True
        else:
            logging.warning(f"The directory '{path}' is NOT created.")

        return is_dir_created

    def _get_time_sorted_and_grouped_by_camera_fnames(self) -> dict[dict]:

        fnames = cols.defaultdict(dict)  # {camera_id: {timestamp: fname}}

        for name in os.listdir(self._dir):

            if not os.path.isfile(os.path.join(self._dir, name)):
                continue

            ok, camera_id, tstamp = self._parse_fname(name)
            if not ok:
                continue

            fnames[camera_id][tstamp] = name

        for cam_id in fnames.keys():
            fnames[cam_id] = dict(sorted(fnames[cam_id].items()))

        return fnames

    def _parse_fname(self, basename: str) -> (bool, str, int):

        name = os.path.splitext(basename)[0]
        is_parsed = False

        if '-' in basename:

            camera_id, timestamp = name.split('-')
            timestamp = float(timestamp) * 1e-3  # to seconds
            is_parsed = True

        elif '_' in basename:

            camera_id, *date_and_time = name.replace('__', '_').split('_')

            date_and_time_iso = "{}-{}-{} {}:{}:{}".format(*date_and_time)
            timestamp = datetime.timestamp(
                datetime.fromisoformat(date_and_time_iso)
            )
            is_parsed = True

        else:

            logging.warning(
                f"The file '{basename}' has wrong file name format. "
                "The allowable formats are: "
                "camera_id-timestamp.png or "
                "camera_id_YYYY_MM_DD__HH_mm_SS.png. "
                "Continue..."
            )
            camera_id, timestamp = None, None

        return is_parsed, camera_id.lower(), timestamp
