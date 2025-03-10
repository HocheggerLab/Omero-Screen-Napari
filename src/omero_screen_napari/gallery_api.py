import datetime
import logging
import random
from pathlib import Path
from typing import Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from omero.gateway import BlitzGateway
from qtpy.QtWidgets import QMessageBox
from skimage.measure import find_contours, label, regionprops

from omero_screen_napari.gallery_userdata import UserData
from omero_screen_napari.omero_data import OmeroData
from omero_screen_napari.utils import omero_connect, save_fig, scale_image
from omero_screen_napari.welldata_api import well_image_parser
from ast import literal_eval
from collections.abc import Iterable

logger = logging.getLogger("omero-screen-napari")


def show_gallery(omero_data: OmeroData, user_data: UserData, classifier=False):
    try:
        if user_data.reload or omero_data.cropped_images == []:
            cropped_image_parser = CroppedImageParser(omero_data, user_data)
            cropped_image_parser.parse_crops()

        random_image_parser = RandomImageParser(
            omero_data, user_data, classifier
        )
        random_image_parser.parse_random_images()
        gallery_parser = ParseGallery(omero_data, user_data)
        gallery_parser.plot_gallery()
    except Exception as e:  # noqa: BLE001
        logger.exception(e)
        # Show a message box with the error message
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(str(e))
        msg_box.setWindowTitle("Error")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()


class CroppedImageParser:
    """
    Class to crop images and labels based on user input.
    """

    def __init__(self, omero_data: OmeroData, user_data: UserData):
        self._omero_data: OmeroData = omero_data
        self._user_data: UserData = user_data
        self._start_idx: int = 0
        self._end_idx: int = 0
        self._images: Optional[np.ndarray] = None
        self._image_ids: list[int] = []
        self._labels: Optional[np.ndarray] = None
        self._centroids_row: list[int] = []
        self._centroids_col: list[int] = []
        self._ids: list = []
        self._cropped_images: list[np.ndarray] = []
        self._cropped_labels: list[np.ndarray] = []

    def parse_crops(self):
        self._select_data_index()
        self._images, self._image_ids = self._select_images()
        self._labels = self._select_labels()
        self._crop_data()
        self._remove_duplicate_images()
        self._omero_data.cropped_images = self._cropped_images
        self._omero_data.cropped_masks = self._cropped_labels

    def _select_images(self) -> tuple[np.ndarray, list[int]]:
        images = scale_image(
            self._omero_data.images[self._start_idx : self._end_idx, ...],
            self._omero_data.intensities,
        )
        images = self._select_channels(images)
        image_ids = self._omero_data.image_ids[self._start_idx : self._end_idx]
        return images, image_ids

    def _select_labels(self) -> np.ndarray:
        labels = self._omero_data.labels[self._start_idx : self._end_idx, ...]
        if len(labels.shape) == 3:
            return labels
        elif self._user_data.segmentation == "nucleus":
            return labels[..., 0]
        elif self._user_data.segmentation == "cell":
            return labels[..., 1]
        else:
            logger.error(
                f"Invalid segmentation type: {self._user_data.segmentation}"
            )
            raise ValueError(
                f"Invalid segmentation type: {self._user_data.segmentation}"
            )

    def _select_data_index(self):
        """
        Select images for a specific well loaded to omero_data.
        For example, if A1, A2 have been loaded this function ensures that
        only A1 images are used.

        Raises:
            ValueError: Indicates that the selected well has not been loaded to omero_data
        """
        wells = self._omero_data.well_pos_list

        try:
            index = wells.index(self._user_data.well)
        except ValueError as e:
            logger.error(
                f"The selected well {self._user_data.well} has not been loaded from the plate"
            )
            raise ValueError(
                f"The selected well {self._user_data.well} has not been loaded from the plate"
            ) from e
        num_images_per_well = self._omero_data.images.shape[0] // len(wells)
        self._start_idx = index * num_images_per_well
        self._end_idx = self._start_idx + num_images_per_well

    def _select_channels(self, image_array: np.ndarray):
        # transform channel number from str to int
        channel_data = {
            k: int(v) for k, v in self._omero_data.channel_data.items()
        }
        order_indices = [
            channel_data[channel] for channel in self._user_data.channels
        ]
        return image_array[..., order_indices]

    def _select_image_data(self, df: pl.DataFrame, image_id: int):
        logger.debug(f"Selecting data for image id: {image_id}")
        if "timepoint" in df.columns:
            filtered_df = df.filter(
                (df["image_id"] == image_id)
                & (df["timepoint"] == self._user_data.timepoint)
            )
        else:
            filtered_df = df.filter(df["image_id"] == image_id)
        logger.debug(f"The length of the filtered df is {len(filtered_df)}")
        
        cellcycle_filtered_df = self._select_cellcycledata(filtered_df)
        logger.debug(
            f"The length of the cellcycle filtered df is {len(cellcycle_filtered_df)}"
        )
        if cellcycle_filtered_df.shape[0] == 0:
            logger.error(
                f"The selected image id: {image_id} does not have data in the well_ifdata dataframe"
            )
            # raise ValueError(
            #     f"The selected image id {image_id} does not have data in the well_ifdata dataframe"
            # )

        self._centroids_row, self._centroids_col, self._ids = self._select_centroids(
            cellcycle_filtered_df
        )
        self.log_duplcate_centroids()

    def log_duplcate_centroids(self):
        # Step 1: Find the indices where duplicates occur in self._centroids_row
        row_duplicate_indices = [
            i
            for i, val in enumerate(self._centroids_row)
            if self._centroids_row.count(val) > 1
        ]

        # Step 2: Find the indices where duplicates occur in self._centroids_col
        col_duplicate_indices = [
            i
            for i, val in enumerate(self._centroids_col)
            if self._centroids_col.count(val) > 1
        ]

        # Step 3: Find the common indices where both lists have duplicates
        common_duplicate_indices = set(row_duplicate_indices) & set(
            col_duplicate_indices
        )

        # Step 4: Count how many such indices exist
        count_common_duplicates = len(common_duplicate_indices)
        if count_common_duplicates > 0:
            logger.error(
                f"Number of common duplicate indices: {count_common_duplicates}"
            )

    def _select_cellcycledata(self, df):
        cellcycle = self._user_data.cellcycle
        if cellcycle == "All":
            return df
        if cellcycle in df["cell_cycle"].unique():
            return df.filter(df["cell_cycle"] == cellcycle)
        logger.error(f"Invalid cell cycle phase: {cellcycle}")
        raise ValueError(f"Invalid cell cycle phase: {cellcycle}")

    def _select_centroids(self, df) -> tuple[list[int], list[int], list[int]]:
        if self._user_data.segmentation == "nucleus":
            # labels may be average IDs (float64) or tuple of IDs. Convert strings if present.
            ids = df["label"]
            if ids.dtype == pl.datatypes.String:
                ids = ids.apply(literal_eval)
            return df["centroid-0"].to_list(), df["centroid-1"].to_list(), ids.to_list()
        unique_centroids_df = df.unique(
            subset=["centroid-0_x", "centroid-1_x"]
        )
        return unique_centroids_df[
            "centroid-0_x"
        ].to_list(), unique_centroids_df["centroid-1_x"].to_list(), unique_centroids_df["Cyto_ID"].to_list()

    def _crop_data(self):
        for i, image_id in enumerate(self._image_ids):
            self._process_image_for_cropping(i, image_id)

    def _process_image_for_cropping(self, index: int, image_id: int) -> None:
        """
        Checks if image and lable data are available and applies cropping and processing
        via _crop_and_process_image function
        """
        if self._images is None or self._labels is None:
            logger.error("No images or labels to use for cropping galleries")
            raise ValueError(
                "No images or labels to use for cropping galleries"
            )
        logger.debug(
            f"Shape of selected images for cropping  is {self._images.shape}"
        )
        logger.debug(
            f"Shape of selected labels for cropping  is {self._labels.shape}"
        )
        if len(self._images.shape) == 5:
            timepoint = self._user_data.timepoint
            current_data, current_labels = (
                self._images[index, timepoint, ...],
                self._labels[index, timepoint, ...],
            )
        else:
            current_data, current_labels = (
                self._images[index, ...],
                self._labels[index, ...],
            )
        self._select_image_data(self._omero_data.well_ifdata, image_id)

        for row, col, id in zip(self._centroids_row, self._centroids_col, self._ids):
            self._crop_and_process_image(
                current_data, current_labels, row, col, id
            )
        crop_count = len(self._cropped_images)
        logger.info(
            f"{crop_count} cropped images and labels have been generated for image {image_id}"
        )

    def _crop_and_process_image(
        self, image: np.ndarray, labels: np.ndarray, row: int, col: int, id
    ):
        """
        Performs crop cleans mask, checks if mask is present before appending data to cropped_images and cropped_labels lists
        Uses helper functions crop_region, pad_region, erase_masks
        """
        cropped_image, cropped_label = crop_region(
            image, labels, row, col, self._user_data.crop_size
        )

        if cropped_image.shape != (
            self._user_data.crop_size,
            self._user_data.crop_size,
        ):
            cropped_image, cropped_label = pad_region(
                cropped_image, cropped_label, self._user_data.crop_size
            )

        # Support multiple labels in the mask
        for corrected_cropped_label in erase_masks(cropped_label.copy(), id):
            if np.any(
                corrected_cropped_label
            ):  # Check if the label is effectively empty
                self._cropped_images.append(cropped_image)
                self._cropped_labels.append(corrected_cropped_label)

    def _remove_duplicate_images(self):
        """
        Remove duplicate images and their corresponding labels from the dataset.
        """
        unique_images = []
        unique_labels = []
        seen_images = set()
        initial_count = len(self._images)

        for image, unique_label in zip(self._images, self._labels):
            image_tuple = tuple(
                image.flatten()
            )  # Convert image to a hashable type
            if image_tuple not in seen_images:
                seen_images.add(image_tuple)
                unique_images.append(image)
                unique_labels.append(unique_label)

        self._images = unique_images
        self._labels = unique_labels

        # Calculate and log the number of removed images
        final_count = len(self._images)
        removed_count = initial_count - final_count
        logger.info(f"Removed {removed_count} duplicate images.")


# helper functions for cropping images
def crop_region(
    current_data: np.ndarray,
    current_labels: np.ndarray,
    centroid_row: int,
    centroid_col: int,
    crop_size: int,
):
    """
    Crop a region around the centroid of a segmented object.
    """
    # Calculate crop coordinates
    crop_row_start, crop_row_end = calculate_crop_coordinates(
        centroid_row, current_data.shape[-3], crop_size
    )
    crop_col_start, crop_col_end = calculate_crop_coordinates(
        centroid_col, current_data.shape[-2], crop_size
    )

    # Crop the region from the image
    cropped_region = current_data[
        crop_row_start:crop_row_end, crop_col_start:crop_col_end, :
    ]
    # Crop the corresponding region from the segmentation labels
    cropped_label = current_labels[
        crop_row_start:crop_row_end, crop_col_start:crop_col_end
    ]

    return cropped_region, cropped_label


def calculate_crop_coordinates(
    centroid: int, max_length: int, crop_size: int
) -> tuple[int, int]:
    """
    Calculate start and end points for cropping around a centroid.
    """
    start = int(max(0, centroid - crop_size // 2))
    end = int(min(max_length, centroid + crop_size // 2))
    return start, end


def pad_region(
    cropped_region: np.ndarray, cropped_label: np.ndarray, crop_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pad the cropped region and label to the desired crop size.
    """

    pad_row = crop_size - cropped_region.shape[0]
    pad_col = crop_size - cropped_region.shape[1]
    # Pad centrally
    cropped_region = np.pad(
        cropped_region,
        ((pad_row // 2, pad_row - pad_row // 2), (pad_col // 2, pad_col - pad_col // 2), (0, 0)),
        mode="constant",
    )
    cropped_label = np.pad(
        cropped_label,
        ((pad_row // 2, pad_row - pad_row // 2), (pad_col // 2, pad_col - pad_col // 2)),
        mode="constant",
    )
    return cropped_region, cropped_label


def erase_masks(cropped_label: np.ndarray, id, tol = 10) -> np.ndarray:
    """
    Extracts label regions from the cropped_label. Uses the ID. If this is iterable extract each ID in the iteration.
    If a scalar integer type then extract the ID. Otherwise ignore (legacy IDs could be a scalar float type)
    and extract labels within a distance tolerance of the centroid.

    Note that an integer type is detected by checking int(id) == id. This will not detect cases where multiple
    IDs have been averaged to a single integral value. This is a minority case and allows supporting legacy
    data with float IDs representing a single label.
    """
    if np.isscalar(id) and int(id) == id:
        cropped_label[cropped_label != id] = 0
        return [cropped_label]

    labels = []
    if isinstance(id, Iterable):
        for i in id:
            l = cropped_label.copy()
            l[cropped_label != i] = 0
            labels.append(l)
        return labels

    center_row, center_col = np.array(cropped_label.shape) // 2

    unique_labels = np.unique(cropped_label)
    for unique_label in unique_labels:
        if unique_label == 0:  # Skip background
            continue

        binary_mask = cropped_label == unique_label

        if np.sum(binary_mask) == 0:  # Check for empty masks
            continue

        label_props = regionprops(label(binary_mask))

        if len(label_props) == 1:
            cropped_centroid_row, cropped_centroid_col = label_props[
                0
            ].centroid

            # Using a small tolerance value for comparing centroids
            if (
                abs(cropped_centroid_row - center_row) <= tol
                and abs(cropped_centroid_col - center_col) <= tol
            ):
                l = cropped_label.copy()
                l[cropped_label != unique_label] = 0
                labels.append(l)

    return labels


# --------------------------Select random images for gallery--------------------


class RandomImageParser:
    def __init__(
        self, omero_data: OmeroData, user_data: UserData, classifier: bool
    ):
        self._omero_data: OmeroData = omero_data
        self._user_data: UserData = user_data
        self._classifier: bool = classifier
        self._chosen_indices: list[int] = []  # indices of images to be used
        self._random_images: list[np.ndarray] = []
        self._random_labels: list[np.ndarray] = []

    def parse_random_images(self):
        self._parse_random_index()
        self._parse_random_images()
        if self._classifier:
            self._omero_data.selected_crops = self._random_images
        # self._check_identical_arrays()
        self._omero_data.cropped_images = self._remove_chosen_crops(
            self._omero_data.cropped_images
        )
        self._omero_data.cropped_masks = self._remove_chosen_crops(
            self._omero_data.cropped_masks
        )
        self._apply_mask_to_images()
        self._random_images = [
            fill_missing_channels(img, self._user_data.channels)
            for img in self._random_images
        ]
        if self._user_data.contour:
            self._random_images = [
                draw_contours(img, label)
                for img, label in zip(self._random_images, self._random_labels)
            ]
        self._omero_data.selected_images = self._random_images
        self._omero_data.selected_labels = self._random_labels

    def _parse_random_index(self):
        """
        Select random index to be used to choose images the gallery from the croped images and labels.
        """
        if self._user_data.columns == 0 and self._user_data.rows == 0:
            self._chosen_indices = list(
                range(len(self._omero_data.cropped_images))
            )
        else:
            sample_size = min(
                self._user_data.columns * self._user_data.rows,
                len(self._omero_data.cropped_images),
            )
            self._chosen_indices = random.sample(
                range(len(self._omero_data.cropped_images)), sample_size
            )

    def _parse_random_images(self):
        """
        Use the random_indeces to select the images and labels to be used in the gallery.
        """
        self._random_images = [
            self._omero_data.cropped_images[i] for i in self._chosen_indices
        ]
        self._random_labels = [
            self._omero_data.cropped_masks[i] for i in self._chosen_indices
        ]

    def _check_identical_arrays(self):
        """
        Saftey check to ensure that gallery images have been chosen and that no identical arrays
        are present in the chosen cells.
        """
        if not self._random_images:
            logger.error("Slelection of crops for gallery has failed")
            raise ValueError("Slelection of crops for gallery has failed")
        # Check each array with every other array in the list
        n = len(self._random_images)
        for i in range(n):
            for j in range(
                i + 1, n
            ):  # Start from i+1 to avoid comparing the same array
                if np.array_equal(
                    self._random_images[i], self._random_images[j]
                ):
                    raise ValueError(
                        "There are identical arrays in the chosen cells. Please try again."
                    )
        return  # No identical arrays found

    def _remove_chosen_crops(self, array_list):
        """
        Removes the chosen images and labels from the cropped_images and cropped_labels lists.
        """
        return [
            item
            for index, item in enumerate(array_list)
            if index not in self._chosen_indices
        ]

    def _apply_mask_to_images(self):
        """
        Nullify pixels in color images that don't overlap with the corresponding masks.
        Images are expected to be in the shape of (H, W, 3) and masks in the shape of (H, W).
        """
        masked_images = []
        for image, mask in zip(self._random_images, self._random_labels):  # type: ignore
            # Ensure the mask is expanded to match the image's 3 channels
            expanded_mask = (
                np.repeat(mask[:, :, np.newaxis], image.shape[2], axis=2) > 0
            )
            # Apply the expanded mask to the image
            masked_image = np.where(expanded_mask, image, 0)
            masked_images.append(masked_image)

        self._random_images = masked_images


# Helper functions for RandomImageParser


def fill_missing_channels(img, channels):
    # Initialize an empty list to hold the channel arrays
    result_img = []
    empty_image = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)

    # Count the number of non-empty channels
    num_non_empty = len(channels)

    if num_non_empty == 1:
        # If there's only one non-empty channel, ensure it's placed at the front
        result_img = [img[..., 0], empty_image, empty_image]
    elif num_non_empty == 2:
        result_img = [img[..., 0], img[..., 1], empty_image]
    else:
        result_img = [img[..., 2], img[..., 1], img[..., 0]]
    return np.stack(result_img, axis=-1)


def draw_contours(img, label):
    channel_num = img.shape[-1]
    contours = find_contours(label, 0.5)
    for contour in contours:
        for coords in contour:
            x, y = coords.astype(int)
            img[x, y] = [1] * channel_num  # White color
    return img


# --------------------------------Gallery Constructor--------------------------------


class ParseGallery:
    def __init__(
        self, omero_data: OmeroData, user_data: UserData, show_gallery=True
    ):
        self._omero_data: OmeroData = omero_data
        self._user_data: UserData = user_data
        self._gallery_image: np.ndarray = np.empty((0,))
        self._metadata: dict[str, str] = {}
        self._show_gallery = show_gallery

    def plot_gallery(self):
        padding_height, padding_width = self._calculate_dynamic_padding()
        self._gallery_image = self._create_gallery_image(
            padding_height, padding_width
        )
        self._parse_metadata()
        return self._build_gallery()

    def _build_gallery(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        if len(self._user_data.channels) == 1:
            ax.imshow(self._gallery_image[..., 0], cmap="gray_r")
        else:
            ax.imshow(self._gallery_image)
        metadata_str = ", ".join(
            [f"{key}: {value}" for key, value in self._metadata.items()]
        )
        channel_list = [
            channel for channel in self._user_data.channels if channel != ""
        ]
        ax.set_title(
            f"well: {self._user_data.well}\n{metadata_str}\nchannels: {', '.join(channel_list)}, cellcycle phase: {self._user_data.cellcycle}, timepoint: {self._user_data.timepoint}",
            fontsize=12,
            fontweight="bold",
        )
        plt.axis("off")
        # Add scale bar
        self._add_scale_bar(ax)
        logger.info("plotting gallery")
        if self._show_gallery:
            plt.show(block=False)
        return fig

    def _create_gallery_image(
        self, padding_height: int, padding_width: int
    ) -> np.ndarray:
        img_height, img_width, img_channels = self._omero_data.selected_images[
            0
        ].shape
        n_row, n_col = self._user_data.rows, self._user_data.columns
        # Adjust gallery dimensions to include border padding
        gallery_height = n_row * img_height + (n_row + 1) * padding_height
        gallery_width = n_col * img_width + (n_col + 1) * padding_width

        # Create an array filled with 1.0 (white background)
        gallery_image = np.full(
            (gallery_height, gallery_width, img_channels),
            fill_value=1.0,
            dtype=np.float64,
        )

        for row in range(n_row):
            for col in range(n_col):
                idx = row * n_col + col
                if idx >= len(self._omero_data.selected_images):
                    break
                # Adjust start positions to account for the border padding
                start_row = (
                    row * (img_height + padding_height) + padding_height
                )
                end_row = start_row + img_height
                start_col = col * (img_width + padding_width) + padding_width
                end_col = start_col + img_width
                gallery_image[start_row:end_row, start_col:end_col, :] = (
                    self._omero_data.selected_images[idx]
                )

        return gallery_image

    def _calculate_dynamic_padding(self) -> tuple[int, int]:
        img_height, img_width = self._omero_data.selected_images[0].shape[:2]

        padding_height = int(img_height * 0.02)
        padding_width = int(img_width * 0.02)
        return padding_height, padding_width

    def _parse_metadata(self):
        index_number = self._omero_data.well_pos_list.index(
            self._user_data.well
        )
        self._metadata = self._omero_data.well_metadata_list[index_number]

    def _add_scale_bar(self, ax):
        gallery_height, gallery_width, _ = self._gallery_image.shape
        physical_scale_bar_length = (
            10 if self._user_data.crop_size <= 30 else 25
        )  # in microns
        scale_bar_length_in_pixels = int(
            physical_scale_bar_length / self._omero_data.pixel_size[0]
        )
        scale_bar_length_in_pixels = int(
            physical_scale_bar_length / self._omero_data.pixel_size[0]
        )
        bar_height = 1
        start_x = (
            gallery_width - scale_bar_length_in_pixels - gallery_width * 0.04
        )
        start_y = gallery_height - bar_height - gallery_width * 0.01
        color = "black" if len(self._user_data.channels) == 1 else "white"
        scale_bar = patches.Rectangle(
            (start_x, start_y),
            scale_bar_length_in_pixels,
            bar_height,
            linewidth=1,
            edgecolor=color,
            facecolor=color,
        )
        ax.add_patch(scale_bar)
        label_x = start_x + scale_bar_length_in_pixels / 2
        label_y = start_y - 0.5
        ax.text(
            label_x,
            label_y,
            f"{physical_scale_bar_length} µm",
            color=color,
            ha="center",
            va="bottom",
            fontsize=12,
        )


# -----------------------------Well Galleries-----------------------------------


@omero_connect
def run_gallery_parser(
    omero_data: OmeroData,
    user_data: UserData,
    well_input: list[str],
    galleries: int,
    conn: Optional[BlitzGateway] = None,
):
    if conn:
        try:
            gallery_path = _manage_path(omero_data)
            well_list = _get_wells(omero_data, well_input, conn)
            for well in well_list:
                try:
                    _save_gallery(
                        omero_data,
                        user_data,
                        well,
                        galleries,
                        gallery_path,
                        conn,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(e)
        except Exception as e:
            logger.error(e)
            raise e


def _manage_path(omero_data: OmeroData):
    DEFAULT_PATH = Path.home() / "Omero-Screen-Galleries"
    DEFAULT_PATH.mkdir(exist_ok=True)
    experiment = f"{omero_data.plate_id}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    experiment_path = DEFAULT_PATH / experiment
    experiment_path.mkdir(exist_ok=True)
    print(f"Creating directory at {experiment_path}")
    return experiment_path


def _get_wells(
    omero_data: OmeroData, well_input: list[str], conn: BlitzGateway
):
    plate = conn.getObject("Plate", omero_data.plate_id)
    well_list = [well.getWellPos() for well in plate.listChildren()]  # type: ignore
    if well_input == ["All"]:
        return well_list
    else:
        return [well for well in well_input if well in well_list]


def _save_gallery(
    omero_data: OmeroData,
    userdata: UserData,
    wellpos: str,
    galleries: int,
    path,
    conn: BlitzGateway,
):
    _reset_data(omero_data, userdata, wellpos, conn)
    well_image_parser(omero_data, wellpos, conn)
    userdata.reload = False
    cropped_image_parser = CroppedImageParser(omero_data, userdata)
    cropped_image_parser.parse_crops()
    for i in range(galleries):
        random_image_parser = RandomImageParser(
            omero_data, userdata, classifier=False
        )
        random_image_parser.parse_random_images()
        gallery_parser = ParseGallery(omero_data, userdata, show_gallery=False)
        fig = gallery_parser.plot_gallery()
        save_fig(
            fig,
            path,
            f"{omero_data.plate_id}_{wellpos}_gallery_{i}",
            fig_extension="pdf",
        )


def _reset_data(
    omero_data: OmeroData, userdata: UserData, wellpos: str, conn: BlitzGateway
):
    omero_data.reset_well_and_image_data()
    # omero_data.cropped_images == [np.empty((0,))]
    # omero_data.cropped_labels == []
    omero_data.well_pos_list = [wellpos]
    userdata.well = wellpos
    if plate := conn.getObject("Plate", omero_data.plate_id):
        omero_data.plate = plate
    else:
        raise ValueError(f"Error connection to plate {omero_data.plate_id}")
    # Check if both project and dataset can be retrieved, and then assign dataset to omero_data.screen_dataset
    if (project := conn.getObject("Project", omero_data.project_id)) and (
        dataset := conn.getObject(
            "Dataset",
            attributes={"name": str(omero_data.plate_id)},
            opts={"project": project.getId()},
        )
    ):
        omero_data.screen_dataset = dataset
    else:
        raise ValueError(
            f"Error connection to project {omero_data.project_id}"
        )
