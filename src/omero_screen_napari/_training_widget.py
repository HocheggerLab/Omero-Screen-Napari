import json
import logging
from dataclasses import asdict
from pathlib import Path

import napari
import numpy as np
from magicgui import magicgui
from magicgui.widgets import Container, RadioButtons
from qtpy.QtWidgets import QMessageBox

from omero_screen_napari.gallery_api import (
    CroppedImageParser,
    RandomImageParser,
    draw_contours,
    fill_missing_channels,
)
from omero_screen_napari.gallery_userdata import UserData
from omero_screen_napari.gallery_userdata_singleton import userdata
from omero_screen_napari.omero_data import OmeroData
from omero_screen_napari.omero_data_singleton import omero_data
from omero_screen_napari.utils import omero_connect

from omero.gateway import BlitzGateway, MapAnnotationWrapper, FileAnnotationWrapper
from omero.rtypes import rstring
from omero.model import ImageI, FileAnnotationI
from omero.sys import ParametersI
import numpy as np
from PIL import Image
from pathlib import Path
import os


logger = logging.getLogger("omero-screen-napari")
logging.basicConfig(level=logging.DEBUG)

#TODO: timepoint option needs to be integrated into the training widget

def training_widget(
    class_name: str | None = None,
    userdata: UserData | None = userdata,
) -> Container:
    widget = TrainingWidget(class_name, userdata, omero_data)
    return widget.container


class ImageNavigator:
    def __init__(self, class_options: list | None, omero_data: OmeroData):
        self.omero_data = omero_data
        self.current_index = 0
        self.first_load = True  # Flag to check if it is the first image load
        self.saved_contrast_limits = None  # Variable to save user settings
        self.class_options = (
            [
                "unassigned",
            ]
            if class_options is None
            else class_options
        )
        self.class_choice = RadioButtons(
            label="Select class:",
            choices=self.class_options,
            value="unassigned",
        )
        self.class_choice.changed.connect(self.assign_class)

    def next_image(self):
        if self.omero_data.selected_images:
            self.current_index = (self.current_index + 1) % len(
                self.omero_data.selected_images
            )
            self.update_image()

    def previous_image(self):
        if self.omero_data.selected_images:
            self.current_index = (self.current_index - 1) % len(
                self.omero_data.selected_images
            )
            self.update_image()

    def update_image(self):
        viewer = napari.current_viewer()
        current_choices = self.class_choice.choices
        self.class_choice.changed.disconnect(self.assign_class)

        self._save_current_settings(viewer)
        self._clear_existing_layers(viewer)

        if self.omero_data.selected_images:
            image = self.omero_data.selected_images[self.current_index]
            logger.info(f"Selected image shape: {image.shape}")

            self._add_image_to_viewer(viewer, image)
            self._verify_layer_added(viewer)
            self._apply_saved_contrast_limits(viewer)

            self.first_load = False  # Update the flag after the first load
        else:
            logger.warning("No selected images to load.")

        self._restore_class_choices(current_choices)
        self._refresh_viewer(viewer)

    def _save_current_settings(self, viewer):
        if not self.first_load and viewer.layers:
            self.saved_contrast_limits = viewer.layers[0].contrast_limits
            logger.info(f"Saving contrast limits: {self.saved_contrast_limits}")

    def _clear_existing_layers(self, viewer):
        viewer.layers.clear()
        logger.info("Viewer layers cleared.")

    def _add_image_to_viewer(self, viewer, image):
        try:
            if self.omero_data.selected_crops[0].shape[-1] == 1:
                self._add_grayscale_image(viewer, image)
            else:
                self._add_rgb_image(viewer, image)
        except Exception as e:
            logger.error(f"Error adding image to viewer: {e}")

    def _add_grayscale_image(self, viewer, image):
        grayscale_image = np.mean(image, axis=-1)
        logger.debug(f"Grayscale image shape: {grayscale_image.shape}")
        inverted_image = 1.0 - grayscale_image  # Invert the image
        viewer.add_image(inverted_image, name=f"Cropped Image {self.current_index}", colormap="gray")
        logger.debug("Inverted grayscale image added to viewer.")

    def _add_rgb_image(self, viewer, image):
        viewer.add_image(image, name=f"Cropped Image {self.current_index}")
        logger.debug("RGB image added to viewer.")

    def _verify_layer_added(self, viewer):
        if not viewer.layers:
            logger.error("No layers present in the viewer after adding the image.")
            return

    def _apply_saved_contrast_limits(self, viewer):
        if not self.first_load and self.saved_contrast_limits:
            min_intensity, max_intensity = viewer.layers[0].data.min(), viewer.layers[0].data.max()
            if self.saved_contrast_limits[0] >= min_intensity and self.saved_contrast_limits[1] <= max_intensity:
                logger.debug(f"Applying contrast limits: {self.saved_contrast_limits}")
                viewer.layers[0].contrast_limits = self.saved_contrast_limits
            else:
                logger.warning(f"Contrast limits {self.saved_contrast_limits} are out of range for the new image intensity values.")
        else:
            logger.debug("No contrast limits to apply or first image load.")

    def _restore_class_choices(self, current_choices):
        self.class_choice.choices = current_choices
        self.class_choice.changed.connect(self.assign_class)
        self.update_class_choice()
        logger.debug("Class choices restored and signal reconnected.")

    def _refresh_viewer(self, viewer):
        viewer.update_console({"layers": viewer.layers})
        logger.debug("Viewer updated successfully.")

    def assign_class(self, class_name: str):
        if self.omero_data.selected_classes:
            self.omero_data.selected_classes[self.current_index] = class_name

    def update_class_choice(self):
        if self.omero_data.selected_classes:
            current_class = self.omero_data.selected_classes[self.current_index]
        else:
            current_class = "unassigned"
        self.class_choice.value = (
            current_class
            if current_class in self.class_choice.choices
            else "unassigned"
        )


class TrainingWidget:
    def __init__(
        self,
        class_name: str | None,
        user_data: UserData | None,
        omero_data: OmeroData,
        class_options: list[str] | None = None,
    ):
        self.image_navigator = ImageNavigator(class_options, omero_data)
        self.user_data = user_data
        self.omero_data = omero_data
        self.class_name = class_name

        self.user_data_dict = {}
        self.class_options = {}
        self.training_data_saver = None
        self.setup_key_bindings(napari.current_viewer())

        self.load_image = magicgui(
            call_button="Load Images",
            text_input={"label": "filename", "value": self.class_name},
        )(self.load_image)
        self.next_image = magicgui(call_button="Next Image")(self.next_image)
        self.previous_image = magicgui(call_button="Previous Image")(
            self.previous_image
        )
        self.save_training_data = magicgui(
            call_button="Save training data",
        )(self.save_training_data)

        self.container = self.create_container()

    def load_image(self, text_input: str):
        try:
            classifier_name = text_input.strip()
            if not classifier_name:
                raise ValueError("Classifier name cannot be empty.")
            self.class_name = classifier_name
            file_name, file_path, metadata_path = self._set_paths()
            if file_path.exists():
                logger.debug(f"Classifier data for {file_path} exists, loading existing data.")
                self._parse_classified_data(file_path, metadata_path)
            else:
                logger.debug(f"Classifier data for {file_path} does not exists, parsing new data.")
                self._parse_metadata(metadata_path)
                if self._check_metadata():
                    self._parse_data()
                    selected_images_length = len(
                        self.omero_data.selected_images
                    )
                    self.omero_data.selected_classes = [
                        "unassigned" for _ in range(selected_images_length)
                    ]

            if not self.omero_data.selected_images:
                logger.warning("Could not load images.")
                return

            self.update_class_options(self.class_options)
            self.image_navigator.current_index = 0
            self.image_navigator.update_image()

            # Initialize TrainingDataSaver after class_name is set
            self.training_data_saver = TrainingDataSaver(
                self.class_name,
                self.omero_data,
                self.user_data,
                self.image_navigator,
            )
            logger.info(
                f"TrainingDataSaver initialized for classifier {self.class_name}"
            )

        except ValueError as ve:
            logger.error(f"ValueError: {ve}")
            _show_error_message(f"Error: {ve}")

        except FileNotFoundError as fnf_error:
            logger.error(f"FileNotFoundError: {fnf_error}")
            _show_error_message(
                f"Metadata file not found: {metadata_path}"
            )

        except json.JSONDecodeError as json_error:
            logger.error(f"JSONDecodeError: {json_error}")
            _show_error_message(
                f"Error decoding metadata file: {metadata_path}"
            )

        except Exception as e:  # noqa: BLE001
            logger.error(f"Unexpected error: {e}")
            _show_error_message(f"An unexpected error occurred: {e}")

    def _set_paths(self):
        plate = self.omero_data.plate_id
        well = self.omero_data.well_pos_list[0]
        image = self.omero_data.image_input
        timepoint = self.user_data.timepoint

        file_name = f"{plate}_{well}_{image}_{timepoint}.npy"
        classifier_dir = (
            Path.home() / "omeroscreen_trainingdata" / self.class_name
        )
        file_path = classifier_dir / file_name
        meta_data_path = classifier_dir / "metadata.json"
        return file_name, file_path, meta_data_path

    def _parse_classified_data(self, file_path, metadata_path):
        if metadata_path.exists():
            logger.info(
                f"Classifier data file and metadata file exist: {file_path}, loading data"
            )
            self._parse_metadata(metadata_path)
            self.user_data.populate_from_dict(self.user_data_dict)
            self._parse_saved_imagedata(file_path)
        else:
            logger.error(
                f"Classifier data file {file_path} but metadata file {metadata_path} not found."
            )
            _show_error_message(
                f"metadata file not found in: {self.class_name} directory"
            )

    def _parse_saved_imagedata(self, file_path):
        try:
            self.omero_data.selected_crops, self.omero_data.selected_labels = (
                np.load(file_path, allow_pickle=True).item()["data"]
            )
            self.omero_data.selected_classes = np.load(
                file_path, allow_pickle=True
            ).item()["target"]
            masked_images = self._apply_mask_to_images()
            processed_masked_images = [
                fill_missing_channels(img, self.user_data.channels)
                for img in masked_images
            ]
            if self.user_data.contour:
                processed_masked_images = [
                    draw_contours(img, label)
                    for img, label in zip(
                        processed_masked_images,
                        self.omero_data.selected_labels,
                    )
                ]
            self.omero_data.selected_images = processed_masked_images
            logger.info(
                f"Loaded {len(self.omero_data.selected_images)} images."
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error loading images: {e}")
            _show_error_message(f"Error loading images: {e}")

    def _apply_mask_to_images(self):
        """
        Nullify pixels in color images that don't overlap with the corresponding masks.
        Images are expected to be in the shape of (H, W, 3) and masks in the shape of (H, W).
        """
        masked_images = []
        for image, mask in zip(
            self.omero_data.selected_crops, self.omero_data.selected_labels
        ):  # type: ignore
            # Ensure the mask is expanded to match the image's 3 channels
            expanded_mask = (
                np.repeat(mask[:, :, np.newaxis], image.shape[2], axis=2) > 0
            )
            # Apply the expanded mask to the image
            masked_image = np.where(expanded_mask, image, 0)
            masked_images.append(masked_image)

        return masked_images

    def _parse_metadata(self, metadata_path: Path):
        try:
            with metadata_path.open("r") as json_file:
                metadata = json.load(json_file)
        except FileNotFoundError as e:
            logger.error(f"Metadata file not found: {metadata_path}")
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}"
            ) from e
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding metadata file: {metadata_path}")
            raise json.JSONDecodeError(
                f"Error decoding metadata file: {metadata_path}"
            ) from e
        self.user_data_dict = metadata["user_data"]
        self.class_options = metadata["class_options"]
        self.user_data_dict["well"] = omero_data.well_pos_list[0]

    def _check_metadata(self) -> bool:
        required_keys = {
            "well": str,
            "segmentation": str,
            "reload": bool,
            "crop_size": int,
            "cellcycle": str,
            "timepoint": int,
            "contour": bool,
            "channels": list,
        }

        for key, expected_type in required_keys.items():
            if key == "timepoint" and key not in self.user_data_dict:
                self.user_data_dict[key] = 0
            elif key != "timepoint" and key not in self.user_data_dict:
                print(f"Missing key: {key}")
                return False
            if self.user_data_dict[key] is None:
                print(f"None value for key: {key}")
                return False
            if not isinstance(self.user_data_dict[key], expected_type):
                print(
                    f"Incorrect type for key: {key}. Expected {expected_type}, got {type(self.user_data_dict[key])}"
                )
                return False

        return True

    def _parse_data(self):
        userdata.populate_from_dict(self.user_data_dict)
        manager = CroppedImageParser(omero_data, userdata)
        manager.parse_crops()
        data_selector = RandomImageParser(
            omero_data, userdata, classifier=True
        )
        data_selector.parse_random_images()
        logger.info(f"Loaded {len(omero_data.selected_images)} images.")

    def update_class_options(self, class_options: list[str]):
        self.image_navigator.class_options = class_options
        self.image_navigator.class_choice.choices = class_options
        self.image_navigator.update_class_choice()

    def next_image(self):
        if not self.omero_data.selected_classes:
            print("No images loaded.")
            return
        self.image_navigator.next_image()

    def previous_image(self):
        if not self.omero_data.selected_classes:
            print("No images loaded.")
            return
        self.image_navigator.previous_image()

    def save_training_data(self):
        if self.training_data_saver:
            self.training_data_saver._save_data()
        else:
            print("Training data saver not initialized.")

    def setup_key_bindings(self, viewer):
        @viewer.bind_key("w")
        def trigger_next_image(event=None):
            self.next_image()

        @viewer.bind_key("q")
        def trigger_previous_image(event=None):
            self.previous_image()

    def create_container(self):
        return Container(
            widgets=[
                self.load_image,
                self.previous_image,
                self.next_image,
                self.image_navigator.class_choice,
                self.save_training_data,
            ]
        )


class TrainingDataSaver:
    def __init__(
        self,
        classifier_name: str,
        omero_data: OmeroData,
        user_data: UserData,
        image_navigator: ImageNavigator,
    ):
        self.classifier_name = classifier_name
        self.omero_data = omero_data
        self.user_data = user_data
        self.image_navigator = image_navigator
        self.home_dir = Path.home() / "omeroscreen_trainingdata"
        self.classifier_dir = self.home_dir / self.classifier_name
        self.file_name, self.file_path, self.meta_data_path = self._set_paths()
        self.training_dict = self._create_training_dict()
        self.metadata = self._create_metadata_dict()

    def _save_data(self):
        try:
            self._validate_classifier_name(self.classifier_name)
            if not self.classifier_dir.exists():
                self._create_and_save()
                return

            file_check = self._check_directory_contents()
            self._handle_saving_logic(file_check)
        except Exception as e:  # noqa: BLE001
            logger.error(e)
            _show_error_message(str(e))

    def _create_and_save(self):
        self._create_directory(self.classifier_dir)
        self.save_both()

    def _check_directory_contents(self):
        logger.info(f"Directory {self.classifier_dir} already exists.")
        contents = list(self.classifier_dir.iterdir())
        return self.check_files(contents)

    def _handle_saving_logic(self, file_check):
        if file_check in ["empty", "neither"]:
            self.save_both()
        elif file_check == "metadata":
            self._handle_metadata_present()
        elif file_check == "training_data":
            self._handle_training_data_present()
        elif file_check == "both":
            self._handle_both_present()

    def _handle_metadata_present(self):
        if self.compare_metadata():
            self._save_training_data()
        elif _show_confirmation_dialog(
            f"Metadata has changed. Do you want to overwrite them and save {self.file_name}?"
        ):
            self.save_both()

    def _handle_training_data_present(self):
        if _show_confirmation_dialog(
            f"The file {self.file_name} already exists without metadata. Do you want to overwrite the file and save the metadata?"
        ):
            self.save_both()

    def _handle_both_present(self):
        if self.compare_metadata():
            if _show_confirmation_dialog(
                f"Do you want to overwrite {self.file_name}?"
            ):
                self._save_training_data()
        elif _show_confirmation_dialog(
            "{self.file_name} and metadata have changed. Do you want to overwrite both?"
        ):
            self.save_both()

    def save_both(self):
        self.training_dict = self._create_training_dict()
        np.save(self.file_path, self.training_dict)
        self._save_metadata(self.meta_data_path, self.metadata)
        logger.info(f"File and metadata saved to: {self.classifier_dir}")
        _show_success_message(
            f"Data for {self.file_name} and metadata successfully saved."
        )

    def _save_training_data(self):
        self.training_dict = self._create_training_dict()
        np.save(self.file_path, self.training_dict)
        logger.info(
            f"File saved to: {self.file_path}, metadata already present"
        )
        _show_success_message(
            f"Data for image {self.file_name} successfully saved, with metadata present."
        )

        self._upload_npy_to_omero()
        
    def _set_paths(self):
        plate = self.omero_data.plate_id
        well = self.omero_data.well_pos_list[0]
        image = self.omero_data.image_input
        time_point = self.user_data.timepoint
        file_name = f"{plate}_{well}_{image}_{time_point}.npy"
        file_path = self.classifier_dir / file_name
        meta_data_path = self.classifier_dir / "metadata.json"
        return file_name, file_path, meta_data_path

    def check_files(self, contents):
        has_metadata = any(file.name == "metadata.json" for file in contents)
        has_self_file = any(file.name == self.file_name for file in contents)

        if not contents:
            return "empty"
        if has_metadata:
            return "both" if has_self_file else "metadata"
        return "training_data" if has_self_file else "neither"

    def compare_metadata(self) -> bool:
        with self.meta_data_path.open("r") as json_file:
            existing_metadata = json.load(json_file)
        return existing_metadata == self.metadata

    def _validate_classifier_name(self, text_input: str):
        if not text_input.strip():
            logger.error("No classifier name provided.")
            raise ValueError(
                "Failed to create directory: no classifier name provided."
            )

    def _create_directory(self, directory: Path):
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise ValueError(
                f"Failed to create directory {directory}: {e}"
            ) from e

    def _create_training_dict(self) -> dict:
        logger.info(
            f"Creating training data dictionary with {len(self.omero_data.selected_classes)} entries."
        )
        return {
            "data": (
                self.omero_data.selected_crops,
                self.omero_data.selected_labels,
            ),
            "target": self.omero_data.selected_classes,
        }

    def _create_metadata_dict(self) -> dict:
        user_data_dict = asdict(self.user_data)
        user_data_dict.pop("well", None)
        return {
            "user_data": user_data_dict,
            "class_options": self.image_navigator.class_options,
        }

    def _save_metadata(self, meta_data_path: Path, metadata: dict):
        with meta_data_path.open("w") as json_file:
            json.dump(metadata, json_file)

    @omero_connect
    def _upload_npy_to_omero(self, conn=None):

        npy_file_path = Path(self.file_path) 
        dataset_name = self.classifier_name + "_Dataset"
        dataset_id = self.get_dataset_ids_by_name(conn, dataset_name)
        dataset = conn.getObject("Dataset", dataset_id[0])
        
        if dataset is None:
            raise Exception(f"Dataset not found. Dataset ID: {dataset_id}")

       # Check for existing File Annotations and delete any with the same name
        for ann in dataset.listAnnotations():
            if isinstance(ann, FileAnnotationWrapper):
                file_name = ann.getFile().getName()
                if file_name == npy_file_path.name:  # If file names match
                    print(f"Existing file found: {file_name}. Deleting it.")
                    conn.deleteObjects("Annotation", [ann.getId()], wait=True)
                    print(f"File {file_name} has been deleted.")

        # Upload the .npy file as a File Annotation
        file_ann = conn.createFileAnnfromLocalFile(
            str(npy_file_path),
            mimetype="application/octet-stream",  # MIME type for binary files
            ns="example.namespace",               # Namespace, customizable if desired
            desc="Uploaded .npy file"             # Description
        )

        # Add the File Annotation to the dataset
        dataset.linkAnnotation(file_ann)
        _show_success_message("npy file successfully uploaded to omero")

    def get_dataset_ids_by_name(self, conn, dataset_name):
        query = "select obj from Dataset obj where obj.name = :name"
        params = ParametersI()
        params.addString("name", dataset_name)

        # Use findAllByQuery to get all datasets that match the given name
        datasets = conn.getQueryService().findAllByQuery(query, params)

        if datasets:
            dataset_ids = [dataset.getId().getValue() for dataset in datasets]
            return dataset_ids
        else:
            return None  # Return None if no matching datasets are found

def _show_error_message(message: str):
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Warning)
    msg_box.setText(message)
    msg_box.setWindowTitle("Error")
    msg_box.setStandardButtons(QMessageBox.Ok)
    msg_box.exec_()

def _show_success_message(message: str):
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Information)
    msg_box.setText(message)
    msg_box.setWindowTitle("Success")
    msg_box.setStandardButtons(QMessageBox.Ok)
    msg_box.exec_()

def _show_confirmation_dialog(message: str) -> bool:
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Warning)
    msg_box.setText(message)
    msg_box.setWindowTitle("Warning")
    msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    msg_box.setDefaultButton(QMessageBox.No)
    reply = msg_box.exec_()
    return reply == QMessageBox.Yes
