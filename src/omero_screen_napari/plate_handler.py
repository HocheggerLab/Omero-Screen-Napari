import logging
import random
from pathlib import Path
from typing import List, Tuple

import omero
import polars as pl
from ezomero import get_image
from omero.gateway import (
    BlitzGateway,
    FileAnnotationWrapper,
    MapAnnotationWrapper,
    PlateWrapper,
)
from qtpy.QtWidgets import QMessageBox
from skimage import exposure
from tqdm import tqdm

from omero_screen_napari._omero_utils import omero_connect
from omero_screen_napari.omero_data import OmeroData
from omero_screen_napari.omero_data_singleton import (
    omero_data,
    reset_omero_data,
)

logger = logging.getLogger("omero-screen-napari")


@omero_connect
def parse_plate_data(
    omero_data,
    plate_id: int,
    conn: BlitzGateway = None,
) -> None:
    """
    This functions controls the flow of the data
    via the seperate parser classes to the omero_data class that stores omero plate metadata
    to be supplied to the napari viewer.
    This function is passed to welldata_widget and supplied by parameters from the
    plugin gui.
    Args:
        plate_id (int): Omero Screen Plate ID number provided by welldata_widget
        well_list (List[str]): List of wells provided by welldata_widget (e.g ['A1', 'A2'])
        images (str): String Describing the combinations of images to be supplied for each well
        The images options are 'All'; 0-3; 1, 3, 4. This will be processed by the ImageHandler to retrieve the relevant images
    """
    # check if plate-ID already supplied, if it is then skip PlateHandler
    if plate_id != omero_data.plate_id:
        reset_omero_data()
        try:
            logger.info(f"Retrieving new plate data for plate {plate_id}")  # noqa: G004
            omero_data.plate_id = plate_id
            plate = conn.getObject("Plate", plate_id)
            csv_parser = CsvFileParser(omero_data, plate)
            csv_parser.parse_csv()
            channel_parser = ChannelDataParser(omero_data, plate)
            channel_parser.parse_channel_data()
            flatfield_parser = FlatfieldMaskParser(omero_data, conn)
            flatfield_parser.parse_flatfieldmask()
            scale_intensity_parser = ScaleIntensityParser(omero_data)
            scale_intensity_parser.parse_intensities()
            pixel_size_parser = PixelSizeParser(omero_data, plate)
            pixel_size_parser.parse_pixel_size_values()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error parsing plate data: {e}")
            # # Show a message box with the error message
            # msg_box = QMessageBox()
            # msg_box.setIcon(QMessageBox.Warning)
            # msg_box.setText(str(e))
            # msg_box.setWindowTitle("Error")
            # msg_box.setStandardButtons(QMessageBox.Ok)
            # msg_box.exec_()
    else:
        logger.info(f"Plate data for plate {plate_id} already exists.")


# -----------------------------------------------CSV FILE -----------------------------------------------------
class CsvFileParser:
    """
    Class to handle csv file retrieval and processing from Omero Plate
    Class methods:
    handle_csv: Orchestrate the csv file handling. Check if csv file is available, if not download it.
    and make it available to the omero_data class as a polars LazyFrame object.
    _csv_available: Helper; checks if any csv file in the target directory contains the string self.plate_id in its name.
    _get_csv_file: Helper; gets the csv file from the Omero Plate, prioritizing '_cc.csv' files over 'final_data.csv' files.
    _download_csv: Helper; downloads the csv file from the Omero Plate. If _cc.csv the file name will be

    """

    def __init__(
        self, omero_data: OmeroData, plate: omero.gateway.PlateWrapper
    ):
        self._omero_data = omero_data
        self._plate_id: int = omero_data.plate_id
        self._plate = plate
        self._data_path: Path = omero_data.data_path
        self._csv_file_path: Path = None

    def parse_csv(self):
        """Orchestrate the csv file handling. Check if csv file is available, if not download it.
        and make it available to the omero_data class as a polars LazyFrame object.

        """
        if not self._csv_available():
            logger.info("Downloading csv file from Omero")  # noqa: G004
            self._get_csv_file()
            self._download_csv()
        else:  # csv file already exists
            logger.info(
                "CSV file already exists in local directory. Skipping download."
            )
        omero_data.plate_data = pl.scan_csv(self._csv_file_path)
        self._omero_data.csv_path = self._csv_file_path

    # helper functions for csv handling

    def _csv_available(self):
        """
        Check if any csv file in the directory contains the string self.plate_id in its name.
        """
        # make the directory to store the csv file if it does not exist
        self._data_path.mkdir(exist_ok=True)
        for file in self._data_path.iterdir():
            if file.is_file() and str(self._plate_id) in file.name:
                self._csv_file_path = file
                return True

    def _get_csv_file(self):
        """
        Get the csv file from the Omero Plate, prioritizing '_cc.csv' files over 'final_data.csv' files.
        """
        self._original_file = None
        file_anns = self._plate.listAnnotations()
        for ann in file_anns:
            if isinstance(ann, FileAnnotationWrapper):
                name = ann.getFile().getName()
                if name.endswith("_cc.csv"):
                    self._file_name = name
                    self._original_file = ann.getFile()
                    break  # Prioritize and stop searching if _cc.csv is found
                elif (
                    name.endswith("final_data.csv")
                    and self._original_file is None
                ):
                    self._file_name = name
                    self._original_file = ann.getFile()
                    # Don't break to continue searching for a _cc.csv file

        if self._original_file is None:
            logger.error("No suitable csv file found for the plate.")
            raise ValueError("No suitable csv file found for the plate.")

    def _download_csv(self):
        """
        Download the csv file from the Omero Plate. If _cc.csv the file name will be
        {plate_id}_cc.csv, otherwise {plate_id}_data.csv
        """
        saved_name = f"{self._plate_id}_{self._file_name.split('_')[-1]}"
        logger.info(f"Downloading csv file {self._file_name} to {saved_name}")  # noqa: G004
        self._csv_file_path = self._data_path / saved_name
        file_content = self._original_file.getFileInChunks()
        with open(self._csv_file_path, "wb") as file_on_disk:
            for chunk in file_content:
                file_on_disk.write(chunk)


# -----------------------------------------------CHANNEL DATA -----------------------------------------------------


class ChannelDataParser:
    """
    Class to handle channel data retrieval and processing from Omero Plate.
    class methods:
    _get_channel_data: Coordinates the retrieval of channel data from Omero Plate
    _get_map_ann: Hhlper method that scans plate map annotations for channel data
    and raises an exceptions if no suitable annotation is found
    _tidy_up_channel_data: Helper method that processes the channel data to a dictionary
    """

    def __init__(self, omero_data: OmeroData, plate: PlateWrapper):
        self._omero_data = omero_data
        self._plate = plate
        self._plate_id: int = omero_data.plate_id

    def parse_channel_data(self):
        """Process channel data to dictionary {channel_name: channel_index}
        and add it to viewer_data; raise exception if no channel data found."""
        self._get_map_ann()
        self._tidy_up_channel_data()
        self._omero_data.channel_data = self.channel_data
        logger.info(
            f"Channel data: {self.channel_data} loaded to omero_data.channel_data"
        )  # noqa: G004

    def _get_map_ann(self):
        """
        Get channel information from Omero map annotations
        or raise exception if no appropriate annotation is found.
        """

        annotations = self._plate.listAnnotations()

        map_annotations = [
            ann for ann in annotations if isinstance(ann, MapAnnotationWrapper)
        ]

        if not map_annotations:
            logger.error(
                f"No MapAnnotations found for plate {self._plate_id}."
            )  # noqa: G004
            raise ValueError(
                f"No MapAnnotations found for plate {self._plate_id}."
            )

        # Search for a map annotation with "dapi" or "hoechst" in its value (case-insensitive)
        for map_ann in map_annotations:
            ann_value = map_ann.getValue()
            # Assuming the value is a list of tuples or similar structure
            for key, _value in ann_value:
                if key.lower() in ["dapi", "hoechst"]:
                    self.map_annotations = ann_value
                    return  # Return the first matching map annotation's value

        # If no matching annotation is found, raise a specific exception
        logger.error(
            f"No DAPI or Hoechst channel information found for plate {self._plate_id}."  # noqa: G004
        )  # noqa: G004
        raise ValueError(
            f"No DAPI or Hoechst channel information found for platplate {self._omero_data.plate_id}."
        )

    def _tidy_up_channel_data(self):
        """
        Convert channel map annotation from list if tuples to dictionary.
        Eliminate spaces and swap Hoechst to DAPI if necessary.
        """
        channel_data = dict(self.map_annotations)
        self.channel_data = {
            key.strip(): value for key, value in channel_data.items()
        }
        if "Hoechst" in self.channel_data:
            self.channel_data["DAPI"] = self.channel_data.pop("Hoechst")
        # check if one of the channels is DAPI otherwise raise exception


# -----------------------------------------------FLATFIELD CORRECTION-----------------------------------------------------


class FlatfieldMaskParser:
    def __init__(self, omero_data: OmeroData, conn: BlitzGateway):
        self._omero_data: OmeroData = omero_data
        self._conn: BlitzGateway = conn

    def parse_flatfieldmask(self):
        self._load_dataset()
        self._get_flatfieldmask()
        self._get_map_ann()
        self._check_flatfieldmask()
        omero_data.flatfield_mask = self._flatfield_array

    def _load_dataset(self):
        """
        Looks for a data set that matches the plate name in the screen project
        with id supplied by omero_data.project_id via the env variable.
        if it finds the data set it assigns it to the omero_data object
        otherwise it will throw an error becuase the program cant proceed without
        flatfielmasks and segmentationlabels stored in the data set.

        Raises:
            ValueError: If the plate has not been assigned a dataset.
        """
        project = self._conn.getObject("Project", self._omero_data.project_id)
        if dataset := self._conn.getObject(
            "Dataset",
            attributes={"name": str(self._omero_data.plate_id)},
            opts={"project": project.getId()},
        ):
            self._omero_data.screen_dataset = dataset
        else:
            logger.error(
                f"The plate {omero_data.plate_name} has not been assigned a dataset."
            )
            raise ValueError(
                f"The plate {omero_data.plate_name} has not been assigned a dataset."
            )

    def _get_flatfieldmask(self):
        """Gets flatfieldmasks from project linked to screen"""
        flatfield_mask_name = f"{self._omero_data.plate_id}_flatfield_masks"
        logger.debug(f"Flatfield mask name: {flatfield_mask_name}")  # noqa: G004
        flatfield_mask_found = False
        for image in self._omero_data.screen_dataset.listChildren():
            image_name = image.getName()
            logger.debug(f"Image name: {image_name}")  # noqa: G004
            if flatfield_mask_name == image_name:
                flatfield_mask_found = True
                flatfield_mask_id = image.getId()
                self._flatfield_obj, self._flatfield_array = get_image(
                    self._conn, flatfield_mask_id
                )
                break  # Exit the loop once the flatfield mask is found
        if not flatfield_mask_found:
            logger.error(
                f"No flatfieldmasks found in dataset {self._omero_data.screen_dataset}."
            )  # noqa: G004
            raise ValueError(
                f"No flatfieldmasks found in dataset {self._omero_data.screen_dataset}."
            )

    def _get_map_ann(self):
        """
        Get channel information from Omero map annotations
        or raise exception if no appropriate annotation is found.
        """
        annotations = self._flatfield_obj.listAnnotations()
        map_annotations = [
            ann for ann in annotations if isinstance(ann, MapAnnotationWrapper)
        ]
        if not map_annotations:
            logger.error(
                f"No Flatfield Mask Channel info found in datset {self._omero_data.screen_dataset}."
            )  # noqa: G004
            raise ValueError(
                f"No Flatfield Mask Channel info found in datset {self._omero_data.screen_dataset}."
            )
        # Search for a map annotation with "dapi" or "hoechst" in its value (case-insensitive)
        for map_ann in map_annotations:
            ann_value = map_ann.getValue()
            # Assuming the value is a list of tuples or similar structure
            for _key, value in ann_value:
                if value.lower().strip() in ["dapi", "hoechst"]:
                    self._flatfield_channels = dict(ann_value)
                    logger.debug(
                        f"Flatfield mask channels: {self._flatfield_channels}"  # noqa: G004
                    )  # noqa: G004
                    return
        # If no matching annotation is found, raise a specific exception
        logger.error(
            "No DAPI or Hoechst channel information found for flatfieldmasks."
        )  # noqa: G004
        raise ValueError(
            "No DAPI or Hoechst channel information found for flatfieldmasks."
        )

    def _check_flatfieldmask(self):
        """
        Checks if the mappings with the plate channel date are consistent.
        self.flatfield_channels are {'channel_0' : 'DAPI}
        omero_data.channel_data are {'DAPI' : 0}

        Raises :
            InconsistencyError: If the mappings are inconsistent.
        """

        class InconsistencyError(ValueError):
            pass

        if list(self._flatfield_channels.values()) != list(
            omero_data.channel_data.keys()
        ):
            error_message = "Inconsistency found: flatfield_mask and plate_map have different channels"
            logger.error(error_message)
            raise InconsistencyError(error_message)


# -----------------------------------------------SCALE INTENSITY -----------------------------------------------------


class ScaleIntensityParser:
    """
    The class extracts the caling values for image contrasts to display the different channels in napari.
    To compare intesities across different wells a single global contrasting value is set. This is based on the
    the min and max values for each channel. These data are extracted from the polars LazyFrame object stored in
    omero_data.plate_data.
    """

    def __init__(self, omero_data: OmeroData):
        self._omero_data = omero_data
        self._plate_data: pl.LazyFrame = omero_data.plate_data
        self._keyword: str = None
        self._intensities: Tuple[dict] = ({}, {})

    def parse_intensities(self):
        self._set_keyword()
        self._get_values()
        self._omero_data.intensities = self._intensities

    def _set_keyword(self):
        if not hasattr(self, "_plate_data") or self._plate_data is None:
            raise ValueError("Dataframe 'plate_data' does not exist.")

        # Initialize keyword to None
        self._keyword = None

        # Check for presence of "_cell" and "_nucleus" in the list of strings
        has_cell = any("_cell" in s for s in self._plate_data.columns)
        has_nucleus = any("_nucleus" in s for s in self._plate_data.columns)

        # Set keyword based on presence of "_cell" or "_nucleus"
        if has_cell:
            self._keyword = "_cell"
        elif has_nucleus:
            self._keyword = "_nucleus"
        else:
            # If neither is found, raise an exception
            raise ValueError(
                "Neither '_cell' nor '_nucleus' is present in the dataframe columns."
            )

    def _get_values(self):
        """
        Filter the dataframe to only include columns with the keyword.
        """
        intensity_dict = {}
        for channel, _value in self._omero_data.channel_data.items():
            cols = (
                f"intensity_max_{channel}{self._keyword}",
                f"intensity_min_{channel}{self._keyword}",
            )
            for col in cols:
                if col not in self._plate_data.columns:
                    logger.error(f"Column '{col}' not found in DataFrame.")
                    raise ValueError(f"Column '{col}' not found in DataFrame.")

            max_value = (
                self._plate_data.select(pl.col(cols[0])).mean().collect()
            )
            min_value = (
                self._plate_data.select(pl.col(cols[1])).min().collect()
            )

            intensity_dict[channel] = (
                int(min_value[0, 0]),
                int(max_value[0, 0]),
            )

        self._intensities = intensity_dict


# -----------------------------------------------PIXEL SIZE -----------------------------------------------------


class PixelSizeParser:
    """
    The class extracts the pixel size from the plate metadata. For this purpose it uses the first well and image
    to extract the pixel size in X and Y dimensions. The pixel size is then stored in the omero_data class.
    """

    def __init__(
        self, omero_data: OmeroData, plate: omero.gateway.PlateWrapper
    ):
        self._omero_data = omero_data
        self._plate = plate
        self._random_wells: List = None
        self._random_images: List = None
        self._pixel_size: Tuple[int] = None

    def parse_pixel_size_values(self):
        self._check_wells_and_images()
        self._check_pixel_values()
        self._omero_data.pixel_size = self._pixel_size

    def _check_wells_and_images(self):
        """
        Check if any wells are found in the plate and if each well has images.
        Extract the first image of two random wells.
        """
        wells = list(self._plate.listChildren())
        if not wells:
            logger.debug("No wells found in the plate, raising ValueError.")
            raise ValueError("No wells found in the plate.")
        self._random_wells = random.sample(wells, 2)
        image_list = []
        for i, well in enumerate(self._random_wells):
            try:
                image = well.getImage(0)  # Attempt to get the first image
                image_list.append(image)
            except Exception as e:  # Catches any exception raised by getImage(0)  # noqa: BLE001
                logger.error(f"Unable to retrieve image from well {i}: {e}")
                raise ValueError(
                    f"Unable to retrieve image from well {i}: {e}"
                ) from e

        self._random_images = image_list

    def _get_pixel_values(self, image) -> tuple:
        """
        Get pixel values from a single OMERO image object.
        Returns a tuple of floats for x, y values.
        """
        x_size = image.getPixelSizeX()
        y_size = image.getPixelSizeY()

        # Check if either x_size or y_size is None before rounding
        if x_size is None or y_size is None:
            logger.error(
                "No pixel data found for the image, raising ValueError."
            )
            raise ValueError("No pixel data found for the image.")

        # Since both values are not None, proceed to round them
        return (round(x_size, 1), round(y_size, 1))

    def _check_pixel_values(self):
        """
        Check if pixel values from the two random images are identical and not 0
        and load the pixel size tuple to the omero_data class.
        """
        pixel_well_1 = self._get_pixel_values(self._random_images[0])
        pixel_well_2 = self._get_pixel_values(self._random_images[1])

        if 0 in pixel_well_1 + pixel_well_2:
            logger.error("One of the pixel sizes is 0")
            raise ValueError("One of the pixel sizes is 0")
        elif pixel_well_1 == pixel_well_2:
            self._pixel_size = pixel_well_1
        else:
            logger.error("Pixel sizes are not identical between wells")
            raise ValueError("Pixel sizes are not identical between wells")
        
#-----------------------------------------------Well Data-----------------------------------------------------


class WellDataParser:
    """
    """
    def __init__(self, omero_data: OmeroData, plate: omero.gateway.PlateWrapper, well_pos: str):
        self._omero_data = omero_data
        self._plate = plate
        self._well_pos = well_pos
        self._well = None
        self._well_id = None
        self._images = None

    def parse_well(self): 
        well_found = False  
        for well in self._plate.listChildren():
            if well.getWellPos() == self._well_pos:
                self._well = well
                self._well_id = well.getId()
                well_found = True
                break
        if not well_found:
            logger.error('f"Well with position {well_pos} does not exist."')  # Raise an error if the well was not found
            raise ValueError(f"Well with position {self.well_pos} does not exist.")
        
    def parse_well_data(self):
        self._load_well_specific_data()
        self._get_well_metadata()
        self._get_images()
    def _get_well_object(viewer_data, conn, well_pos: str, index: int, images: str):
    if well_pos not in viewer_data.well_name:
        viewer_data.well_name.append(well_pos)
        well_found = False  # Initialize a flag to check if the well is found
        # get well object
        for well in viewer_data.plate.listChildren():
            if well.getWellPos() == well_pos:
                viewer_data.well.append(well)
                viewer_data.well_id.append(well.getId())
                _load_well_specific_data(viewer_data, well_pos)
                _get_well_metadata(viewer_data, index)
                _get_images(viewer_data, images, index, conn)
                well_found = (
                    True  # Set the flag to True when the well is found
                )
                break  # Exit the loop as the well is found
        if not well_found:  # Raise an error if the well was not found
            raise ValueError(f"Well with position {well_pos} does not exist.")
       
    else:
        logger.info(f"Well {well_pos} already retrieved")


def _load_well_specific_data(viewer_data, well_pos: str):
    """
    Load and process data for a specific well from the CSV file.
    """
    if not viewer_data.csv_path:
        logger.error("CSV path is not set in viewer_data.")
        return

    try:
        data = pd.read_csv(viewer_data.csv_path, index_col=0)
        _filter_csv_data(viewer_data, data, well_pos)
    except pd.errors.EmptyDataError:
        logger.error("CSV file is empty or all data is NA")
    except pd.errors.ParserError:
        logger.error("Error parsing CSV file")
    except Exception as e:
        logger.error(f"Unexpected error processing CSV: {e}")
        raise


def _filter_csv_data(viewer_data, data, wellpos):
    filtered_data = data[data["well"] == wellpos]
    if filtered_data.empty:
        logger.warning(
            "No data found for well %s in 'final_data.csv'",
            viewer_data.well.getWellPos(),
        )
    else:
        viewer_data.plate_data = pd.concat(
            [viewer_data.plate_data, filtered_data]
        )
        viewer_data.intensities = _get_intensity_values(viewer_data, data)
        logger.info("'final_data.csv' processed successfully")


# TODO this function needs to check if the well mapp ann is actually the right data
def _get_well_metadata(viewer_data, index):
    map_ann = None
    for ann in viewer_data.well[index].listAnnotations():
        if ann.getValue():
            map_ann = dict(ann.getValue())
    if map_ann:
        viewer_data.metadata.append(map_ann)
    else:
        raise ValueError(
            f"No map annotation found for well {viewer_data.well[index].getWellPos()}"