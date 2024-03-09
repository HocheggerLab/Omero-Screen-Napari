import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from dotenv import load_dotenv
from omero.gateway import _DatasetWrapper, _PlateWrapper, _WellWrapper

from omero_screen_napari import set_env_vars

logger = logging.getLogger("omero-screen-napari")


def get_project_id() -> int:
    """_summary_
    Fetch Omero PROJECT_ID from the environment, converting it to int if necessary
    Returns:
        int: project id to find flat field masks and segmentation directory in Omero
    """
    default_project_id = 0
    dotenv_path = set_env_vars()
    load_dotenv(dotenv_path=dotenv_path, override=True)
    logger.info(f"Loading environment variables from {dotenv_path}")  # noqa: G004
    return int(os.getenv("PROJECT_ID", default_project_id))


def get_data_path() -> Path:
    """_summary_
    Fetch data_path from the environment
    Returns:
        Path: path to folder that saves the csv data to avoid reloading from the server.
    """
    default_data_path = "default_data_path"
    dotenv_path = set_env_vars()
    load_dotenv(dotenv_path=dotenv_path, override=True)
    return Path.home() / Path(os.getenv("DATA_PATH", default_data_path))


@dataclass
class OmeroData:
    """
    Dataclass to store all the data related to the omero project and plate.
    """

    # Screen data
    project_id: int = field(default_factory=get_project_id)
    screen_dataset: _DatasetWrapper = field(
        default_factory=_DatasetWrapper
    )  # dataset with flatfield masks and segementations
    plate_id: int = field(default_factory=int)
    plate_name: str = field(default_factory=str)
    plate: _PlateWrapper = field(default_factory=_PlateWrapper)
    plate_data: pl.LazyFrame = field(default_factory=pl.LazyFrame)
    data_path: Path = field(default_factory=get_data_path)
    csv_path: Path = field(default_factory=Path)
    flatfield_masks: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    pixel_size: tuple = field(default_factory=tuple)
    channel_data: dict = field(default_factory=dict)
    intensities: dict = field(default_factory=dict)

    # Well data
    well: list[_WellWrapper] = field(default_factory=list)
    well_name: list[str] = field(default_factory=list)
    well_id: list[int] = field(default_factory=list)
    metadata: list[dict] = field(default_factory=list)

    # Image data
    images: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    image_ids: list = field(default_factory=list)
    labels: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    label_ids: list = field(default_factory=list)

    # Stitched images
    stitched_images: np.ndarray = field(default_factory=lambda: np.empty((0,)))

    def reset(self):
        """
        Reset the omero data to default values.
        This is used when reading in a new plate from napari.
        """
        self.__init__()
