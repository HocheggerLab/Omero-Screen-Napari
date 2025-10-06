"""
This module handles the widget to call Omero and load all images from a single well
from multiple aligned plates into napari.
The plugin can be run from napari as Aligned Plate Widget under Plugins.
"""

import logging
from typing import Optional

import numpy as np
import re
from magicgui import magic_factory
from magicgui.widgets import Container
from napari.layers import Image
from napari.viewer import Viewer

from omero_screen_napari.omero_data_singleton import omero_data
from omero_screen_napari.welldata_api import (
    get_plate_alignments,
    parse_omero_data,
)
from ._welldata_widget import (
    add_label_layers,
    clear_viewer_layers,
    set_color_maps,
)

logger = logging.getLogger("omero-screen-napari")


def aligned_plate_widget_gui():
    """
    This function combines the widgets into a single widget.
    """
    # Call the magic factories to get the widget instances
    aligned_plate_widget_instance = aligned_plate_widget()
    return Container(
        widgets=[
            aligned_plate_widget_instance,
        ]
    )


# Widget to call Omero and load well images
@magic_factory(call_button="Enter")
def aligned_plate_widget(
    viewer: Viewer,
    plate_id: str = "Plate ID",
    well_pos: str = "Well Position",
    image: int = 0,
) -> None:
    """
    This function is a widget for handling well data in a napari viewer.
    It retrieves data based on the provided plate ID and well position,
    and then adds the images and labels to the viewer. It also handles metadata,
    sets color maps, and adds label layers to the viewer.
    """
    # Single well only
    if not re.match("^[A-Z]+[0-9]+$", well_pos):
        raise ValueError("Invalid well position: " + well_pos)

    # Get alignment for the plate
    alignments = get_plate_alignments(plate_id)
    plates = alignments["plate"].unique()
    logger.info("Loaded alignments for plates: %s", plates)

    all_channels = set()

    options = ["ignore_result_data", "ignore_scale_intensities"]
    parse_omero_data(
        omero_data, plate_id, well_pos, str(image), options=options
    )
    clear_viewer_layers(viewer)
    _add_image_to_viewer(viewer, all_channels)
    labels = omero_data.labels

    all_channels = set(omero_data.channel_data.keys())

    options.append("ignore_labels")
    for plate_other in plates:
        # Get the images (not the labels)
        parse_omero_data(
            omero_data,
            plate_other,
            well_pos,
            str(image),
            options=options,
        )
        # Translate
        df = alignments[
            (alignments["well"] == well_pos)
            & (alignments["plate"] == plate_other)
        ]
        if df.empty:
            raise Exception(
                f"Plate {plate_other} is missing alignment for well: {well_pos}"
            )
        # Translation is from plate2 to plate1 so invert
        trans = (df.iloc[0]["y"], df.iloc[0]["x"])
        logger.info("Plate %d %s translation %s", plate_other, well_pos, trans)

        # Filter channels already added to the viewer (e.g duplicate alignment channel)
        _add_image_to_viewer(viewer, all_channels, trans)

    set_color_maps(viewer)

    add_label_layers(viewer, labels)


def _add_image_to_viewer(
    viewer: Viewer, all_channels: set, trans: tuple[float, float] | None = None
) -> None:
    num_channels = omero_data.images.shape[-1]
    print(
        f"The images shape is {omero_data.images.shape} ({omero_data.images.dtype})"
    )
    channel_names: dict = {
        int(value): key for key, value in omero_data.channel_data.items()
    }
    # Create translation
    translate = None
    if trans:
        n = omero_data.images[..., 0].ndim
        translate = np.zeros(n)
        translate[-2] = trans[0]
        translate[-1] = trans[1]

    for i in range(num_channels):
        if channel_names[i] in all_channels:
            continue
        all_channels.add(channel_names[i])
        image_data = omero_data.images[..., i]
        layer = viewer.add_image(
            image_data, scale=omero_data.pixel_size, translate=translate
        )
        assert isinstance(layer, Image), (
            "Expected layer to be an instance of Image"
        )
        layer.contrast_limits_range = (0, 65535)
        layer.contrast_limits = (np.min(image_data), np.max(image_data))
        layer.blending = "additive"
        layer.name = channel_names[i]

    # Configure the scale bar
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "µm"
