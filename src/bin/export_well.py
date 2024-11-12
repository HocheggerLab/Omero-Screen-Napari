#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# This is free and unencumbered software released into the public domain.

# Anyone is free to copy, modify, publish, use, compile, sell, or
# distribute this software, either in source code form or as a compiled
# binary, for any purpose, commercial or non-commercial, and by any
# means.

# In jurisdictions that recognize copyright laws, the author or authors
# of this software dedicate any and all copyright interest in the
# software to the public domain. We make this dedication for the benefit
# of the public at large and to the detriment of our heirs and
# successors. We intend this dedication to be an overt act of
# relinquishment in perpetuity of all present and future rights to this
# software under copyright law.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

# For more information, please refer to <https://unlicense.org>
# ------------------------------------------------------------------------------

import os
import argparse
import tifffile
import logging
import numpy as np
from omero_screen_napari.omero_data_singleton import omero_data
from omero_screen_napari.utils import omero_connect
from omero_screen_napari.welldata_api import (
  UserInput, FlatfieldMaskParser, WellDataParser, ImageParser,
  parse_plate_data, stitch_images, stitch_labels)

# Check if the path is a valid directory, or raise an error
def dir_path(path):
    if os.path.isdir(path):
      return path
    else:
      raise FileNotFoundError(path)

def parse_args():
  parser = argparse.ArgumentParser(
    description='''Program to export a composed well image''')

  parser.add_argument('plate_id', type=int, help='Plate ID')
  parser.add_argument('well_pos',
    help='Well position (e.g. "A1, A2")')

  group = parser.add_argument_group('Options')
  group.add_argument('--out', default='.',
    type=dir_path,
    help='Output directory (default: %(default)s)')
  parser.add_argument('--time', default='All',
    help='Time position (e.g. All; 1-3; 2) (default: %(default)s)')

  group = parser.add_argument_group('Composition Options')
  group.add_argument('--rotation', type=float, default=0.15,
    help='Rotation angle in degrees counter clockwise (default: %(default)s)')
  group.add_argument('--ox', type=int, default=7,
    help='Pixel overlap in x (default: %(default)s)')
  group.add_argument('--oy', type=int, default=7,
    help='Pixel overlap in y (default: %(default)s)')

  group = parser.add_argument_group('Image Options')
  group.add_argument('--edge', type=int, default=7,
    help='Pixel edge for blending overlap (default: %(default)s)')
  group.add_argument('--mode', default='reflect',
    help='Mode to fill points outside the image during rotation (default: %(default)s)')

  return parser.parse_args()

@omero_connect
def export_plate(args, conn = None):
  if conn is not None:
    parse_plate_data(omero_data, args.plate_id, args.well_pos, 'All', conn, time=args.time)

    # Export each position as a series of Tiffs for a XYCZT hyperstack in ImageJ.
    logger = logging.getLogger('omero-screen-napari')
    for well_pos in omero_data.well_pos_list:
      # Adapted from well_image_parser.
      logger.info(f"Processing well {well_pos}")
      basedir = os.path.join(args.out, str(args.plate_id), str(well_pos))
      os.makedirs(basedir, exist_ok=True)
      # This loads the well into omero_data
      well_data_parser = WellDataParser(omero_data, well_pos)
      well_data_parser.parse_well()
      well = omero_data.well_list[-1]
      # Get image dimensions
      # TODO: This does not check for a MaxIntensityProject (mip) for z-stacks.
      # For now just error.
      xyzct = get_xyzct(well)
      if (xyzct[2] > 1):
        raise Exception(f'Unsupported z-stack - XYZCT: {xyzct}')
      # Extract each time point by setting omero_data crop
      time_points = range(xyzct[-1]) if args.time == 'All' else range(omero_data.crop_start[-1], omero_data.crop_length[-1])
      start = [0,0,0,0,0]
      xyzct[-1] = 1
      omero_data.crop_length = tuple(xyzct)
      for t in time_points:
        # Reset omero data and load images/labels for the timepoint using a crop
        omero_data.images = np.empty((0,))
        omero_data.labels = np.empty((0,))
        start[-1] = t
        omero_data.crop_start = tuple(start)
        image_parser = ImageParser(omero_data, well, conn)
        image_parser.parse_images_and_labels()
        # Compose and save. Compose returns YXC format so we create a view of CYX.
        stitched_images = stitch_images(omero_data, rotation=args.rotation,
          overlap_x=args.ox, overlap_y=args.oy, edge=args.edge, mode=args.mode)
        tifffile.imwrite(os.path.join(basedir, f'i{t}.tif'), 
          np.moveaxis(stitched_images, -1, 0))
        if len(omero_data.labels):
          stitched_labels = stitch_labels(omero_data, rotation=args.rotation,
            overlap_x=args.ox, overlap_y=args.oy)
          tifffile.imwrite(os.path.join(basedir, f'm{t}.tif'),
            np.moveaxis(stitched_labels, -1, 0))

  else:
    raise Exception('No connection to OMERO')

def get_xyzct(well):
  image = well.getImage(0)
  return [image.getSizeX(), image.getSizeY(), image.getSizeZ(), image.getSizeC(), image.getSizeT()]

# Gather our code in a main() function
def main():
  args = parse_args()
  export_plate(args)

# Standard boilerplate to call the main()
if __name__ == '__main__':
    main()
