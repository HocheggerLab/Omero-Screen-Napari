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
import numpy as np
import re
from omero_screen_napari.welldata_api import compose_tiles, compose_labels

# Check if the path is a valid file, or raise an error
def file_path(path):
    if os.path.isfile(path):
      return path
    else:
      raise FileNotFoundError(path)

def parse_args():
  parser = argparse.ArgumentParser(
    description='''Program to compose image tiles after rotation''')

  parser.add_argument('image', nargs='+',
    type=file_path, help='Image tile (xNyM.npy)')

  group = parser.add_argument_group('Options')
  group.add_argument('--rotation', type=float, default=0,
    help='Rotation angle in degrees counter clockwise (default: %(default)s')
  group.add_argument('--ox', type=int, default=0,
    help='Pixel offset in x (use negative for overlap) (default: %(default)s')
  group.add_argument('--oy', type=int, default=0,
    help='Pixel offset in y (use negative for overlap) (default: %(default)s')
  group.add_argument('--out', default='out',
    help='Output image (file-type is appended) (default: %(default)s')
  group.add_argument('--npy', action=argparse.BooleanOptionalAction, default=False,
    help='Output image as npy array (default is tif')
  group.add_argument('--labels', action=argparse.BooleanOptionalAction, default=False,
    help='Labels mode (default is images')
  group = parser.add_argument_group('Image Options')
  group.add_argument('--edge', type=int, default=0,
    help='Pixel edge for blending overlap (default: %(default)s')
  group.add_argument('--mode', default='reflect',
    help='Mode to fill points outside the image during rotation (default: %(default)s')

  return parser.parse_args()

# Gather our code in a main() function
def main():
  args = parse_args()

  # Load images and validate their shape and tile position
  data = {}
  s = None
  p = re.compile(r"x(\d+)y(\d+)")
  for i in args.image:
    m = p.search(i)
    if not m:
      raise Exception(f'Unknown x and y position: {i}')
    im = np.load(i);
    if s and s != im.shape:
      raise Exception('Shape mismatch')
    else:
      s = im.shape
    if im.ndim == 2:
      # Requrie YXC
      im = im[..., np.newaxis]
    if im.ndim != 3:
      raise Exception(f'Unsupported dimensions: {im.shape}')
    x = int(m.group(1))
    y = int(m.group(2))
    d = data.get(x)
    if not d:
      data[x] = d = dict()
    d[y] = im
    
  if args.labels:
    c = compose_labels(data, rotation=args.rotation, ox=args.ox, oy=args.oy)  
  else:
    c = compose_tiles(data, rotation=args.rotation, ox=args.ox, oy=args.oy,
      edge=args.edge, mode=args.mode)
  
  # Save
  if args.npy:
    np.save(args.out + '.npy', c)
  else:
    import tifffile
    tifffile.imwrite(args.out + '.tif', c)

# Standard boilerplate to call the main()
if __name__ == '__main__':
    main()
