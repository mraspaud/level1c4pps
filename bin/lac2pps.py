#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 level1c4pps developers
#
# This file is part of level1c4pps
#
# level1c4pps is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# level1c4pps is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with level1c4pps.  If not, see <http://www.gnu.org/licenses/>.

"""Script to make AVHRR LAC and FRAC level1c in PPS-format with pytroll."""

import argparse

from level1c4pps.lac2pps_lib import process_one_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make a PPS level1c file from an AVHRR level 1b file.")
    parser.add_argument('file', type=str, help='AVHRR level 1b file to process')
    parser.add_argument('-o', '--out_dir', type=str, default='.', help="Output directory where to store level1c file.")
    parser.add_argument('-td', '--tle_dir', type=str, default='.', help="TLE directory.")
    parser.add_argument('-tn', '--tle_name', type=str, default='.', help="TLE file name.")
    options = parser.parse_args()
    process_one_file(options.file, options.out_dir,
                     reader_kwargs={'tle_dir': options.tle_dir, 'tle_name': options.tle_name})
