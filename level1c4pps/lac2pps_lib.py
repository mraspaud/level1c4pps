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

"""Convert AVHRR LAC and FRAC data to PPS level-1c format."""

from level1c4pps import compose_filename, save_data, set_header_and_band_attrs_defaults

PPS_TAGS = {"1": "ch_r06"}
ONE_IR_CHANNEL = "4"


def process_scene(scene, out_path=".", orbit_n=0):
    """Convert an already loaded AVHRR scene in place and write it as PPS level1c."""
    ir_channel = scene[ONE_IR_CHANNEL]
    set_header_and_band_attrs_defaults(scene, PPS_TAGS, ir_channel, orbit_n=orbit_n)
    filename = compose_filename(scene, out_path, instrument="avhrr", band=ir_channel)
    save_data(scene, filename, header_attrs=None, engine=None)
    return filename
