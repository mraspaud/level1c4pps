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

import xarray as xr
from satpy import Scene

from level1c4pps import (compose_filename, convert_angles, rename_latitude_longitude, save_data,
                         set_header_and_band_attrs_defaults, update_angle_attributes)

PPS_TAGS = {"1": "ch_r06",
            "2": "ch_r09",
            "3a": "ch_r16",
            "3": "ch_tb37",
            "3b": "ch_tb37",
            "4": "ch_tb11",
            "5": "ch_tb12"}
ONE_IR_CHANNEL = "4"


def label_quality_flags(scene):
    """Give the pygac quality flags the tag and name PPS reads them by."""
    scene["qual_flags"].attrs.update(id_tag="qual_flags", long_name="pygac quality flags")


def process_scene(scene, out_path=".", orbit_n=0):
    """Convert an already loaded AVHRR scene in place and write it as PPS level1c."""
    ir_channel = scene[ONE_IR_CHANNEL]
    scanline_timestamps = xr.DataArray(ir_channel.coords["acq_time"].values, dims=["y"])
    set_header_and_band_attrs_defaults(scene, PPS_TAGS, ir_channel, orbit_n=orbit_n)
    rename_latitude_longitude(scene)
    convert_angles(scene)
    update_angle_attributes(scene, ir_channel)
    scene["scanline_timestamps"] = scanline_timestamps
    label_quality_flags(scene)
    filename = compose_filename(scene, out_path, instrument="avhrr", band=ir_channel)
    save_data(scene, filename, header_attrs={"source": "lac2pps.py"}, engine=None)
    return filename


def process_one_file(level1b_file, out_path=".", reader_kwargs=None):
    """Read an AVHRR level 1b file and write it as PPS level1c."""
    scene = Scene(reader="avhrr_l1b_gaclac", filenames=[level1b_file], reader_kwargs=reader_kwargs)
    scene.load(["1", "2", "3", "4", "latitude", "longitude", "qual_flags", "solar_zenith_angle",
                "sensor_zenith_angle", "sun_sensor_azimuth_difference_angle"])
    return process_scene(scene, out_path=out_path)
