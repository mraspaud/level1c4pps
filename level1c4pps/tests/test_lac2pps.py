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

"""Unit tests for the lac2pps_lib module."""

import datetime as dt
import os
import unittest
from unittest import mock

import numpy as np
import xarray as xr
from satpy import Scene

import level1c4pps.lac2pps_lib as lac2pps


def _make_scene():
    """Make a scene shaped like one the avhrr_l1b_gaclac reader loads."""
    acq_time = np.array(["2009-07-01T12:16", "2009-07-01T12:27"], dtype="datetime64[ms]")
    scene = Scene()
    scene["4"] = xr.DataArray(
        [[280.0, 281.0], [282.0, 283.0]],
        dims=("y", "x"),
        coords={"acq_time": ("y", acq_time)},
        attrs={"name": "4",
               "wavelength": [10.3, 10.8, 11.3, "um"],
               "start_time": dt.datetime(2009, 7, 1, 12, 16),
               "end_time": dt.datetime(2009, 7, 1, 12, 27),
               "platform_name": "noaa19",
               "sensor": "avhrr-3"})
    return scene


class TestProcessScene(unittest.TestCase):
    """Test converting an already loaded scene to PPS level1c."""

    def test_the_scene_is_written_under_the_pps_file_name(self):
        """The file name is what PPS uses to find a pass, so it must follow the PPS pattern."""
        expected = os.path.join("/out", "S_NWC_avhrr_noaa19_12345_20090701T1216000Z_20090701T1227000Z.nc")
        with mock.patch("level1c4pps.lac2pps_lib.save_data") as writer:
            filename = lac2pps.process_scene(_make_scene(), out_path="/out", orbit_n=12345)
        self.assertEqual((filename, writer.call_args.args[1]), (expected, expected))
