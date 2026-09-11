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


SCANLINE_TIMES = np.array(["2009-07-01T12:16", "2009-07-01T12:27"], dtype="datetime64[ms]")


def _make_channel(name, values, wavelength):
    """Make a channel carrying what the avhrr_l1b_gaclac reader gives every channel."""
    return xr.DataArray(
        values,
        dims=("y", "x"),
        coords={"acq_time": ("y", SCANLINE_TIMES)},
        attrs={"name": name,
               "wavelength": wavelength,
               "start_time": dt.datetime(2009, 7, 1, 12, 16),
               "end_time": dt.datetime(2009, 7, 1, 12, 27),
               "platform_name": "noaa19",
               "sensor": "avhrr-3"})


def _make_scene():
    """Make a scene shaped like one the avhrr_l1b_gaclac reader loads."""
    scene = Scene()
    scene["4"] = _make_channel("4", [[280.0, 281.0], [282.0, 283.0]], [10.3, 10.8, 11.3, "um"])
    scene["1"] = _make_channel("1", [[10.0, 11.0], [12.0, 13.0]], [0.58, 0.63, 0.68, "um"])
    return scene


class TestProcessScene(unittest.TestCase):
    """Test converting an already loaded scene to PPS level1c."""

    @staticmethod
    def _write(scene, **kwargs):
        """Convert the scene with the writer stubbed; return the file name and what reached the writer."""
        with mock.patch("level1c4pps.lac2pps_lib.save_data") as writer:
            filename = lac2pps.process_scene(scene, out_path="/out", **kwargs)
        return filename, writer.call_args.args

    def _pps_identity(self, scene, channel):
        """Return the image name and tag the channel carries when it reaches the writer."""
        _, (written_scene, _) = self._write(scene)
        attrs = written_scene[channel].attrs
        return attrs["name"], attrs.get("id_tag")

    def _pps_identity_of_added_channel(self, channel, values, wavelength):
        """Add the channel to the shared scene and return the PPS identity it reaches the writer with."""
        scene = _make_scene()
        scene[channel] = _make_channel(channel, values, wavelength)
        return self._pps_identity(scene, channel)

    def test_the_scene_is_written_under_the_pps_file_name(self):
        """The file name is what PPS uses to find a pass, so it must follow the PPS pattern."""
        expected = os.path.join("/out", "S_NWC_avhrr_noaa19_12345_20090701T1216000Z_20090701T1227000Z.nc")
        filename, written = self._write(_make_scene(), orbit_n=12345)
        self.assertEqual((filename, written[1]), (expected, expected))

    def test_channel_1_reaches_the_writer_as_the_06_micron_reflectance_image(self):
        """PPS finds a channel by its image name and tag, never by the AVHRR channel name."""
        self.assertEqual(self._pps_identity(_make_scene(), "1"), ("image1", "ch_r06"))

    def test_channel_3_of_older_platforms_reaches_the_writer_as_the_37_micron_image(self):
        """AVHRR/1 and AVHRR/2 call their 3.7 micron channel 3; PPS must still find it as image5."""
        identity = self._pps_identity_of_added_channel("3", [[290.0, 291.0], [292.0, 293.0]], [3.55, 3.74, 3.93, "um"])
        self.assertEqual(identity, ("image5", "ch_tb37"))

    def test_channel_2_reaches_the_writer_as_the_09_micron_reflectance_image(self):
        """A LAC or FRAC file must look to PPS exactly like a GAC one from the same instrument."""
        identity = self._pps_identity_of_added_channel("2", [[20.0, 21.0], [22.0, 23.0]], [0.725, 0.8625, 1.0, "um"])
        self.assertEqual(identity, ("image2", "ch_r09"))

    def test_channel_3a_reaches_the_writer_as_the_16_micron_reflectance_image(self):
        """AVHRR/3 measures 1.6 micron in channel 3a by day; PPS knows that channel as image6."""
        identity = self._pps_identity_of_added_channel("3a", [[5.0, 6.0], [7.0, 8.0]], [1.58, 1.61, 1.64, "um"])
        self.assertEqual(identity, ("image6", "ch_r16"))
