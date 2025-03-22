#!/usr/local/bin/python3
# Copyright (c) 2021 Robert Bosch GmbH Copyright holder of the paper "DiagViB-6: A Diagnostic Benchmark Suite for Vision Models in the Presence of Shortcut and Generalization Opportunities" accepted at ICCV 2021.
# All rights reserved.
###
# The paper "DiagViB-6: A Diagnostic Benchmark Suite for Vision Models in the Presence of Shortcut and Generalization Opportunities" accepted at ICCV 2021.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Author: Elias Eulig, Volker Fischer
# -*- coding: utf-8 -*-

import numpy as np
import colorsys

from .config import *

__all__ = ['sample_attribute',
           'get_mt_labels']

# Draw a random factor-class instance from a given factor-class label.
# =============================================================================

def sample_attribute(name, semantic_attr, **kwargs):
    """Calls dinamically the function get_<name>."""
    get_fn = globals()['get_' + name]
    return get_fn(semantic_attr, **kwargs)


def get_position(semantic_attr):
    """Generates a range of positions by within the limits defined for this attribute in config.POSITION."""
    return np.random.uniform(*POSITION[semantic_attr][0]), np.random.uniform(*POSITION[semantic_attr][1])


def get_scale(semantic_attr):
    """Generates a random scale factor within the range defined for this attribute in config.SCALE."""
    return np.random.uniform(*SCALE[semantic_attr])


def get_colorgrad(hue_attr, light_attr):
    """
    Generates a gradient defined by two RGB colors based on specified hue and lightness attributes.
    
    Args:
        hue_attr (str): Specifies the hue of the colors. Acceptable values in config.HUES.keys().
        light_attr (str): Determines the lightness range of the colors. Acceptable values in config.LIGHTNESS.keys().

    Returns:
        tuple: A pair of RGB colors represented as tuples, each with integer values in the range [0, 255].
    """

    # Select two random lightness values from the specified range.
    l1 = np.random.uniform(*LIGHTNESS[light_attr][0])
    l2 = np.random.uniform(*LIGHTNESS[light_attr][1])
    if hue_attr == 'gray': # If 'hue' is grey, then no saturation (i.e. grayscale) is used.
        col1, col2 = (0., l1, 0.), (0., l2, 0.)
        col1, col2 = colorsys.hls_to_rgb(*col1), colorsys.hls_to_rgb(*col2)
        return tuple((int(x*255.) for x in col1)), tuple((int(x*255.) for x in col2))
    
    hue = np.random.uniform(*HUES[hue_attr])
    if hue < 1.:
        hue += 360.
    col1, col2 = (hue / 360., l1, 1.0), (hue / 360., l2, 1.0)
    col1, col2 = colorsys.hls_to_rgb(*col1), colorsys.hls_to_rgb(*col2)
    return tuple((int(x * 255.) for x in col1)), tuple((int(x * 255.) for x in col2))


def get_mt_labels(task_label, OBJECT_ATTRIBUTES=OBJECT_ATTRIBUTES):
    """
    Retrieves the index of a specified value within a given factor from config.OBJECT_ATTRIBUTES.
    
    Args:
        task_label (tuple): A tuple containing two elements. The first element is the factor, and the second element is the specific value for which the index needs to be found.

    Returns:
        (int): The index of the specified value within its factor.
    """
    return np.argmax([cls == task_label[1] for cls in OBJECT_ATTRIBUTES[task_label[0]]])