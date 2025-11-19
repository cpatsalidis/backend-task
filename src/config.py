"""
Configuration settings for facial region processing.
"""

import numpy as np

# Facial region definitions with landmark indices, colors, and labels
FACIAL_REGIONS = {
    'forehead': {
        'indices': [139, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 368, 301, 293, 334, 296, 336, 8, 107, 66, 105, 63],
        'color': (139, 0, 81),  # BGR format
        'label': '1',
        'name': 'Forehead'
    },
    'left_under_eye': {
        'indices': [226, 110, 24, 23, 22, 26, 244, 245, 188, 174, 47, 117, 35],
        'color': (139, 0, 81),
        'label': '2',
        'name': 'Left Under-Eye'
    },
    'right_under_eye': {
        'indices': [453, 452, 451, 450, 261, 265, 346, 347, 329, 437, 277, 343, 465],
        'color': (139, 0, 81),
        'label': '3',
        'name': 'Right Under-Eye'
    },
    'lower_face': {
        'indices': [93, 234, 127, 34, 116, 100, 195, 236, 126, 142, 203, 167, 164, 393, 426, 399, 371, 345, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132],
        'color': (139, 0, 81),
        'label': '4',
        'name': 'Lower Face'
    },
    'nose': {
        'indices': [55, 168, 285, 417, 412, 399, 363, 420, 279, 429, 327, 358, 326, 2, 97, 98, 129, 49, 198, 236, 196, 122, 193],
        'color': (131, 54, 131),
        'label': '5',
        'name': 'Nose'
    },
    'left_ear': {
        'indices': [127, 234, 93, 177, 215],  # Left side landmarks for ear boundary
        'color': (139, 0, 81),
        'label': '7',
        'name': 'Left Ear'
    },
    'right_ear': {
        'indices': [356, 454, 323, 401, 435],  # Right side landmarks for ear boundary
        'color': (139, 0, 81),
        'label': '6',
        'name': 'Right Ear'
    }
}

# Processing parameters
ALPHA_BLEND = 0.25  # Transparency for region overlay (lower = more transparent)
CONTOUR_COLOR = (58, 36, 59)  # BGR format

# Mask smoothing parameters
MORPH_KERNEL_SIZE = (57, 57)
GAUSSIAN_KERNEL_SIZE = (41, 41)
GAUSSIAN_SIGMA = 9

# Contour drawing parameters
DASH_LENGTH = 3
GAP_LENGTH = 3
CONTOUR_THICKNESS = 1

# Extension parameters
EAR_OFFSET = 1  # Pixel gap between face boundary and ear region
CROP_PADDING_FACTOR = 0.03  # Padding factor for final crop

# Landmark indices for special operations
EYEBROW_LANDMARKS = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
LIP_LANDMARKS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375]
LEFT_SIDE_LANDMARKS = [127, 234, 93, 177, 215]
RIGHT_SIDE_LANDMARKS = [356, 454, 323, 401, 435]

# File paths (relative to project root)
INPUT_ORIGINAL_IMAGE = '../Backend_Engineer/original_image.png'
INPUT_SEGMENTATION_MAP = '../Backend_Engineer/segmentation_map.png'
INPUT_LANDMARKS_FILE = '../Backend_Engineer/landmarks.txt'
OUTPUT_RESULT_IMAGE = '../Backend_Engineer/result_final.png'

