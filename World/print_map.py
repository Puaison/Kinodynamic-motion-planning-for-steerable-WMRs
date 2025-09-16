import sys
import os
import pickle

# Get the directory containing the current file
project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("Current folder:", project_folder)

sys.path.append(project_folder)

from World.occupancyGridMap import OccupancyGridMap
from Utils.utils import * 
from Utils.constants import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
import math
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D

# RESOLUTION = 1
# WIDTH_ROBOT_FIGURE = 0.741
# HEIGHT_ROBOT_FIGURE = 0.590
# WIDTH_WHEELS = 0.003
# HEIGHT_WHEELS = 0.18
# JOINT_POSITION_DISPLACMENT_X = 0.024
# JOINT_POSITION_DISPLACMENT_Y = 0.019
# WHEELS_AX_OFFSET = 0.045

# WIDTH_ROBOT_FIGURE = 5 / RESOLUTION
# HEIGHT_ROBOT_FIGURE = 5 / RESOLUTION
# WIDTH_WHEELS = 2 / RESOLUTION
# HEIGHT_WHEELS = 0.5 / RESOLUTION
# JOINT_POSITION_DISPLACMENT_X = 1 / RESOLUTION
# JOINT_POSITION_DISPLACMENT_Y = 1 / RESOLUTION
# WHEELS_AX_OFFSET = 1 / RESOLUTION
    
def initMap():
    # Example usage with visualization:
    heigth = MAP_HEIGHT
    width = MAP_WIDTH
    inflate = INFLATE  # Set to True to inflate obstacles
    inflation_radius = INFLATION_RADIUS
    obstacle_probability = (
        OBSTACLE_PROBABILITY  # Probability of each cell being an obstacle (before inflating)
    )

    grid_map = OccupancyGridMap(
        width, heigth, OBSTACLE_POSITION, inflate, inflation_radius, obstacle_probability, resolution = RESOLUTION, seed = SEED
    )

    return grid_map


grid_map = initMap()
occupancy_grid = grid_map.occupancy_grid

grid_map.save_map()