import sys
sys.path.append("../../src")

from util import *
from surface import *

import matplotlib.pyplot as plt
import numpy as np

surf = Surface(origin=(0, 0), dims=(500, 500), fs=10, overlap=0)

gap = 1000

hs = repeating_gaussian(surf.x, 130, 1000/2.5, gap)
surf.arr_along_axis(hs, axis=0)
surf.show_2d_heatmap()

del surf

surf = Surface(origin=(0, 0), dims=(500, 500), fs=10, overlap=0)

gap = 1000

hs = repeating_slant_gaussian(surf.x, 130, 1000/2.5, 250/2.5, gap)
surf.arr_along_axis(hs, axis=0)
surf.show_2d_heatmap()
