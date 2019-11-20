#/usr/bin/env python
# -*-coding:utf-8 -*-

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from pylab import interactive
from matplotlib.patches import FancyArrowPatch
from scipy.optimize import fmin as simplex
from numpy import linalg as LA
import math
interactive(True)
