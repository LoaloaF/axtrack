from matplotlib.patches import Polygon
from matplotlib import pyplot as plt
import numpy as np


fig, ax = plt.subplots(figsize=(8,4))
ax.set_ylim(-3,12)
ax.set_xlim(-3,12.5)
ax.set_yticks(np.arange(1,12))


def _plus3D(xy_center, x_diameter, y_diameter, linewidth, thickness, facecolor='w', edgecolor='k'):
    top, halftop, halfbottom, bottom = (-y_diameter/2, -y_diameter*thickness, y_diameter*thickness, y_diameter/2)
    left, halfleft, halfright, right = (-x_diameter/2, -x_diameter*thickness, x_diameter*thickness, x_diameter/2)
    x = (left, halfleft, halfleft, halfright, halfright, right, right, halfright, halfright, halfleft, halfleft, left)
    y = (halftop, halftop, top, top, halftop, halftop, halfbottom, halfbottom, bottom, bottom, halfbottom, halfbottom)

    coordinates = np.array((x,y)).T + xy_center
    p = Polygon(coordinates, linewidth=linewidth, facecolor=facecolor, 
                edgecolor=edgecolor)
    return p

def _minus3D(xy_center, x_diameter, y_diameter, linewidth, thickness, facecolor='w', edgecolor='k'):
    top, halftop, halfbottom, bottom = (-y_diameter/2, -y_diameter*thickness, y_diameter*thickness, y_diameter/2)
    left, halfleft, halfright, right = (-x_diameter/2, -x_diameter*thickness, x_diameter*thickness, x_diameter/2)
    x = (left, right, right, left)
    y = (top, top, bottom, bottom)

    coordinates = np.array((x,y)).T + xy_center
    p = Polygon(coordinates, linewidth=linewidth, facecolor=facecolor, 
                edgecolor=edgecolor)
    return p

p = minus3D(xy_center=(2,3), y_diameter=1, x_diameter=2, linewidth=2, thickness=.1)
ax.add_patch(p)
p = plus3D(xy_center=(4,8), y_diameter=3, x_diameter=2, linewidth=2, thickness=.1)
ax.add_patch(p)
plt.show()




    