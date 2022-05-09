import pickle
import matplotlib.pyplot as plt
import numpy as np

from inv_area_intns import cubify, spherify

with open("structure_at_400", "rb") as fp:  # Unpickling
    points = pickle.load(fp)

with open("axes_at_400", "rb") as fa:  # Unpickling
    axes = pickle.load(fa)


def scale_points(arr, factor, shift=0):
    new_arr = []
    for tup in arr:
        new_arr.append((tup[0] * factor - shift, tup[1] * factor + 10, tup[2] * factor - shift))

    return new_arr


new_points = scale_points(points, 2, 10)
[x, y, z] = axes
R = 0.02
xn = np.linspace(x[0], x[-1], 2 * len(x))
yn = np.linspace(x[0], x[-1], 2 * len(x))
zn = np.linspace(x[0], x[-1], 2 * len(x))

spheres = spherify(new_points, [xn, yn, zn], R)
voxels = cubify(spheres, [xn, yn, zn])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set(xlabel='x', ylabel='y', zlabel='z')
ax.set_title('Structure'
             )
ax = ax.voxels(voxels, facecolors='cornflowerblue', alpha=0.8)


plt.show()
