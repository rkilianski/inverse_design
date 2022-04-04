import meep as mp
import numpy as np
import inverse_design as inv
import module_hg_beam as mhg

from matplotlib import pyplot as plt, animation

#  Slice of the 3D data to plot the results it in 2D
SLICE_AXIS = 2
CHOSEN_POINT = 20

# #  Parameters of the simulation
RESOLUTION = 6
ITERATIONS = 2
T = 20

pixel_size = 1 / RESOLUTION
block_size = pixel_size
radius_squared = block_size ** 2

# #  Setting up the simulation box
DPML = 1
CELL_X, CELL_Y, CELL_Z = 6, 6, 6  # dimensions of the computational cell, not including PML
OBS_VOL = mp.Vector3(5, 5, 5)

sx, sy, sz = CELL_X + 2 * DPML, CELL_Y + 2 * DPML, CELL_Z + 2 * DPML
cell_3d = mp.Vector3(sx, sy, sz)
pml_layers = [mp.PML(DPML)]
material = mp.Medium(epsilon=1)
geom_list = []

# #  Source and observation points
SRC_POS_X, SRC_POS_Y, SRC_POS_Z = -3, 0, 0
OBS_POS_X, OBS_POS_Y, OBS_POS_Z = 0.5, -2, 0

src_loc = [SRC_POS_X, SRC_POS_Y, SRC_POS_Z]
obs_loc = [OBS_POS_X, OBS_POS_Y, OBS_POS_Z]

# #  HG beam parameters
M, N = 0, 0
WAVELENGTH = 1
WAIST = 2
DT = 5

freq = 1 / WAVELENGTH
slice_volume = mp.Volume(center=mp.Vector3(), size=OBS_VOL)  # area of fourier transform

#  Setting up various utility lists
points_to_delete = []
points_for_3D_plot = []
intensity_at_obs = []
intensity_anim = []
lists = [intensity_anim, intensity_at_obs, points_to_delete, points_for_3D_plot]

blocks_added = np.arange(ITERATIONS)
components = [mp.Ex, mp.Ey, mp.Ez]

# ***************************************** CREATING A BEAM ************************************************************

hg_beam = mhg.make_hg_beam(freq, WAVELENGTH, [0, sy, sz], [SRC_POS_X, SRC_POS_Y, SRC_POS_Z], dir_prop=0, waist=WAIST,
                           m=M, n=N)

src_data = [cell_3d, hg_beam, pml_layers, RESOLUTION, geom_list]

# **********************************************************************************************************************

data = inv.produce_simulation(src_data, block_size, freq, slice_volume, T, OBS_VOL, obs_loc, src_loc, lists, ITERATIONS,
                              SLICE_AXIS,
                              DT)

[x, y, z], forward_field, adjoint_field, forward_field_ft, adjoint_field_ft, intensities = data
Ex, Ey, Ez, eps = forward_field
Ex_a, Ey_a, Ez_a, eps_a = adjoint_field
Ex_ft, Ey_ft, Ez_ft, eps_ft = forward_field_ft
Ex_a_ft, Ey_a_ft, Ez_a_ft, eps_a_ft = adjoint_field_ft

intensity_for_anim, intensity_for_plot = intensities

merit_function = inv.df(forward_field, adjoint_field)
merit_function_ft = inv.df(forward_field_ft, adjoint_field_ft)
e_squared = inv.get_intensity(forward_field)
e_squared_ft = inv.get_intensity(forward_field_ft)


########################################################################################################################
#                                                   PLOTTING
########################################################################################################################

# Creates cubes for plotting from coordinates of inclusion points

def cubify(arr, axes):
    u, v, w = axes
    cube = False
    x_ax, y_ax, z_ax = np.indices((len(u), len(v), len(w)))
    for tup in arr:
        cube |= (x_ax == tup[0]) & (y_ax == tup[1]) & (z_ax == tup[2])
    return cube


def spherify(arr, axes, rad):
    """Loops through existing points and appends points(x0,y0,z0) which are within a specified radius rad
     from the existing point. The data should be passed on to cubify which puts voxels on the grid."""
    new_points = []
    u, v, w = axes

    for tup in arr:
        x1 = u[tup[0]]
        x2 = v[tup[1]]
        x3 = w[tup[2]]
        l, m, n = len(u), len(v), len(w)
        for i in range(l):
            for j in range(m):
                for k in range(n):
                    if (u[i] - x1) ** 2 + (v[j] - x2) ** 2 + (w[k] - x3) ** 2 < rad:
                        new_points.append((i, j, k))
    return new_points


x_obs_index = inv.find_nearest(x, OBS_POS_X)
y_obs_index = inv.find_nearest(y, OBS_POS_Y)
z_obs_index = inv.find_nearest(z, OBS_POS_Z)
observation_index = [x_obs_index, y_obs_index, z_obs_index]

grid = cubify(points_for_3D_plot, [x, y, z])


fig = plt.figure()
ax = fig.add_subplot(3, 2, 1)
ax.pcolormesh(x, y, np.transpose(np.real(merit_function)))
ax.set_title('dF')
# ax.pcolormesh(x, y, np.transpose(np.real(eps)), cmap='Greys', alpha=1)
ax.plot(x[x_obs_index], y[y_obs_index], 'ro')

ax = fig.add_subplot(3, 2, 2)
ax.pcolormesh(x, y, np.transpose(np.real(merit_function_ft)))
ax.set_title('Fourier transformed dF')
# ax.pcolormesh(x, y, np.transpose(np.real(eps_data2d_ft)), cmap='Greys', alpha=1)
ax.plot(x[x_obs_index], y[y_obs_index], 'ro')

ax = fig.add_subplot(3, 2, 3)
ax.pcolormesh(x, y, np.transpose(np.real(e_squared_ft)))
ax.set_title('Fourier transformed intensity')
# ax.pcolormesh(x, y, np.transpose(np.real(eps_data2d_ft)), cmap='Greys', alpha=1)
ax.plot(x[x_obs_index], y[y_obs_index], 'ro')

ax = fig.add_subplot(3, 2, 4)
ax.pcolormesh(x, y, np.transpose(np.real(e_squared)))
ax.set_title('intensity')
# ax.pcolormesh(x, y, np.transpose(np.real(eps_data2d)), cmap='Greys', alpha=1)
ax.plot(x[x_obs_index], y[y_obs_index], 'ro')

ax = fig.add_subplot(3, 2, 5)
ax.plot(blocks_added, intensity_at_obs)
ax.set_title('Intensity after adding a block')

ax = fig.add_subplot(3, 2, 6, projection='3d')
ax.set_title(f"3D structure optimizing intensity at the point: {OBS_POS_X, OBS_POS_Y, OBS_POS_Z}")
ax = ax.voxels(grid, edgecolor='k')

plt.savefig(f"TEM{M}{N} at {ITERATIONS}.")
plt.show()

plt.rcParams["figure.figsize"] = [6.00, 6.00]
plt.rcParams["figure.autolayout"] = True
fig_a, ax_a = plt.subplots()

intns = ax_a.pcolormesh(x, y, np.transpose(intensity_anim[0]))
ax_a.plot(x[x_obs_index], y[y_obs_index], 'ro')
fig_a.colorbar(intns)


def animate(i):
    intns.set_array(np.transpose(intensity_anim[i][:, :]).flatten())


anim = animation.FuncAnimation(fig_a, animate, interval=100, frames=ITERATIONS)
anim.save(f'Intensity animation up to {ITERATIONS} frames.gif')
plt.show()
