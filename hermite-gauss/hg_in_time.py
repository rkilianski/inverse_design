import meep as mp
import numpy as np
import inverse_design_no_dft as inv
import module_hg_beam as mhg

from matplotlib import pyplot as plt, animation

#  Slice of the 3D data to plot the results it in 2D
SLICE_AXIS = 2
CHOSEN_POINT = 20

# #  Parameters of the simulation
RESOLUTION = 6
ITERATIONS = 1
T = 10

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

e_squared_over_time = []
e_squared_over_time_ft = []

for j in range(0, T):
    data = inv.produce_simulation(src_data, block_size, freq, slice_volume, j, OBS_VOL, obs_loc, src_loc, lists,
                                  ITERATIONS,
                                  SLICE_AXIS,
                                  DT)

    [x, y, z], forward_field, adjoint_field, forward_field_ft, adjoint_field_ft, intensities = data

    x_obs_index = inv.find_nearest(x, OBS_POS_X)
    y_obs_index = inv.find_nearest(y, OBS_POS_Y)
    z_obs_index = inv.find_nearest(z, OBS_POS_Z)

    Ex, Ey, Ez, eps = forward_field
    Ex_a, Ey_a, Ez_a, eps_a = adjoint_field
    Ex_ft, Ey_ft, Ez_ft, eps_ft = forward_field_ft
    Ex_a_ft, Ey_a_ft, Ez_a_ft, eps_a_ft = adjoint_field_ft

    e_squared_over_time.append(inv.get_intensity(forward_field))
    e_squared_over_time_ft.append(inv.get_intensity(forward_field_ft))

    merit_function = inv.df(forward_field, adjoint_field)
    merit_function_ft = inv.df(forward_field_ft, adjoint_field_ft)
    e_squared = inv.get_intensity(forward_field)
    e_squared_ft = inv.get_intensity(forward_field_ft)

########################################################################################################################
#                                                   ANIMATION
########################################################################################################################
#
x_obs_index = inv.find_nearest(x, OBS_POS_X)
y_obs_index = inv.find_nearest(y, OBS_POS_Y)
z_obs_index = inv.find_nearest(z, OBS_POS_Z)

t = np.arange(T)

observation_index = [x_obs_index, y_obs_index, z_obs_index]

plt.rcParams["figure.figsize"] = [6.00, 6.00]
plt.rcParams["figure.autolayout"] = True
fig_a, ax_a = plt.subplots(1, 1)

intns = ax_a.pcolormesh(x, y, np.transpose(e_squared_over_time[0]), vmin=0, vmax=0.5)

ax_a.plot(x[x_obs_index], y[y_obs_index], 'ro')


def animate(i):
    intns.set_array(np.transpose(e_squared_over_time[i][:, :]).flatten())


anim = animation.FuncAnimation(fig_a, animate, interval=100, frames=T)
anim.save(f'Intensity animation up to {ITERATIONS} frames.gif')
plt.show()

plt.plot(t, intensity_at_obs)
plt.show()
