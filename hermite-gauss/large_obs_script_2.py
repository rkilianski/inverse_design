import meep as mp
import numpy as np
import large_obs_inverse_2 as inv
import module_hg_beam as mhg
from matplotlib import pyplot as plt, animation

#  Slice of the 3D data to plot the results it in 2D
SLICE_AXIS = 2
CHOSEN_POINT = 0

# #  Parameters of the simulation
RESOLUTION = 6
ITERATIONS = 2
T = 30

pixel_size = 1 / RESOLUTION
MULTIPLIER = 2  # must be an int !
block_size = MULTIPLIER * pixel_size
blk_data = [MULTIPLIER, block_size]
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

# #  Source  points
SRC_POS_X, SRC_POS_Y, SRC_POS_Z = -3, 0, 0
src_loc = [SRC_POS_X, SRC_POS_Y, SRC_POS_Z]

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
intensity_anim = []
lists = [intensity_anim, points_to_delete, points_for_3D_plot]

blocks_added = np.arange(ITERATIONS)
components = [mp.Ex, mp.Ey, mp.Ez]


# ***************************************** INTENSITY PATTERN **********************************************************

def fun(u, v):
    function = np.cos(4 * (u + v)) + np.cos(4 * (u - v))
    norm_fun = 1 / np.amax(function)
    return norm_fun * function


# **********************************************************************************************************************
# SIMULATION FIRST STEP - producing a dipole and obtaining parameters for the sim (meep chosen axes and obs points)
# **********************************************************************************************************************

def produce_simulation(fun, src_param_arr, multi_block_arr, ft_freq, time, obs_vol, src_pt_arr, lsts,
                       iter,
                       slc_ax,
                       adj_dt):
    global adjoint_field

    SRC_POS_X, SRC_POS_Y, SRC_POS_Z = src_pt_arr
    cell_size, source, pml, res, geo_lst = src_param_arr
    multiplier, block_size = multi_block_arr
    intensity_anim, points_to_delete, points_for_3D_plot = lsts
    iterations = iter
    slice_axis = slc_ax

    sim = mp.Simulation(
        cell_size=cell_size,
        sources=source,
        boundary_layers=pml,
        resolution=res,
        geometry=geo_lst,
        force_all_components=True,
        force_complex_fields=True
    )

    sim.run(until=time)

    x, y, z, w = sim.get_array_metadata(center=mp.Vector3(), size=obs_vol)
    [x, y, z] = [coordinate[1:-1] for coordinate in [x, y, z]]

    x_src_index, y_src_index, z_src_index = [inv.find_nearest(i, j) for i, j in
                                             zip([x, y, z], [SRC_POS_X, SRC_POS_Y, SRC_POS_Z])]
    z_obs_index = inv.find_nearest(z, CHOSEN_POINT)
    # Desired pattern for the sim to achieve
    fun_pattern = inv.install_function([x, y, z], fun)

    # Simulate a field and use its values at obs points to simulate a fictitious field - adjoint field.
    old_field = inv.get_fields(sim, obs_vol)
    old_field = (1 / (np.amax(old_field))) * old_field

    # Recording a snapshot of 2D intensity pattern for animation
    intensity_2D = inv.get_intensity(inv.get_fields(sim, obs_vol, True, slice_axis, z_obs_index))

    # Deleting grid points where blocks have been placed
    intensity_2D_blocks = inv.delete_existing(intensity_2D, points_for_3D_plot, False, multiplier)
    intensity_anim.append(intensity_2D_blocks)
    # Exciting a fictitious dipole for the adjoint field
    dipole_at_obs = inv.produce_adjoint_volume(old_field, ft_freq, adj_dt, [x, y, z])

    sim_adjoint = mp.Simulation(
        cell_size=cell_size,
        sources=dipole_at_obs,
        boundary_layers=pml,
        resolution=res,
        geometry=geo_lst,
        force_all_components=True,
        force_complex_fields=True,

    )

    sim_adjoint.run(until=time)

    adjoint_field = inv.get_fields(sim_adjoint, obs_vol)
    adjoint_field = (1 / np.amax(adjoint_field)) * adjoint_field

    delta_f = inv.df_point(old_field, adjoint_field, fun_pattern)

    ########################################################################################################################
    # SIMULATION SECOND STEP: updating geometry from starting conditions and repeating the process.
    ########################################################################################################################

    inv.exclude_points([x, y, z], [x_src_index, y_src_index, z_src_index], points_to_delete)

    x_index, y_index, z_index = inv.pick_extremum(delta_f, points_to_delete)
    [x_inclusion, y_inclusion, z_inclusion] = x[x_index], y[y_index], z[z_index]

    points_to_delete.append((x_index, y_index, z_index))
    points_for_3D_plot.append((x_index, y_index, z_index))

    geometry = inv.add_block([x_inclusion, y_inclusion, z_inclusion], block_size, geo_lst)

    for i in range(iterations):
        sim = mp.Simulation(
            cell_size=cell_size,
            sources=source,
            boundary_layers=pml,
            resolution=res,
            geometry=geometry,
            force_all_components=True,
            force_complex_fields=True
        )

        sim.run(until=time)

        old_field = inv.get_fields(sim, obs_vol)
        old_field = (1 / (np.amax(old_field))) * old_field
        # Recording a snapshot of 2D intensity pattern for animation
        intensity_2D = inv.get_intensity(inv.get_fields(sim, obs_vol, True, slice_axis, z_obs_index))

        # Deleting grid points where blocks have been placed
        intensity_2D_blocks = inv.delete_existing(intensity_2D, points_for_3D_plot, False, multiplier)
        intensity_anim.append(intensity_2D_blocks)

        sim_adjoint = mp.Simulation(
            cell_size=cell_size,
            sources=inv.produce_adjoint_volume(old_field, ft_freq, adj_dt, [x, y, z]),
            boundary_layers=pml,
            resolution=res,
            geometry=geo_lst,
            force_all_components=True,
            force_complex_fields=True,

        )

        sim_adjoint.run(until=time)

        adjoint_field = inv.get_fields(sim_adjoint, obs_vol)
        adjoint_field = (1 / (np.amax(adjoint_field))) * adjoint_field

        delta_f = inv.df_point(old_field, adjoint_field, fun_pattern)

        #  picking the coordinates corresponding to the highest change in dF and updating the geometry

        x_index, y_index, z_index = inv.pick_extremum(delta_f, points_to_delete)
        [x_inclusion, y_inclusion, z_inclusion] = x[x_index], y[y_index], z[z_index]

        points_to_delete.append((x_index, y_index, z_index))
        points_for_3D_plot.append((x_index, y_index, z_index))

        inv.add_block([x_inclusion, y_inclusion, z_inclusion], block_size, geo_lst)

    forward_2D = inv.get_fields(sim, obs_vol, True, slice_axis, z_obs_index)
    adjoint_2D = inv.get_fields(sim_adjoint, obs_vol, True, slice_axis, z_obs_index)
    df_2D = delta_f[:, :, z_obs_index]
    fun_2D = fun_pattern[:, :, z_obs_index]
    axes = [x, y, z]
    intensities = [intensity_anim]

    return axes, forward_2D, adjoint_2D, df_2D, fun_2D, intensities


# ***************************************** CREATING A BEAM ************************************************************

hg_beam = mhg.make_hg_beam(freq, WAVELENGTH, [0, sy, sz], [SRC_POS_X, SRC_POS_Y, SRC_POS_Z], dir_prop=0, waist=WAIST,
                           m=M, n=N)

src_data = [cell_3d, hg_beam, pml_layers, RESOLUTION, geom_list]

# **********************************************************************************************************************

data = produce_simulation(fun, src_data, blk_data, freq, T, OBS_VOL, src_loc, lists,
                          ITERATIONS,
                          SLICE_AXIS,
                          DT)

[x, y, z], forward_field, adjoint_field, df_2D, fun_2D, intensities = data
Ex, Ey, Ez, eps = forward_field
Ex_a, Ey_a, Ez_a, eps_a = adjoint_field

intensity_a = intensities

pattern = fun_2D
merit_function = df_2D
e_squared = inv.get_intensity(forward_field)
e_squared_adj = inv.get_intensity(adjoint_field)

larger_blocks = inv.enlarge_block(points_for_3D_plot, [x, y, z], MULTIPLIER)
grid = inv.cubify(larger_blocks, [x, y, z])


fig = plt.figure()
ax = fig.add_subplot(3, 2, 1)
ax.pcolormesh(x, y, np.transpose(np.real(merit_function)))
ax.set_title('dF')

ax = fig.add_subplot(3, 2, 2)
ax.pcolormesh(x, y, np.transpose(pattern))
ax.set_title('Desired intensity pattern.')

ax = fig.add_subplot(3, 2, 3)
ax.pcolormesh(x, y, np.transpose(np.real(e_squared_adj)))
ax.set_title('Adjoint field intensity')

ax = fig.add_subplot(3, 2, 4)
ax.pcolormesh(x, y, np.transpose(np.real(e_squared)))
ax.set_title(f'intensity')

ax = fig.add_subplot(3, 2, 5)
ax.plot(blocks_added, blocks_added)

ax.set_title('Intensity after adding a block and desired intensity')

ax = fig.add_subplot(3, 2, 6, projection='3d')
ax.set_title(f"3D structure optimizing intensity.")
ax = ax.voxels(grid, edgecolor='k')

plt.savefig(f"TEM{M}{N} at {ITERATIONS}.")
plt.show()

plt.rcParams["figure.figsize"] = [6.00, 6.00]
plt.rcParams["figure.autolayout"] = True
fig_a, ax_a = plt.subplots()

intns = ax_a.pcolormesh(x, y, np.transpose(intensity_a[0]))

fig_a.colorbar(intns)


def animate(i):
    intns.set_array(np.transpose(intensity_a[i][:, :]).flatten())
    ax_a.set_title(f"Animation of Intensity pattern")


anim = animation.FuncAnimation(fig_a, animate, interval=100, frames=ITERATIONS)
anim.save(f'Intensity animation up to {ITERATIONS} frames.gif')
plt.show()
