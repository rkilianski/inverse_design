import meep as mp
import numpy as np
import large_obs_inverse_2 as inv
import module_hg_beam as mhg
from matplotlib import pyplot as plt, animation, patches

#  Slice of the 3D data to plot the results it in 2D
SLICE_AXIS = 2
CHOSEN_POINT = 0
BEAM_FACE_AXIS = 1
AXES = [SLICE_AXIS, BEAM_FACE_AXIS]
# #  Parameters of the simulation
RESOLUTION = 8
ITERATIONS = 10
T = 20

pixel_size = 1 / RESOLUTION
MULTIPLIER = 2  # must be an int !
block_size = MULTIPLIER * pixel_size
blk_data = [MULTIPLIER, block_size]
radius_squared = block_size ** 2

# #  Setting up the simulation box
DPML = 1
CELL_X, CELL_Y, CELL_Z = 6, 6, 6  # dimensions of the computational cell, not including PML
OBS_VOL = mp.Vector3(6, 6, 6)
# for 3d plotting

sx, sy, sz = CELL_X + 2 * DPML, CELL_Y + 2 * DPML, CELL_Z + 2 * DPML
cell_3d = mp.Vector3(sx, sy, sz)
pml_layers = [mp.PML(DPML)]
material = mp.Medium(epsilon=1)
geom_list = []

# #  Source  points
SRC_POS_X, SRC_POS_Y, SRC_POS_Z = -3, 0, 0
src_loc = [SRC_POS_X, SRC_POS_Y, SRC_POS_Z]

# # Vertices of the area to be optimised ,[x0,xn,y0,z0,zn] s.t. area of (xn-x0)*(zn-z0) at level y0
flux_area = [-1.5, -0.5, -3, -0.5, 0.5]

# #  HG beam parameters
M, N = 0, 0
WAVELENGTH = 1
WAIST = 2
DT = 5

freq = 1 / WAVELENGTH
slice_volume = mp.Volume(center=mp.Vector3(), size=OBS_VOL)  # area of fourier transform


# SIMULATION FIRST STEP - producing a dipole and obtaining parameters for the sim (meep chosen axes and obs points)


def produce_simulation(src_param_arr, vertices, multi_block_arr, ft_freq, time, obs_vol, src_pt_arr, lsts,
                       iter,
                       slc_ax,
                       adj_dt):
    global adjoint_field

    src_pos_x, src_pos_y, src_pos_z = src_pt_arr
    fx0, fxn, fy0, fz0, fzn = vertices
    cell_size, source, pml, res, geo_lst = src_param_arr
    multiplier, block_size = multi_block_arr
    intensity_anim, avg_intensity, points_to_delete, points_for_3D_plot = lsts
    iterations = iter
    slice_axis, beam_face_ax = slc_ax

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
                                             zip([x, y, z], [src_pos_x, src_pos_y, src_pos_z])]
    z_obs_index = inv.find_nearest(z, CHOSEN_POINT)

    # indices of the vertices for the area of interest
    fx0i, fy0i, fz0i = [inv.find_nearest(i, j) for i, j in zip([x, y, z], [fx0, fy0, fz0])]
    fxni, fzni = [inv.find_nearest(i, j) for i, j in zip([x, z], [fxn, fzn])]
    flux_indices = [fx0i, fxni, fy0i, fz0i, fzni]
    centre_area = [fx0i + int((fxni - fx0i) / 2), fy0i, fz0i + int((fzni - fz0i) / 2)]

    # Simulate a field and use its values at obs points to simulate a fictitious field - adjoint field.
    old_field = inv.get_fields(sim, obs_vol)

    # Recording a snapshot of 2D intensity pattern for animation
    intensity_2D = inv.get_intensity(inv.get_fields(sim, obs_vol, True, slice_axis, z_obs_index))

    # Deleting grid points where blocks have been placed
    intensity_2D_blocks = inv.delete_existing(intensity_2D, points_for_3D_plot, False, multiplier)
    intensity_anim.append(intensity_2D_blocks)

    # Exciting a fictitious dipole for the adjoint field
    dipole_at_obs = inv.produce_adjoint_area(old_field, ft_freq, adj_dt, [x, y, z], flux_indices)[0]

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

    delta_f = inv.df(old_field, adjoint_field)

    # SIMULATION SECOND STEP: updating geometry from starting conditions and repeating the process.

    inv.exclude_points([x, y, z], [x_src_index, y_src_index, z_src_index], centre_area, points_to_delete)

    x_index, y_index, z_index = inv.pick_max(delta_f, points_to_delete)
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

        # Recording a snapshot of 2D intensity pattern for animation
        intensity_2D = inv.get_intensity(inv.get_fields(sim, obs_vol, True, slice_axis, z_obs_index))

        # Deleting grid points where blocks have been placed
        intensity_2D_blocks = inv.delete_existing(intensity_2D, points_for_3D_plot, False, multiplier)
        intensity_anim.append(intensity_2D_blocks)

        # Recording the average intensity at the area of interest
        avg_intensity.append(inv.intensity_avg_area([x, y, z], old_field, flux_indices))

        # Adjoint source/s
        dipole_at_obs = inv.produce_adjoint_area(old_field, ft_freq, adj_dt, [x, y, z], flux_indices)[0]

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
        # adjoint_field = (1 / (np.amax(adjoint_field))) * adjoint_field

        delta_f = inv.df(old_field, adjoint_field)

        #  picking the coordinates corresponding to the highest change in dF and updating the geometry

        x_index, y_index, z_index = inv.pick_max(delta_f, points_to_delete)
        [x_inclusion, y_inclusion, z_inclusion] = x[x_index], y[y_index], z[z_index]

        points_to_delete.append((x_index, y_index, z_index))
        points_for_3D_plot.append((x_index, y_index, z_index))

        inv.add_block([x_inclusion, y_inclusion, z_inclusion], block_size, geo_lst)

    axes = [x, y, z]
    src_ind = [(x_src_index, y_src_index, z_src_index)]
    obs_ind = inv.produce_adjoint_area(old_field, ft_freq, adj_dt, [x, y, z], flux_indices)[1]

    forward_2D = inv.get_fields(sim, obs_vol, True, slice_axis, z_obs_index)
    adjoint_2D = inv.get_fields(sim_adjoint, obs_vol, True, slice_axis, z_obs_index)
    forward_2D_beam = inv.get_fields(sim, obs_vol, True, beam_face_ax, fy0i)
    df_2D = delta_f[:, :, z_obs_index]

    intensities_list = [intensity_anim, avg_intensity]

    x_line = np.arange(x[fx0i], x[fxni], 0.02)
    x_line_z = np.full(x_line.shape, -3)
    obs_area_line = [x_line, x_line_z]
    # data for drawing a rectangle of area of interest
    rect_x = x[fxni] - x[fx0i]
    rect_z = z[fzni] - z[fz0i]
    rec_data = [x[fx0i], z[fz0i], rect_x, rect_z]  # x0,z0,len x, len z

    return axes, forward_2D, adjoint_2D, df_2D, forward_2D_beam, intensities_list, src_ind, obs_ind, obs_area_line, rec_data


# ***************************************** CREATING A BEAM ************************************************************

hg_beam = mhg.make_hg_beam(freq, WAVELENGTH, [0, sy, sz], [SRC_POS_X, SRC_POS_Y, SRC_POS_Z], dir_prop=0, waist=WAIST,
                           m=M, n=N)

src_data = [cell_3d, hg_beam, pml_layers, RESOLUTION, geom_list]

# **********************************************************************************************************************
#  Setting up various utility lists
points_to_delete = []
points_for_3D_plot = []
intensity_anim = []
intensity_avg = []
lists = [intensity_anim, intensity_avg, points_to_delete, points_for_3D_plot]

########################################################################################################################


blocks_added = np.arange(ITERATIONS)

data = produce_simulation(src_data, flux_area, blk_data, freq, T, OBS_VOL, src_loc, lists,
                          ITERATIONS,
                          AXES,
                          DT)

[x, y, z], forward_field, adjoint_field, df_2D, beam_face, intensities, source, area, red_line, rect = data

X0, Z0, X_LENGTH, Z_LENGTH = rect
Ex, Ey, Ez, eps = forward_field
Ex_a, Ey_a, Ez_a, eps_a = adjoint_field
intensity_a, intensity_averages = intensities

merit_function = df_2D
e_squared = inv.get_intensity(forward_field)
e_squared_beam = inv.get_intensity(beam_face)
e_squared_adj = inv.get_intensity(adjoint_field)

larger_blocks = inv.enlarge_block(points_for_3D_plot, [x, y, z], MULTIPLIER)

grid_1 = inv.cubify(source, [x, y, z])
grid_2 = inv.cubify(area, [x, y, z])
grid_3 = inv.cubify(larger_blocks, [x, y, z])

voxel_array = grid_1 | grid_2 | grid_3
colors = np.empty(voxel_array.shape, dtype=object)
colors[grid_1] = 'y'
colors[grid_2] = 'r'
colors[grid_3] = 'b'

fig = plt.figure()
ax = fig.add_subplot(3, 2, 1)
ax.pcolormesh(x, y, np.transpose(np.real(merit_function)))
ax.set_title('dF')

ax = fig.add_subplot(3, 2, 2)
ax.pcolormesh(x, y, np.transpose(e_squared))
ax.plot(red_line[0], red_line[1], 'ro')
ax.set_title('Intensity.')

ax = fig.add_subplot(3, 2, 3)
ax.pcolormesh(x, y, np.transpose(np.real(e_squared_adj)))
ax.set_title('Adjoint field intensity')

ax = fig.add_subplot(3, 2, 4)
ax.pcolormesh(x, y, np.transpose(intensity_a[-1]))
ax.plot(red_line[0], red_line[1], 'ro')
ax.set_title(f'Intensity and the shadow of a structure, slicing by z-axis ')

ax = fig.add_subplot(3, 2, 5)
ax.plot(blocks_added, intensity_averages)
ax.set_title(f'Intensity average. Improvement of {round(intensity_averages[-1] / intensity_averages[0], 4)}.')

ax = fig.add_subplot(3, 2, 6)
ax.pcolormesh(x, y, np.transpose(e_squared_beam))
ax.set_title('Intensity at the optimised wall.')
# Drawing the rectangle
rect = patches.Rectangle((X0, Z0), X_LENGTH, Z_LENGTH, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)
# ax = fig.add_subplot(3, 2, 6, projection='3d')
# ax.set_title(f"The 3D structure optimizing intensity.")
# ax = ax.voxels(voxel_array, facecolors=colors, edgecolor='k')

plt.savefig(f"TEM{M}{N} at {ITERATIONS}.")
plt.show()

fig2 = plt.figure()
ax2 = fig2.add_subplot(projection='3d')
ax2.set_title(
    f"The 3D structure optimizing intensity, between x:({flux_area[0]},{flux_area[1]}), "
    f"z:({flux_area[3]},{flux_area[4]} )"
    f"at y:{flux_area[2]}.")
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2 = ax2.voxels(voxel_array, facecolors=colors, edgecolor='k')

plt.show()

# plt.rcParams["figure.figsize"] = [6.00, 6.00]
# plt.rcParams["figure.autolayout"] = True
# fig_a, ax_a = plt.subplots()
#
# intns = ax_a.pcolormesh(x, y, np.transpose(intensity_a[0]))
#
# fig_a.colorbar(intns)
#
#
# def animate(i):
#     intns.set_array(np.transpose(intensity_a[i][:, :]).flatten())
#     ax_a.set_title(f"Animation of Intensity pattern")
#
#
# anim = animation.FuncAnimation(fig_a, animate, interval=100, frames=ITERATIONS)
# anim.save(f'Intensity animation up to {ITERATIONS} frames.gif')
# plt.show()
