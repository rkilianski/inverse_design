import meep as mp
import numpy as np


# **********************************************************************************************************************
# UTILITY FUNCTIONS
# **********************************************************************************************************************
def get_intensity(arr):
    """Returns intensity I for all x and y. """
    e_sq = 0
    fields = arr[: 3]
    for i in fields:
        e_sq += np.real(i * np.conjugate(i))

    return e_sq


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def get_fields(simulation, obs_vol, fields_2D=False, *slice_axis_and_which_point):
    """If boolean fields_2D is True(default is False), slice axis and point need to be specified."""
    fields_data = [simulation.get_array(center=mp.Vector3(), size=obs_vol, component=field) for field in
                   [mp.Ex, mp.Ey, mp.Ez, mp.Dielectric]]
    fields_data_elements = [element[1:-1, 1:-1, 1:-1] for element in fields_data]
    if fields_2D:
        slice_axis, which_point = slice_axis_and_which_point
        fields_data_elements = [[a[which_point, :, :], a[:, which_point, :], a[:, :, which_point]][slice_axis]
                                for a
                                in fields_data_elements]
    # ex,ey,ez,epsilon
    return fields_data_elements


def exclude_points(arr_axes, arr_src, arr_obs, points_list):
    # points_list = points_to_delete
    x, y, z = arr_axes
    x_src_i, y_src_i, z_src_i = arr_src
    x_obs_i, y_obs_i, z_obs_i = arr_obs

    def _delta(l, m, n):
        return (l - m[n]) ** 2

    for x_i in x:
        for y_i in y:
            for z_i in z:
                src_dist = _delta(x_i, x, x_src_i) + _delta(y_i, y, y_src_i) + _delta(z_i, z, z_src_i)
                obs_dist = _delta(x_i, x, x_obs_i) + _delta(y_i, y, y_obs_i) + _delta(z_i, z, z_obs_i)

                if src_dist < 1 or obs_dist < 1:
                    x_index = np.where(x == x_i)[0][0]
                    y_index = np.where(y == y_i)[0][0]
                    z_index = np.where(z == z_i)[0][0]

                    points_list.append((x_index, y_index, z_index))


# Calculate the derivative of a merit function dF

def df(old_field_arr, adj_field_arr):
    e1, e2, e3, eps1 = old_field_arr
    a1, a2, a3, eps2 = adj_field_arr
    d_func = np.real((a1 * e1 + a2 * e2 + a3 * e3))
    return d_func


def df_point(old_field_arr, adj_field_arr, value):
    e1, e2, e3, eps1 = old_field_arr
    a1, a2, a3, eps2 = adj_field_arr
    intensity = get_intensity(old_field_arr)
    d_func = -np.real((a1 * e1 + a2 * e2 + a3 * e3)) / (intensity - value) ** 2
    return d_func


def delete_existing(arr, lst, dim_3D=True):
    # lst = points_to_delete
    # 2D working only when 2d is x and y - rewrite!
    if dim_3D:
        for tup in lst:
            arr[tup[0], tup[1], tup[2]] = 0
    else:
        for tup in lst:
            arr[tup[0], tup[1]] = 0
    return arr


def pick_extremum(delta, lst, lst_3D, arr_coords, maximum=True):
    # lst = points_to_delete
    # lst_3D = points_for_3D_plot
    """Returns a tuple of points (x,y,z) corresponding to the highest value of the dF.
    Also updates the list of excluded points and the list of points for the 3D plot.
    Also internally """
    x, y, z = arr_coords
    if len(lst) > 0:
        delta = delete_existing(delta, lst)
    if maximum:
        extr_x, extr_y, extr_z = np.unravel_index(delta.argmax(), delta.shape)
    else:
        extr_x, extr_y, extr_z = np.unravel_index(delta.argmin(), delta.shape)

    lst.append((extr_x, extr_y, extr_z))
    lst_3D.append((extr_x, extr_y, extr_z))

    return x[extr_x], y[extr_y], z[extr_z]


def intensity_at_point(field, x_index, y_index, z_index):
    ex, ey, ez, eps = field
    intensity = np.real((ex * np.conjugate(ex) + ey * np.conjugate(ey) + ez * np.conjugate(ez)))
    intensity_at_x0 = intensity[x_index, y_index, z_index]
    return intensity_at_x0


def produce_adjoint_field(forward_field, freq, dt, arr_coord, arr_obs_pts):
    """Takes in an array of components of the simulated forward field,
     an array of axes i.e. [x,y,z] and an array of indices of  observation points [i0,j0,k0],
     where e.g. x[i0] = x0."""
    source_at_obs = []
    x_ax, y_ax, z_ax = arr_coord
    i0, j0, k0 = arr_obs_pts
    for element in range(3):
        source_at_obs.append(mp.Source(
            mp.ContinuousSource(freq, width=dt, is_integrated=True),
            component=[mp.Ex, mp.Ey, mp.Ez][element],
            size=mp.Vector3(),
            center=mp.Vector3(x_ax[i0], y_ax[j0], z_ax[k0]),
            amplitude=np.conjugate(forward_field[element][i0, j0, k0])))
    return source_at_obs


def add_block(arr_centre, block_size, geo_lst):
    # geo_lst = geom_list
    x1, x2, x3 = arr_centre

    block = mp.Block(
        center=mp.Vector3(x1, x2, x3),
        size=mp.Vector3(block_size, block_size, block_size), material=mp.Medium(epsilon=1.3))
    # sphere = mp.Sphere(block_size, center=mp.Vector3(x1, x2, x3), material=mp.Medium(epsilon=1.3))
    geo_lst.append(block)
    return geo_lst


# **********************************************************************************************************************
# SIMULATION FIRST STEP - producing a dipole and obtaining parameters for the sim (meep chosen axes and obs points)
# **********************************************************************************************************************

def produce_simulation(src_param_arr, blk, ft_freq, ft_vol, time, obs_vol, obs_pt_arr, src_pt_arr, lsts, iter, slc_ax,
                       adj_dt):
    intensity_over_time = []

    global adjoint_field
    cell_size, source, pml, res, geo_lst = src_param_arr
    block_size = blk
    OBS_POS_X, OBS_POS_Y, OBS_POS_Z = obs_pt_arr
    SRC_POS_X, SRC_POS_Y, SRC_POS_Z = src_pt_arr
    components = [mp.Ex, mp.Ey, mp.Ez, mp.Dielectric]
    intensity_anim, intensity_at_obs, points_to_delete, points_for_3D_plot = lsts
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

    # dft_obj = sim.add_dft_fields(components, ft_freq, ft_freq, 1, where=ft_vol)
    sim.run(until=time)

    x, y, z, w = sim.get_array_metadata(center=mp.Vector3(), size=obs_vol)
    [x, y, z] = [coordinate[1:-1] for coordinate in [x, y, z]]

    x_obs_index, y_obs_index, z_obs_index = [find_nearest(i, j) for i, j in
                                             zip([x, y, z], [OBS_POS_X, OBS_POS_Y, OBS_POS_Z])]
    x_src_index, y_src_index, z_src_index = [find_nearest(i, j) for i, j in
                                             zip([x, y, z], [SRC_POS_X, SRC_POS_Y, SRC_POS_Z])]

    # Simulate a field and use its values at obs points to simulate a fictitious field - adjoint field.
    old_field = get_fields(sim, obs_vol)

    # Recording a snapshot of 2D intensity pattern for animation
    intensity_2D = get_intensity(get_fields(sim, obs_vol, True, slice_axis, z_obs_index))

    # Deleting grid points where blocks have been placed
    intensity_2D_blocks = delete_existing(intensity_2D, points_for_3D_plot, False)
    intensity_anim.append(intensity_2D_blocks)

    # old_field_dft = get_fields(sim, obs_vol)

    dipole_at_obs = produce_adjoint_field(old_field, ft_freq, adj_dt, [x, y, z],
                                          [x_obs_index, y_obs_index, z_obs_index])

    sim_adjoint = mp.Simulation(
        cell_size=cell_size,
        sources=dipole_at_obs,
        boundary_layers=pml,
        resolution=res,
        geometry=geo_lst,
        force_all_components=True,
        force_complex_fields=True,

    )

    # dft_adjoint_obj = sim_adjoint.add_dft_fields(components, ft_freq, 0, 1, where=ft_vol)
    sim_adjoint.run(until=time)

    adjoint_field = get_fields(sim_adjoint, obs_vol)

    delta_f = df(old_field, adjoint_field)
    delta_f[x_obs_index:, :, :] = np.zeros((len(x) - x_obs_index, len(y), len(z)))

    ########################################################################################################################
    # SIMULATION SECOND STEP: updating geometry from starting conditions and repeating the process.
    ########################################################################################################################

    exclude_points([x, y, z], [x_src_index, y_src_index, z_src_index], [x_obs_index, y_obs_index, z_obs_index],
                   points_to_delete)
    # lst = points_to_delete
    # lst_3D = points_for_3D_plot
    x_inclusion, y_inclusion, z_inclusion = pick_extremum(delta_f, points_to_delete, points_for_3D_plot, [x, y, z])
    geometry = add_block([x_inclusion, y_inclusion, z_inclusion], block_size, geo_lst)

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

        dft_obj = sim.add_dft_fields(components, ft_freq, 0, 1, where=ft_vol)
        sim.run(until=time)

        old_field = get_fields(sim, obs_vol)
        # Recording a snapshot of 2D intensity pattern for animation
        intensity_2D = get_intensity(get_fields(sim, obs_vol, True, slice_axis, z_obs_index))

        # Deleting grid points where blocks have been placed
        intensity_2D_blocks = delete_existing(intensity_2D, points_for_3D_plot, False)
        intensity_anim.append(intensity_2D_blocks)

        # old_field_dft = get_fields(sim, obs_vol)

        # recording intensity at observation point after adding a block in a previous turn
        intensity_at_obs.append(intensity_at_point(old_field, x_obs_index, y_obs_index, z_obs_index))
        # recording intensity over a time interval

        sim_adjoint = mp.Simulation(
            cell_size=cell_size,
            sources=produce_adjoint_field(old_field, ft_freq, adj_dt, [x, y, z],
                                          [x_obs_index, y_obs_index, z_obs_index]),
            boundary_layers=pml,
            resolution=res,
            geometry=geo_lst,
            force_all_components=True,
            force_complex_fields=True,

        )

        # dft_obj_adjoint = sim_adjoint.add_dft_fields(components, ft_freq, 0, 1, where=ft_vol)
        sim_adjoint.run(until=time)

        adjoint_field = get_fields(sim_adjoint, obs_vol)
        # adjoint_field_dft = get_fields(sim_adjoint, obs_vol)
        #  Calculating the dF and restricting it to the left side of the observation point
        delta_f = df(old_field, adjoint_field)
        delta_f[x_obs_index:, :, :] = np.zeros((len(x) - x_obs_index, len(y), len(z)))

        #  picking the coordinates corresponding to the highest change in dF and updating the geometry
        x_inclusion, y_inclusion, z_inclusion = pick_extremum(delta_f, points_to_delete, points_for_3D_plot, [x, y, z])
        add_block([x_inclusion, y_inclusion, z_inclusion], block_size, geo_lst)

    forward_2D = get_fields(sim, obs_vol, True, slice_axis, z_obs_index)
    adjoint_2D = get_fields(sim_adjoint, obs_vol, True, slice_axis, z_obs_index)

    forward_ft_2D = get_fields(sim, obs_vol, True, slice_axis, z_obs_index)
    adjoint_ft_2D = get_fields(sim_adjoint, obs_vol, True, slice_axis, z_obs_index)

    axes = [x, y, z]
    intensities = [intensity_anim, intensity_at_obs]

    return axes, forward_2D, adjoint_2D, forward_ft_2D, adjoint_ft_2D, intensities
