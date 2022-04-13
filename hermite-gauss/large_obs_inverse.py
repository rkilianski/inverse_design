import meep as mp
import numpy as np


# **********************************************************************************************************************
# UTILITY FUNCTIONS
# **********************************************************************************************************************
def get_intensity(arr):
    """Returns normalised intensity I for all x and y. """
    ex, ey, ez = arr[: 3]
    e_sq = np.real(ex * np.conjugate(ex) + ey * np.conjugate(ey) + ez * np.conjugate(ez))
    norm_e_sq = (1 / (np.amax(e_sq))) * e_sq
    return norm_e_sq


def intensity_at_point(field, x_index, y_index, z_index):
    ex, ey, ez, eps = field
    intensity = np.real((ex * np.conjugate(ex) + ey * np.conjugate(ey) + ez * np.conjugate(ez)))
    intensity_at_x0 = intensity[x_index, y_index, z_index]
    return intensity_at_x0


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def get_fields(simulation, obs_vol, fields_2D=False, *slice_axis_and_which_point):
    """If boolean fields_2D is True(default is False), slice axis and point need to be specified."""
    fields_data = np.array([simulation.get_array(center=mp.Vector3(), size=obs_vol, component=field) for field in
                            [mp.Ex, mp.Ey, mp.Ez, mp.Dielectric]])
    fields_data_elements = np.array([element[1:-1, 1:-1, 1:-1] for element in fields_data])
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

    def _delta(l, m, n):
        return (l - m[n]) ** 2

    for x_i in x:
        for y_i in y:
            for z_i in z:
                src_dist = _delta(x_i, x, x_src_i) + _delta(y_i, y, y_src_i) + _delta(z_i, z, z_src_i)

                if src_dist < 0.5:
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


def df_point(old_field_arr, adj_field_arr, axes, fun):
    x, y, z = axes
    xx, yy, zz = np.meshgrid(x, y, z)
    fun = fun(xx, yy)
    # fun = np.sin(4 * (xx + yy)) + np.cos(4 * (xx - yy))

    e1, e2, e3, eps1 = old_field_arr
    a1, a2, a3, eps2 = adj_field_arr
    intensity = get_intensity(old_field_arr)
    d_func = -2 * np.real((a1 * e1 + a2 * e2 + a3 * e3)) * (intensity - fun)
    return d_func


def delete_existing(arr, lst, dim_3D=True, multi=1):
    """multi is the multiplier applied to block size,
    it is only used in 2D projection to adjust the size of inclusions for animation."""
    # lst = points_to_delete
    # 2D working only when 2d is x and y - rewrite!
    axis = arr[0]

    if dim_3D:
        for tup in lst:
            arr[tup[0], tup[1], tup[2]] = 0
    else:
        for tup in lst:
            for i in range(multi):
                for j in range(multi):
                    #  Edge cases
                    if tup[0] + i >= len(axis) > tup[1] + j:
                        arr[tup[0], tup[1] + j] = 0
                    if tup[0] + i < len(axis) <= tup[1] + j:
                        arr[tup[0] + i, tup[1]] = 0
                    # Regular case
                    if tup[0] + i < len(axis) > tup[1] + j:
                        arr[tup[0] + i, tup[1] + j] = 0

    return arr


def pick_extremum(delta, lst):
    # lst = points_to_delete
    """Returns a tuple of points (x,y,z) corresponding to the highest value of the dF."""
    if len(lst) > 0:
        delta = delete_existing(delta, lst)
    extr_x, extr_y, extr_z = np.unravel_index(delta.argmin(), delta.shape)
    return extr_x, extr_y, extr_z


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


def produce_obs_area(field, freq, dt, arr_coord):
    source_area = []
    x_ax, y_ax, z_ax = arr_coord
    for i in range(len(x_ax)):
        for j in range(len(y_ax)):
            for k in range(len(z_ax)):
                for element in range(3):
                    source_area.append(mp.Source(
                        mp.ContinuousSource(freq, width=dt, is_integrated=True),
                        component=[mp.Ex, mp.Ey, mp.Ez][element],
                        size=mp.Vector3(),
                        center=mp.Vector3(x_ax[i], y_ax[j], z_ax[k]),
                        amplitude=np.conjugate(field[element][i, j, k])))
    return source_area


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

def produce_simulation(fun, src_param_arr, multi_block_arr, ft_freq, time, obs_vol, obs_pt_arr, src_pt_arr, lsts,
                       iter,
                       slc_ax,
                       adj_dt):
    global adjoint_field

    OBS_POS_X, OBS_POS_Y, OBS_POS_Z = obs_pt_arr
    SRC_POS_X, SRC_POS_Y, SRC_POS_Z = src_pt_arr
    cell_size, source, pml, res, geo_lst = src_param_arr
    multiplier, block_size = multi_block_arr
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

    sim.run(until=time)

    x, y, z, w = sim.get_array_metadata(center=mp.Vector3(), size=obs_vol)
    [x, y, z] = [coordinate[1:-1] for coordinate in [x, y, z]]

    x_obs_index, y_obs_index, z_obs_index = [find_nearest(i, j) for i, j in
                                             zip([x, y, z], [OBS_POS_X, OBS_POS_Y, OBS_POS_Z])]
    x_src_index, y_src_index, z_src_index = [find_nearest(i, j) for i, j in
                                             zip([x, y, z], [SRC_POS_X, SRC_POS_Y, SRC_POS_Z])]

    # Simulate a field and use its values at obs points to simulate a fictitious field - adjoint field.
    old_field = get_fields(sim, obs_vol)
    old_field = (1 / (np.amax(old_field))) * old_field

    # Recording a snapshot of 2D intensity pattern for animation
    intensity_2D = get_intensity(get_fields(sim, obs_vol, True, slice_axis, z_obs_index))

    # Deleting grid points where blocks have been placed
    intensity_2D_blocks = delete_existing(intensity_2D, points_for_3D_plot, False, multiplier)
    intensity_anim.append(intensity_2D_blocks)
    # Exciting a fictitious dipole for the adjoint field
    dipole_at_obs = produce_obs_area(old_field, ft_freq, adj_dt, [x, y, z])

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

    adjoint_field = get_fields(sim_adjoint, obs_vol)
    adjoint_field = (1 / np.amax(adjoint_field)) * adjoint_field

    delta_f = df_point(old_field, adjoint_field, [x, y, z], fun)
    # delta_f[x_obs_index:, :, :] = np.zeros((len(x) - x_obs_index, len(y), len(z)))

    ########################################################################################################################
    # SIMULATION SECOND STEP: updating geometry from starting conditions and repeating the process.
    ########################################################################################################################

    exclude_points([x, y, z], [x_src_index, y_src_index, z_src_index], [x_obs_index, y_obs_index, z_obs_index],
                   points_to_delete)

    x_index, y_index, z_index = pick_extremum(delta_f, points_to_delete)
    [x_inclusion, y_inclusion, z_inclusion] = x[x_index], y[y_index], z[z_index]

    points_to_delete.append((x_index, y_index, z_index))
    points_for_3D_plot.append((x_index, y_index, z_index))

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

        sim.run(until=time)

        old_field = get_fields(sim, obs_vol)
        old_field = (1 / (np.amax(old_field))) * old_field
        # Recording a snapshot of 2D intensity pattern for animation
        intensity_2D = get_intensity(get_fields(sim, obs_vol, True, slice_axis, z_obs_index))

        # Deleting grid points where blocks have been placed
        intensity_2D_blocks = delete_existing(intensity_2D, points_for_3D_plot, False, multiplier)
        intensity_anim.append(intensity_2D_blocks)

        # recording intensity at observation point after adding a block in a previous turn
        intensity_at_obs.append(intensity_at_point(old_field, x_obs_index, y_obs_index, z_obs_index))
        # recording intensity over a time interval

        sim_adjoint = mp.Simulation(
            cell_size=cell_size,
            sources=produce_obs_area(old_field, ft_freq, adj_dt, [x, y, z]),
            boundary_layers=pml,
            resolution=res,
            geometry=geo_lst,
            force_all_components=True,
            force_complex_fields=True,

        )

        sim_adjoint.run(until=time)

        adjoint_field = get_fields(sim_adjoint, obs_vol)
        adjoint_field = (1 / (np.amax(adjoint_field))) * adjoint_field

        #  Calculating the dF and restricting it to the left side of the observation point
        delta_f = df_point(old_field, adjoint_field, [x, y, z], fun)
        # delta_f[x_obs_index:, :, :] = np.zeros((len(x) - x_obs_index, len(y), len(z)))

        #  picking the coordinates corresponding to the highest change in dF and updating the geometry

        x_index, y_index, z_index = pick_extremum(delta_f, points_to_delete)
        [x_inclusion, y_inclusion, z_inclusion] = x[x_index], y[y_index], z[z_index]

        points_to_delete.append((x_index, y_index, z_index))
        points_for_3D_plot.append((x_index, y_index, z_index))

        add_block([x_inclusion, y_inclusion, z_inclusion], block_size, geo_lst)

    forward_2D = get_fields(sim, obs_vol, True, slice_axis, z_obs_index)
    adjoint_2D = get_fields(sim_adjoint, obs_vol, True, slice_axis, z_obs_index)
    df_2D = delta_f[:, :, z_obs_index]
    axes = [x, y, z]
    intensities = [intensity_anim, intensity_at_obs]

    return axes, forward_2D, adjoint_2D, df_2D, intensities
