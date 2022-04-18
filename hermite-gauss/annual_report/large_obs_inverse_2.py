import meep as mp
import numpy as np


# **********************************************************************************************************************
# UTILITY FUNCTIONS
# **********************************************************************************************************************
def normalise_complex_field(arr):
    f1, f2, f3, f4 = arr
    field = np.array([f1, f2, f3])
    norm = np.sqrt(np.real(f1 * np.conjugate(f1) + f2 * np.conjugate(f2) + f3 * np.conjugate(f3)))
    return field / norm


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def reduce_volume(fun, endpoint, startpoint):
    """Returns a reduced version of (l,l,l) matrix fun.In all 3 axes: the elements between 0:endpoint and
    startpoint:-1 are the only non-zero entries left """
    fun[:, :endpoint, :] = 0
    fun[:, startpoint:, :] = 0
    fun[:, :, :endpoint] = 0
    fun[:, :, startpoint:] = 0
    fun[:endpoint, :, :] = 0
    fun[startpoint:, :, :] = 0
    return fun


def install_function(axes, fun, reduced_vol=False):
    """Returns user defined function on a grid chosen by meep."""
    x, y, z = axes
    xx, yy, zz = np.meshgrid(x, y, z)
    fun = fun(xx, yy)
    if reduced_vol:
        c = int(len(x) / 4)
        d = 3 * c
        fun = reduce_volume(fun, c, d)
    return fun


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


def intensity_avg_area(axes, field, flux_indices):
    x, y, z = axes
    xi, xn, y0, zi, zn = flux_indices
    total = 0
    area = (x[xn] - x[xi]) * (z[zn] - z[zi])
    for i in range(xi, xn):
        for j in range(zi, zn):
            total += intensity_at_point(field, i, y0, j)

    return round(total / area, 4)


def get_helicity(arr_e, arr_h):
    ex, ey, ez, eps = arr_e
    hx, hy, hz, eps_h = arr_h
    # norm = 1 / np.real(ex * np.conjugate(ex) + ey * np.conjugate(ey) + ez * np.conjugate(ez))
    norm = 1
    helicity = np.imag(norm * (ex * np.conjugate(hx) + ey * np.conjugate(hy) + ez * np.conjugate(hz)))
    return helicity


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


def get_fields_h(simulation, obs_vol, fields_2D=False, *slice_axis_and_which_point):
    """If boolean fields_2D is True(default is False), slice axis and point need to be specified."""
    fields_data = np.array([simulation.get_array(center=mp.Vector3(), size=obs_vol, component=field) for field in
                            [mp.Hx, mp.Hy, mp.Hz,mp.Dielectric]])
    fields_data_elements = np.array([element[1:-1, 1:-1, 1:-1] for element in fields_data])
    if fields_2D:
        slice_axis, which_point = slice_axis_and_which_point
        fields_data_elements = [[a[which_point, :, :], a[:, which_point, :], a[:, :, which_point]][slice_axis]
                                for a
                                in fields_data_elements]
    # ex,ey,ez,epsilon

    return fields_data_elements


def exclude_points(arr_axes, arr_src, points_list):
    # points_list = points_to_delete
    x, y, z = arr_axes
    x_src_i, y_src_i, z_src_i = arr_src

    def _delta(l, m, n):
        return (l - m[n]) ** 2

    for x_i in x:
        for y_i in y:
            for z_i in z:
                src_dist = _delta(x_i, x, x_src_i) + _delta(y_i, y, y_src_i) + _delta(z_i, z, z_src_i)

                if src_dist < 1:
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


def df_point(old_field_arr, adj_field_arr, fun):
    e1, e2, e3, eps1 = old_field_arr
    a1, a2, a3, eps2 = adj_field_arr
    intensity = get_intensity(old_field_arr)
    d_func = 2 * np.real((a1 * e1 + a2 * e2 + a3 * e3)) * (intensity - fun)
    return d_func


def df_helicity(old_field_arr_e, old_field_arr_h, adj_field_arr, fun):
    e1, e2, e3, eps1 = old_field_arr_e
    a1, a2, a3 = adj_field_arr
    helicity = get_helicity(old_field_arr_e, old_field_arr_h)
    d_func = -2 * np.real((a1 * e1 + a2 * e2 + a3 * e3)) * (helicity - fun)
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


def produce_adjoint_volume(field, freq, dt, arr_coord):
    source_volume = []
    x_ax, y_ax, z_ax = arr_coord
    for i in range(len(x_ax)):
        for j in range(len(y_ax)):
            for k in range(len(z_ax)):
                for element in range(3):
                    source_volume.append(mp.Source(
                        mp.ContinuousSource(freq, width=dt, is_integrated=True),
                        component=[mp.Ex, mp.Ey, mp.Ez][element],
                        size=mp.Vector3(),
                        center=mp.Vector3(x_ax[i], y_ax[j], z_ax[k]),
                        amplitude=np.conjugate(field[element][i, j, k])))
    return source_volume


def produce_electric_dipole(field, freq, dt, arr_coord):
    source_volume = []
    x_ax, y_ax, z_ax = arr_coord
    for i in range(len(x_ax)):
        for j in range(len(y_ax)):
            for k in range(len(z_ax)):
                for element in range(3):
                    source_volume.append(mp.Source(
                        mp.ContinuousSource(freq, width=dt, is_integrated=True),
                        component=[mp.Ex, mp.Ey, mp.Ez][element],
                        size=mp.Vector3(),
                        center=mp.Vector3(x_ax[i], y_ax[j], z_ax[k]),
                        amplitude=np.conjugate(field[element][i, j, k])))
    return source_volume


def produce_magnetic_dipole(field, freq, dt, arr_coord):
    source_volume = []
    x_ax, y_ax, z_ax = arr_coord
    for i in range(len(x_ax)):
        for j in range(len(y_ax)):
            for k in range(len(z_ax)):
                for element in range(3):
                    source_volume.append(mp.Source(
                        mp.ContinuousSource(freq, width=dt, is_integrated=True),
                        component=[mp.Hx, mp.Hy, mp.Hz][element],
                        size=mp.Vector3(),
                        center=mp.Vector3(x_ax[i], y_ax[j], z_ax[k]),
                        amplitude=np.conjugate(field[element][i, j, k])))
    return source_volume


def produce_adjoint_area(field, freq, dt, arr_coord, flux_params):
    source_area = []
    x_ax, y_ax, z_ax = arr_coord
    x0, xn, y0, z0, zn = flux_params
    print(flux_params, x_ax[x0], x_ax[xn], y_ax[x0], y_ax[xn], z_ax[z0])
    for i in range(x0, xn):
        for j in range(z0, zn):
            for element in range(3):
                print("Source point", x_ax[i], z_ax[j])
                source_area.append(mp.Source(
                    mp.ContinuousSource(freq, width=dt, is_integrated=True),
                    component=[mp.Ex, mp.Ey, mp.Ez][element],
                    size=mp.Vector3(),
                    center=mp.Vector3(x_ax[i], y_ax[y0], z_ax[j]),
                    amplitude=np.conjugate(field[element][i, y0, j])))
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


def enlarge_block(arr, axes, multi):
    new_points = []
    u, v, w = axes
    for tup in arr:
        for i in range(0, multi):
            new_x = tup[0] + i
            for j in range(multi):
                new_y = tup[1] + j
                for k in range(multi):
                    new_z = tup[2] + k
                    new_points.append((new_x, new_y, new_z))

    return new_points
