import meep as mp
import numpy as np
from scipy import special


def change_coords(k_vector):
    k_vector = k_vector.astype('float64')
    k_vector /= np.linalg.norm(k_vector)
    kx, ky, kz = k_vector

    PHI = 0
    THETA = 0
    ETA = 0

    if kx != 0:
        PHI = np.arctan(kz / kx)
        if ky != 0:
            THETA = np.arctan(kx / ky)
            if kz != 0:
                ETA = np.arctan(kz / ky)
    else:
        if ky != 0:
            ETA = np.arctan(kz / ky)

    rot_theta = np.array([[np.cos(THETA), -np.sin(THETA), 0], [np.sin(THETA), np.cos(THETA), 0], [0, 0, 1]])
    rot_phi = np.array([[np.cos(PHI), 0, -np.sin(PHI)], [0, 1, 0], [np.sin(PHI), 0, np.cos(PHI)]])
    rot_eta = np.array([[1, 0, 0], [0, np.cos(ETA), -np.sin(ETA)], [0, np.sin(ETA), np.cos(ETA)]])

    rot_all = np.dot(np.dot(rot_phi, rot_theta), rot_eta)

    # create transformed coordinate axes; e.g. [1,0,0]-> [t1,t2,t3]
    xp = np.dot(rot_all, np.array([1, 0, 0]))
    yp = np.dot(rot_all, np.array([0, 1, 0]))
    zp = np.dot(rot_all, np.array([0, 0, 1]))

    return xp, yp, zp


def angle_check(a, b, dim_a, dim_b, side):
    # a,b are 2 elements of kvector, dim parameter corresponds to that element's axis, e.g. 0,1,2
    sign = 1
    if a / b < 0:
        sign = -1
    val = np.array([0, 0, 0])
    ANGLE = np.arctan(b / a)

    if abs(a) > abs(b):
        print(dim_b,"dim_b")
        val[dim_b] = sign * (side / 2 -(b + (side/2-a))/np.tan(np.pi/2-b/a))
        print("a>b", sign * (side / 2 -(b + (side/2-a))/np.tan(np.pi/2-b/a)))
    if abs(b) > abs(a):
        print(dim_a, "dim_a")
        val[dim_a] = sign * (side / 2 - (a/b)*(side/2 - b))
        print("a<b", sign * (side / 2 - (a/b)*(side/2 - b)))
        print(val)
    return val


def place_hg_source(box_dims, k_vec):
    kx, ky, kz = k_vec
    sx, sy, sz = box_dims
    box_dims = np.array(([sx, sy, sz]))
    print(kx, ky, kz)
    for i in range(3):
        if k_vec[i] > 0:
            box_dims[i] = -box_dims[i] / 2
        else:
            if k_vec[i] < 0:
                box_dims[i] = box_dims[i] / 2
            else:
                box_dims[i] = 0

    if kx != 0:
        if ky != 0:
            box_dims += angle_check(kx, ky, 0, 1, sx)
            if kz != 0:
                box_dims += angle_check(kx, kz, 0, 2, sx)
                box_dims += angle_check(ky, kz, 1, 2, sx)
        elif kz != 0:
            box_dims += angle_check(kx, kz, 0, 2, sx)

    else:
        if ky != 0:
            if kz != 0:
                box_dims += angle_check(ky, kz, 1, 2, sx)
    print("box_dims", box_dims, (kx, ky, kz))
    return box_dims


def hg_amp_func(dir_of_prop, waist_radius, wavelength, m, n, beam_focus=mp.Vector3()):
    """dir_of_prop takes in int: 0,1 or 2 corresponding to the propagation direction of the beam."""

    x0, y0, z0 = beam_focus

    def _hermite_fun(n_val, m_val):
        return special.hermite(n_val), special.hermite(m_val)

    def _hg_profile(meep_vector_of_pos):
        indices = [0, 1, 2]
        prop_dir = meep_vector_of_pos[dir_of_prop]
        indices.pop(dir_of_prop)
        x_index, y_index = indices
        x_profile = meep_vector_of_pos[x_index] + x0
        y_profile = meep_vector_of_pos[y_index] + y0
        r_squared = x_profile ** 2 + y_profile ** 2

        k = 2 * np.pi / wavelength
        z_R = np.pi * (waist_radius ** 2) / wavelength

        w = waist_radius * np.sqrt(1 + prop_dir / z_R)
        h_n, h_m = _hermite_fun(n, m)

        h_function = h_n(np.sqrt(2) * x_profile / w) * h_m(np.sqrt(2) * y_profile / w) * np.exp(
            1j * np.arctan(prop_dir / z_R) * (1 + n + m))
        exp_function = (waist_radius / w) * np.exp(-r_squared / (w ** 2)) * np.exp(
            -1j * (k * prop_dir + k * r_squared * prop_dir / (2 * (prop_dir ** 2 + z_R ** 2))))

        hg_beam = h_function * exp_function

        return hg_beam

    return _hg_profile


# put the brackets back over source
def make_hg_beam(fcen, wavelength, arr_src_size, arr_src_cntr, dir_prop, waist, m, n, comp=mp.Ez):
    """fcen- frequency of the beam,
       arr_src_size, arr_src_cntr - lists(or np.arrays or mp.v3) of source size and source center respectively,
       dir_prop - single int from the set 0,1,2 where they represent x, y, z direction of propagation,
       waist - float, waist of a gaussian beam,
       m, n - integers corresponding to the Hermite polynomials, i.e. H(m)H(n),
       comp - component to be excited, mp.Ez by default"""
    source_hg = mp.Source(
        mp.ContinuousSource(frequency=fcen),
        component=comp,
        size=mp.Vector3(arr_src_size[0], arr_src_size[1], arr_src_size[2]),
        center=mp.Vector3(arr_src_cntr[0], arr_src_cntr[1], arr_src_cntr[2]),
        amp_func=hg_amp_func(dir_prop, waist, wavelength, m, n, mp.Vector3(0, 0, 0)))
    return source_hg


def make_hg_beam_any_dir(k_vec, pol_vec, fcen, wavelength, arr_src_size, obs_vol, waist, m, n):
    """fcen- frequency of the beam,
       arr_src_size, arr_src_cntr - lists(or np.arrays or mp.v3) of source size and source center respectively,
       dir_prop - single int from the set 0,1,2 where they represent x, y, z direction of propagation,
       waist - float, waist of a gaussian beam,
       m, n - integers corresponding to the Hermite polynomials, i.e. H(m)H(n),
       comp - component to be excited, mp.Ez by default"""

    s1, s2, s3 = place_hg_source(obs_vol, k_vec)
    source_hg = []

    pol_vec = pol_vec.astype('float64')
    pol_vec /= np.linalg.norm(pol_vec)
    pol_comp = [mp.Ex, mp.Ey, mp.Ez]

    for (element, comp) in zip(pol_vec, pol_comp):
        source_hg.append(mp.Source(
            mp.ContinuousSource(frequency=fcen),
            component=comp,
            size=mp.Vector3(arr_src_size[0], arr_src_size[1], arr_src_size[2]),
            center=mp.Vector3(s1, s2, s3),
            amp_func=hg_amp_func_any_dir(k_vec, element, waist, wavelength, m, n)))
    return source_hg[0], source_hg[1], source_hg[2]


def hg_amp_func_any_dir(k_vector, pol_amp, waist_radius, wavelength, m, n):
    """dir_of_prop takes in int: 0,1 or 2 corresponding to the propagation direction of the beam."""
    xp, yp, zp = change_coords(k_vector)  # new co-ords are normalised vectors,i.e x1 = [v1,v2,v3]

    def _hermite_fun(n_val, m_val):
        return special.hermite(n_val), special.hermite(m_val)

    def _hg_profile(mp_pos):
        prop_dir = xp[0] * mp_pos[0] + xp[1] * mp_pos[1] + xp[2] * mp_pos[2]
        x_profile = yp[0] * mp_pos[0] + yp[1] * mp_pos[1] + yp[2] * mp_pos[2]
        y_profile = zp[0] * mp_pos[0] + zp[1] * mp_pos[1] + zp[2] * mp_pos[2]
        r_squared = x_profile ** 2 + y_profile ** 2

        k = 2 * np.pi / wavelength
        z_R = np.pi * (waist_radius ** 2) / wavelength

        w = waist_radius * np.sqrt(1 + np.abs(prop_dir) / z_R)

        h_n, h_m = _hermite_fun(n, m)

        h_function = h_n(np.sqrt(2) * x_profile / w) * h_m(np.sqrt(2) * y_profile / w) * np.exp(
            1j * np.arctan(prop_dir / z_R) * (1 + n + m))
        exp_function = (waist_radius / w) * np.exp(-r_squared / (w ** 2)) * np.exp(
            -1j * (k * prop_dir + k * r_squared * prop_dir / (2 * (prop_dir ** 2 + z_R ** 2))))

        hg_beam = pol_amp * h_function * exp_function

        return hg_beam

    return _hg_profile
