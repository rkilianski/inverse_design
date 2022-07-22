import meep as mp
import numpy as np
from scipy import special


#
# def change_coords(k_vector):
#     """ Helper function to the hg_amp function. Takes in a propagation direction k-vector, and rotates the axes to align
#     the propagation direction with the z-axis; the Hermite polynomial functions are accordingly rotated to be
#      perpendicular to k.
#      Returns the 3 rotated axes."""
#
#     k_vector = k_vector.astype('float64')
#     k_vector /= np.linalg.norm(k_vector)
#     kx, ky, kz = k_vector
#
#     PHI = 0
#     THETA = 0
#     ETA = 0
#
#     if kx != 0:
#         PHI = np.atan2(kz / kx)
#         if ky != 0:
#             THETA = np.arctan(kx / ky)
#             if kz != 0:
#                 ETA = np.arctan(kz / ky)
#     else:
#         if ky != 0:
#             ETA = np.arctan(kz / ky)
#
#     rot_theta = np.array([[np.cos(THETA), -np.sin(THETA), 0], [np.sin(THETA), np.cos(THETA), 0], [0, 0, 1]])
#     rot_phi = np.array([[np.cos(PHI), 0, -np.sin(PHI)], [0, 1, 0], [np.sin(PHI), 0, np.cos(PHI)]])
#     rot_eta = np.array([[1, 0, 0], [0, np.cos(ETA), -np.sin(ETA)], [0, np.sin(ETA), np.cos(ETA)]])
#
#     rot_all = np.dot(np.dot(rot_phi, rot_theta), rot_eta)
#     # create transformed coordinate axes; e.g. [1,0,0]-> [t1,t2,t3]
#     xp = np.dot(rot_all, np.array([1, 0, 0]))
#     yp = np.dot(rot_all, np.array([0, 1, 0]))
#     zp = np.dot(rot_all, np.array([0, 0, 1]))
#     print(f"k-vector {k_vector}: xp {xp},yp{yp}, zp{zp}")
#     return xp, yp, zp

def change_coords(k_vector):
    """ Helper function to the hg_amp function. Takes in a propagation direction k-vector, and rotates the axes to align
    the propagation direction with the z-axis; the Hermite polynomial functions are accordingly rotated to be
     perpendicular to k.
     Returns the 3 rotated axes."""

    k_vector = k_vector.astype('float64')
    k_vector /= np.linalg.norm(k_vector)
    kx, ky, kz = k_vector

    if kz != 1:
        norm_x = 1 / np.sqrt(1 - kz ** 4)
        norm_y = 1 / np.sqrt(1 - kz ** 2)
        dir_prop = kx, ky, kz
        x_prop = -kx * kz * norm_x, ky * kz * norm_x, (kx ** 2 + ky ** 2) * norm_x
        y_prop = -ky * norm_y, kx * norm_y, 0
    else:
        dir_prop = 0, 0, 1
        x_prop = 1, 0, 0
        y_prop = 0, 1, 0

    new_dir = dir_prop, x_prop, y_prop
    print(f"k-vector {k_vector}: dir prop {dir_prop},x prop{x_prop}, y_prop{y_prop}")
    return new_dir


def place_hg_source(box, k_vec):
    """ Situates the source so the given k-vector passes through the origin.
    Takes the dimensions of the simulation box- box(arr),
    and the k-vector(arr).
    Returns the 3D array source using meep's convention and putting the origin at the centre of the box."""
    # works for a cubic box
    side, _, _ = box
    indices = [0, 1, 2]
    source = np.array([0.0, 0.0, 0.0])
    index = np.argmax(np.abs(k_vec))
    indices.pop(index)
    largest = k_vec[index]
    source[index] = -np.sign(k_vec[index]) * side / 2
    boundary = source[index]
    lam = (boundary + largest) / largest
    for i in indices:
        source[i] = k_vec[i] * (lam - 1)
        if abs(source[i]) < 1e-3:
            source[i] = 0
    return source


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
    source_hg = [mp.Source(
        mp.ContinuousSource(frequency=fcen),
        component=comp,
        size=mp.Vector3(arr_src_size[0], arr_src_size[1], arr_src_size[2]),
        center=mp.Vector3(arr_src_cntr[0], arr_src_cntr[1], arr_src_cntr[2]),
        amp_func=hg_amp_func(dir_prop, waist, wavelength, m, n, mp.Vector3(0, 0, 0)))]
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

    # trying to fix the source problem
    if s1 != 0:
        if s2 == 0 and s3 == 0:
            arr_src_size[0] = 0

    else:
        if s2 == 0 and s3 != 0:
            arr_src_size[2] = 0
        if s3 == 0 and s2 != 0:
            arr_src_size[1] = 0

    pol_vec = pol_vec.astype('float64')
    pol_vec /= np.linalg.norm(pol_vec)
    pol_comp = [mp.Ex, mp.Ey, mp.Ez]

    for (element, comp) in zip(pol_vec, pol_comp):

        if element != 0:
            source_hg.append(mp.Source(
                mp.ContinuousSource(frequency=fcen),
                component=comp,
                size=mp.Vector3(arr_src_size[0], arr_src_size[1], arr_src_size[2]),
                center=mp.Vector3(s1, s2, s3),
                amp_func=hg_amp_func_any_dir(k_vec, element, waist, wavelength, m, n)))
    return source_hg


def hg_amp_func_any_dir(k_vector, pol_amp, waist_radius, wavelength, m, n):
    """Generates the HG amplitude function in a given direction.
    Takes: k_vector - dir. of propagation(arr),
    pol_amp - polarisation(arr),
    waist_radius - Gaussian beam waist,
    wavelength(float),
    m,n(int) - orders of Hermite polynomial in x and y directions, respectively"""

    new_coords = change_coords(k_vector)  # new co-ords are normalised vectors,i.e x1 = [v1,v2,v3]
    xp, yp, zp = new_coords

    def _hermite_fun(n_val, m_val):
        return special.hermite(n_val), special.hermite(m_val)

    def _hg_profile(mp_pos):
        prop_dir = xp[0] * mp_pos[0] + xp[1] * mp_pos[1] + xp[2] * mp_pos[2]
        x_profile = yp[0] * mp_pos[0] + yp[1] * mp_pos[1] + yp[2] * mp_pos[2]
        y_profile = zp[0] * mp_pos[0] + zp[1] * mp_pos[1] + zp[2] * mp_pos[2]
        r_squared = x_profile ** 2 + y_profile ** 2

        k = 2 * np.pi / wavelength
        z_R = np.pi * (waist_radius ** 2) / wavelength

        w = waist_radius * np.sqrt(1 + (prop_dir / z_R) ** 2)

        h_n, h_m = _hermite_fun(n, m)

        h_function = h_n(np.sqrt(2) * x_profile / w) * h_m(np.sqrt(2) * y_profile / w) * np.exp(
            1j * np.arctan(prop_dir / z_R) * (1 + n + m))
        exp_function = (waist_radius / w) * np.exp(-r_squared / (w ** 2)) * np.exp(
            -1j * (k * prop_dir + k * r_squared * prop_dir / (2 * (prop_dir ** 2 + z_R ** 2))))

        hg_beam = pol_amp * h_function * exp_function

        return hg_beam

    return _hg_profile


def make_multiple_hg_beams(k_vec_arr, pol_arr, fcen, wavelength, box, obs_vol, waist, m, n):
    beams = []
    beams_sep = []
    for k, e in zip(k_vec_arr, pol_arr):
        beams.append(make_hg_beam_any_dir(k, e, fcen, wavelength, box, obs_vol, waist, m=m, n=n))
    for beam in beams:
        for element in beam:
            beams_sep.append(element)

    return beams_sep
