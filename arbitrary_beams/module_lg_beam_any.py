import meep as mp
import numpy as np
from scipy import special


def change_coords(k_vector):
    """ Helper function to the hg_amp function. Takes in a propagation direction k-vector, and rotates the axes to align
    the propagation direction with the z-axis; the Hermite polynomial functions are accordingly rotated to be
     perpendicular to k.
     Returns the 3 rotated axes."""

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


def place_lg_source(box, k_vec):
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

    return source


def lg_amp_func_any_dir(k_vector, pol_amp, waist_radius, wavelength, p, l):
    """dir_of_prop takes in int: 0,1 or 2 corresponding to the propagation direction of the beam."""

    xp, yp, zp = change_coords(k_vector)  # new co-ords are normalised vectors,i.e x1 = [v1,v2,v3]

    l_abs = np.abs(l)

    def _laguerre_fun(p_val, alpha_val):
        return special.genlaguerre(p_val, alpha_val)

    def _lg_profile(mp_pos):
        z_dir = xp[0] * mp_pos[0] + xp[1] * mp_pos[1] + xp[2] * mp_pos[2]
        x_profile = yp[0] * mp_pos[0] + yp[1] * mp_pos[1] + yp[2] * mp_pos[2]
        y_profile = zp[0] * mp_pos[0] + zp[1] * mp_pos[1] + zp[2] * mp_pos[2]
        r_squared = x_profile ** 2 + y_profile ** 2
        r = np.sqrt(r_squared)

        k = 2 * np.pi / wavelength
        z_R = np.pi * (waist_radius ** 2) / wavelength
        R_inv = r_squared * z_dir / (2 * (z_dir ** 2 + z_R ** 2))
        w = waist_radius * np.sqrt(1 + (z_dir / z_R) ** 2)

        norm_constant = np.sqrt(2 * special.factorial(p) / (np.pi * (special.factorial(p + l_abs))))
        l_lp = _laguerre_fun(p, l_abs)
        psi = (l_abs + 2 * p + 1) * np.arctan(z_dir / z_R)
        phi = np.arctan2(x_profile, y_profile)

        lag_fun = norm_constant * l_lp(2 * r_squared / w ** 2) * (waist_radius / w) * ((r * np.sqrt(2) / w) ** l_abs)

        exp_fun = np.exp(-r_squared / (w ** 2)) * np.exp(-1j * k * R_inv) * np.exp(-1j * l * phi) * np.exp(
            1j * (psi - k * z_dir))

        lg_beam = pol_amp * lag_fun * exp_fun

        return lg_beam

    return _lg_profile


def make_lg_beam_any_dir(k_vec, pol_vec, fcen, wavelength, arr_src_size, obs_vol, waist, l, p):
    """fcen- frequency of the beam,
       arr_src_size, arr_src_cntr - lists(or np.arrays or mp.v3) of source size and source center respectively,
       dir_prop - single int from the set 0,1,2 where they represent x, y, z direction of propagation,
       waist - float, waist of a gaussian beam,
       l, alpha - integers corresponding to the Laguerre polynomials, i.e. L(l,alpha),
       comp - component to be excited, mp.Ez by default"""

    s1, s2, s3 = place_lg_source(obs_vol, k_vec)
    source_lg = []

    pol_vec = pol_vec.astype('float64')
    pol_vec /= np.linalg.norm(pol_vec)
    pol_comp = [mp.Ex, mp.Ey, mp.Ez]

    for (element, comp) in zip(pol_vec, pol_comp):
        if element != 0:
            source_lg.append(mp.Source(
                mp.ContinuousSource(frequency=fcen),
                component=comp,
                size=mp.Vector3(arr_src_size[0], arr_src_size[1], arr_src_size[2]),
                center=mp.Vector3(s1, s2, s3),
                amp_func=lg_amp_func_any_dir(k_vec, element, waist, wavelength, l, p)))
    return source_lg


def make_multiple_lg_beams(k_vec_arr, pol_arr, fcen, wavelength, box, obs_vol, waist, l, p):
    beams = []
    beams_sep = []
    for k, e in zip(k_vec_arr, pol_arr):
        beams.append(make_lg_beam_any_dir(k, e, fcen, wavelength, box, obs_vol, waist, l=l, p=p))
    for beam in beams:
        for element in beam:
            beams_sep.append(element)
    return beams_sep
