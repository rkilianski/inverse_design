import meep as mp
import numpy as np
from scipy import special


def lg_amp_func(dir_of_prop, waist_radius, wavelength, p, alpha):
    """dir_of_prop takes in int: 0,1 or 2 corresponding to the propagation direction of the beam."""

    np.seterr(invalid='ignore')  # ignoring warning when arctan attempts to div by 0

    alpha = np.abs(alpha)

    def _safe_arctan(y, x, r):
        ans = 0
        if x == 0 and y == 0:
            ans = 0
        if x == 0 and y != 0:
            ans = np.arcsin(y / r)
        if x > 0:
            ans = np.arctan(y / x)
        if x < 0:
            ans = -np.arcsin(y / r) + np.pi

        return ans

    def _laguerre_fun(p_val, alpha_val):
        return special.genlaguerre(p_val, alpha_val)

    def _lg_profile(meep_vector_of_pos):

        indices = [0, 1, 2]
        z_dir = meep_vector_of_pos[dir_of_prop]
        indices.pop(dir_of_prop)

        x_index, y_index = indices
        x_profile = meep_vector_of_pos[x_index]
        y_profile = meep_vector_of_pos[y_index]
        r_squared = x_profile ** 2 + y_profile ** 2
        r = np.sqrt(r_squared)

        k = 2 * np.pi / wavelength
        z_R = np.pi * (waist_radius ** 2) / wavelength
        R_inv = r_squared * z_dir / (2 * (z_dir ** 2 + z_R ** 2))
        w = waist_radius * np.sqrt(1 + (z_dir / z_R))
        norm_constant = np.sqrt(2 * special.factorial(p) / (np.pi * (special.factorial(p + alpha))))
        l_lp = _laguerre_fun(p, alpha)
        psi = (alpha + 2 * p + 1) * _safe_arctan(z_dir, z_R, r)
        phi = _safe_arctan(y_profile, x_profile, r)

        lag_fun = norm_constant * l_lp(2 * r_squared / w ** 2) * (waist_radius / w) * ((r * np.sqrt(2) / w) ** alpha)

        exp_fun = np.exp(-r_squared / (w ** 2)) * np.exp(-1j * k * R_inv) * np.exp(-1j * alpha * phi) * np.exp(1j * psi)

        lg_beam = lag_fun * exp_fun

        return lg_beam

    return _lg_profile


def make_lg_beam(fcen, wavelength, arr_src_size, arr_src_cntr, dir_prop, waist, m, n, comp=mp.Ez):
    """fcen- frequency of the beam,
       arr_src_size, arr_src_cntr - lists(or np.arrays or mp.v3) of source size and source center respectively,
       dir_prop - single int from the set 0,1,2 where they represent x, y, z direction of propagation,
       waist - float, waist of a gaussian beam,
       l, alpha - integers corresponding to the Laguerre polynomials, i.e. L(l,alpha),
       comp - component to be excited, mp.Ez by default"""
    source_lg = [mp.Source(
        mp.ContinuousSource(frequency=fcen),
        component=comp,
        size=mp.Vector3(arr_src_size[0], arr_src_size[1], arr_src_size[2]),
        center=mp.Vector3(arr_src_cntr[0], arr_src_cntr[1], arr_src_cntr[2]),
        amp_func=lg_amp_func(dir_prop, waist, wavelength, m, n))]
    return source_lg


def make_lg_beam_CP(fcen, wavelength, arr_src_size, arr_src_cntr, dir_prop, waist, m, n, polar="right"):
    """Make circularly polarised LG beam by superposing x and y;
       fcen- frequency of the beam,
       arr_src_size, arr_src_cntr - lists(or np.arrays or mp.v3) of source size and source center respectively,
       dir_prop - single int from the set 0,1,2 where they represent x, y, z direction of propagation,
       waist - float, waist of a gaussian beam,
       l, alpha - integers corresponding to the Laguerre polynomials, i.e. L(l,alpha),
       polar - polarisation; left - x - y; right - x + y;
      """

    # polarisation components
    pols = [mp.Ex, mp.Ey, mp.Ez]
    pols.pop(dir_prop)
    x_pol, y_pol = pols

    if polar == "right":
        a = 1
    else:
        a = -1
    beams = [mp.Source(
        mp.ContinuousSource(frequency=fcen),
        component=x_pol,
        size=mp.Vector3(arr_src_size[0], arr_src_size[1], arr_src_size[2]),
        center=mp.Vector3(arr_src_cntr[0], arr_src_cntr[1], arr_src_cntr[2]),
        amp_func=lg_amp_func(dir_prop, waist, wavelength, m, n)), mp.Source(
        mp.ContinuousSource(frequency=fcen),
        component=a * y_pol,
        size=mp.Vector3(arr_src_size[0], arr_src_size[1], arr_src_size[2]),
        center=mp.Vector3(arr_src_cntr[0], arr_src_cntr[1], arr_src_cntr[2]),
        amp_func=lg_amp_func(dir_prop, waist, wavelength, m, n))]

    return beams
