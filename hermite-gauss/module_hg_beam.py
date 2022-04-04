import meep as mp
import numpy as np
from scipy import special


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
