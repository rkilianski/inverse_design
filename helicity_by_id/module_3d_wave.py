from __future__ import division
import math
import meep as mp
import numpy as np


def pw_amp(k, x0):
    """Returns an amplitude exp(ik(x+x0)) at x.
    x0 needs to be specified as the centre of a current source."""

    def _pw_amp(x):
        return np.exp(1j * k.dot(x + x0))

    return _pw_amp


def k_vector(arr, f, n):
    """ Normalises and returns a vector in meep's Vector3 format."""
    k_direction = mp.Vector3(arr[0], arr[1], arr[2])
    k = k_direction.unit().scale(2 * math.pi * f * n)
    return k


def make_3d_wave(k_list, p_list, frequency, width_f, comp_cell, obs_cell, n):
    """Takes a list of k-vectors, along with a list of corresponding polarisation vectors.
    Each component of the k-vector is modified by the pw_amp function and is given an amplitude corresponding to its
    polarisation component.
        If all directions are present, the sources are:
    wave in kx - polarised in z,
    wave in ky - polarised in z,
    wave in kz - polarised in x,
    wave in kz - polarised in y.
        Returns a list of sources."""

    waves = []
    comp_cell = np.array(comp_cell)
    sx, sy, sz = comp_cell
    cx, cy, cz = obs_cell
    p_coefficient = 1  # this is either 0.5 or 1, depending on if we have 1 or 2 waves in the xy plane

    for k_i, p_i in zip(k_list, p_list):
        kx, ky, kz = k_i
        e1, e2, e3 = p_i
        if e3 != 0:  # if the wave is polarised in z, the combination of kx and ky will oscillate in z
            if kx != 0:
                if ky != 0:
                    p_coefficient = 0.5
                if kx > 0:
                    position = -cx / 2
                else:
                    position = cx / 2
                waves.append(mp.Source(
                    mp.ContinuousSource(frequency, fwidth=width_f, is_integrated=True),
                    component=mp.Ez,
                    size=mp.Vector3(0, sy, sz),
                    center=mp.Vector3(position, 0, 0),
                    amp_func=pw_amp(k_vector(k_i, frequency, n), mp.Vector3(position, 0, 0)),
                    amplitude=p_coefficient * e3

                )),
            if ky != 0:
                if ky > 0:
                    position = -cy / 2
                else:
                    position = cy / 2
                waves.append(mp.Source(
                    mp.ContinuousSource(frequency, fwidth=width_f, is_integrated=True),
                    component=mp.Ez,
                    size=mp.Vector3(sx, 0, sz),
                    center=mp.Vector3(0, position, 0),
                    amp_func=pw_amp(k_vector(k_i, frequency, n), mp.Vector3(0, position, 0)),
                    amplitude=p_coefficient * e3
                )),
            if kz != 0:
                if kz > 0:
                    position = -cz / 2
                else:
                    position = cz / 2
                for amp, pol_direction in zip([e1, e2], [mp.Ex, mp.Ey]):
                    waves.append(mp.Source(
                        mp.ContinuousSource(frequency, fwidth=width_f, is_integrated=True),
                        component=pol_direction,
                        size=mp.Vector3(sx, sy, 0),
                        center=mp.Vector3(0, 0, position),
                        amp_func=pw_amp(k_vector(k_i, frequency, n), mp.Vector3(0, 0, position)),
                        amplitude=amp))
        else:  # if the wave is not polarised in z, kx will oscillate in y and ky in x.
            if kx != 0:
                if kx > 0:
                    position = -cx / 2
                else:
                    position = cx / 2
                waves.append(mp.Source(
                    mp.ContinuousSource(frequency, fwidth=width_f, is_integrated=True),
                    component=mp.Ey,
                    size=mp.Vector3(0, sy, sz),
                    center=mp.Vector3(position, 0, 0),
                    amp_func=pw_amp(k_vector(k_i, frequency, n), mp.Vector3(position, 0, 0)),
                    amplitude=e2
                )),
            if ky != 0:
                if ky > 0:
                    position = -cy / 2
                else:
                    position = cy / 2
                waves.append(mp.Source(
                    mp.ContinuousSource(frequency, fwidth=width_f, is_integrated=True),
                    component=mp.Ex,
                    size=mp.Vector3(sx, 0, sz),
                    center=mp.Vector3(0, position, 0),
                    amp_func=pw_amp(k_vector(k_i, frequency, n), mp.Vector3(0, position, 0)),
                    amplitude=e1
                )),

    return waves
