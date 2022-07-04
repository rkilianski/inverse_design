import meep as mp
import numpy as np
import sympy as sym

Rx = sym.rot_axis1(sym.Symbol('x'))
Ry = sym.rot_axis3(sym.Symbol('y'))
Rz = sym.rot_axis3(sym.Symbol('z'))

RM = Rx * Ry * Rz

z_zero_vector = RM.row(2)


def make_6_wave_NI(C, theta, a1, a2, a3):
    """ theta limited to pi/4 < theta < pi/2 or pi/2 < theta < 3pi/4"""
    K1 = C * np.array([1, 0, 0])
    K2 = C * np.array([np.cos(theta), np.sin(theta), 0])
    K3 = C * np.array([np.cos(theta), -np.sin(theta), 0])
    K4 = C * np.array([-1, 0, 0])
    K5 = C * np.array([-np.cos(theta), -np.sin(theta), 0])
    K6 = C * np.array([-np.cos(theta), np.sin(theta), 0])

    amp1 = a1
    amp2 = a2
    amp3 = a3
    amp4 = -np.conjugate(a1) * np.sqrt(np.abs(2 * theta)) / np.cos(theta)
    amp5 = np.conjugate(a2) / np.sqrt(np.abs(2 * theta))
    amp6 = np.conjugate(a3) / np.sqrt(np.abs(2 * theta))

    E1 = amp1 * np.array([0, 0, 1])
    E2 = amp2 * np.array([0, 0, 1])
    E3 = amp3 * np.array([0, 0, 1])
    E4 = amp4 * np.array([0, -1, 0])
    E5 = amp5 * np.array([np.sin(theta), -np.cos(theta), 0])
    E6 = amp6 * np.array([-np.sin(theta), -np.cos(theta), 0])

    k_vec = [K1, K2, K3, K4, K5, K6]
    e_vec = [E1, E2, E3, E4, E5, E6]

    return k_vec, e_vec
