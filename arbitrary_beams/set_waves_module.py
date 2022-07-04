import numpy as np


def make_4_wave_NI(C, theta, a1, a2, a4):
    """ The angle theta can be anything between 0 and pi/2, except pi/4"""
    K1 = C * np.array([np.cos(theta), np.sin(theta), 0])
    K2 = C * np.array([np.cos(theta), -np.sin(theta), 0])
    K3 = C * np.array([-np.cos(theta), np.sin(theta), 0])
    K4 = C * np.array([-np.cos(theta), -np.sin(theta), 0])

    amp1 = a1
    amp2 = a2
    amp3 = -a1 * np.conjugate(a2) * np.sign(np.cos(2 * theta)) / (np.conjugate(a4) * np.sqrt(np.abs(np.cos(2 * theta))))
    amp4 = a4 / (np.sqrt(np.abs(np.cos(2 * theta))))

    E1 = amp1 * np.array([0, 0, 1])
    E2 = amp2 * np.array([0, 0, 1])
    E3 = amp3 * np.array([np.sin(theta), np.cos(theta), 0])
    E4 = amp4 * np.array([-np.sin(theta), np.cos(theta), 0])

    k_vec = [K1, K2, K3, K4]
    e_vec = [E1, E2, E3, E4]

    return k_vec, e_vec


def make_4_wave_b_NI(C, theta, a1, a2, a4):
    """ The lattice doesn't lie on the xy plane. Vectors can be rotated using plane rotator module"""
    K1 = C * np.array([np.cos(theta), np.sin(theta), 0])
    K2 = C * np.array([np.cos(theta), -np.sin(theta), 0])
    K3 = C * np.array([0, np.sin(theta), np.cos(theta)])
    K4 = C * np.array([0, -np.sin(theta), np.cos(theta)])

    amp1 = a1
    amp2 = a2
    amp3 = -a1 * np.conjugate(a2) / np.conjugate(a4)
    amp4 = a4
    E1 = C * amp1 * np.array([0, 0, 1])
    E2 = C * amp2 * np.array([0, 0, 1])
    E3 = C * amp3 * np.array([1, 0, 0])
    E4 = C * amp4 * np.array([1, 0, 0])

    k_vec = [K1, K2, K3, K4]
    e_vec = [E1, E2, E3, E4]

    return k_vec, e_vec


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
