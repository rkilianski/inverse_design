import sympy as sym
import numpy as np
from sympy import *


def create_rot_matrix_row(prp_to):  # to be solved
    RM = sym.eye(3, 3)
    index = 0
    Rx = sym.rot_axis1(sym.Symbol('x'))
    Ry = sym.rot_axis2(sym.Symbol('y'))
    Rz = sym.rot_axis3(sym.Symbol('z'))

    if prp_to == 0:
        RM = Ry * Rz
        index = 0
    if prp_to == 1:
        RM = Rx * Rz
        index = 1
    if prp_to == 2:
        RM = Rx * Ry
        index = 2

    RM_row = np.asarray(RM.row(index))
    return RM_row


z_zero = create_rot_matrix_row("z")


def permute_k(k_arr):
    v_pairs = []
    for i in range(len(k_arr) - 1):
        for j in range(i + 1):
            ki = k_arr[i + 1]
            kj = k_arr[j]
            v_pairs.append(ki - kj)

    return v_pairs


def create_system(v_pairs, row_matrix):
    system_eq = []
    for element in v_pairs:
        system = list(np.dot(row_matrix, np.transpose(element)))[0]
        system_eq.append(system)

    return system_eq


def _get_angles(k_arr, prp_to):
    """Provide an array of numerical k_vectors, i.e. if they contain sin and cos, evaluate at e.g. pi/4."""

    x, y, z = symbols(' x y z ')
    angles = [x, y, z]
    angles.pop(prp_to)
    row_of_rot = create_rot_matrix_row(prp_to)
    vector_pairs = permute_k(k_arr)
    system = create_system(vector_pairs, row_of_rot)
    solution = solve(system, angles)
    angle1 = solution[1][0]
    angle2 = solution[1][1]
    print(f'The angles are: phi: {angle1} and theta: {angle2} ')
    x1 = float(angle1.evalf())
    x2 = float(angle2.evalf())

    return x1, x2


def find_angles_and_rotate(k_arr, e_arr, prp_to):
    """Takes lists of k-vectors and corresponding e-vectors (polarisation vectors).
    Calculates the angle to rotate the axes by, so the pairs of k-vectors lie on the plane perpendicular to the
    axis specified by prp_to, e.g. (x=0,y=1,z=2).
    Returns rotated k-vectors and e-vectors."""

    new_k = []
    new_e = []
    angles = [0.0, 0.0, 0.0]

    phi, theta = _get_angles(k_arr, prp_to)

    if prp_to == 0:
        angles = [0, phi, theta]
    if prp_to == 1:
        angles = [phi, 0, theta]
    if prp_to == 2:
        angles = [phi, theta, 0]
    a_1, a_2, a_3 = angles[0], angles[1], angles[2]

    R_x = np.array([[1, 0, 0], [0, np.cos(a_1), np.sin(a_1)], [0, -np.sin(a_1), np.cos(a_1)]])
    R_y = np.array([[np.cos(a_2), 0, -np.sin(a_2)], [0, 1, 0], [np.sin(a_2), 0, np.cos(a_2)]])
    R_z = np.array([[np.cos(a_3), np.sin(a_3), 0], [-np.sin(a_3), np.cos(a_3), 0], [0, 0, 1]])
    RM = np.dot(R_x, np.dot(R_y, R_z))

    for element, pol_amp in zip(k_arr, e_arr):
        print(element)
        new_k.append(np.dot(RM, element))
        print("new vector:",np.dot(RM, element))
        new_e.append(np.dot(RM, pol_amp))

    print("length:",len(new_k))
    print("new vectors are:",new_k)

    return new_k, new_e


C = 1
THETA = np.pi / 3

K1 = C * np.array([np.cos(THETA), np.sin(THETA), 0])
K2 = C * np.array([np.cos(THETA), -np.sin(THETA), 0])
K3 = C * np.array([0, np.sin(THETA), np.cos(THETA)])
K4 = C * np.array([0, -np.sin(THETA), np.cos(THETA)])
k_v = [K1, K2, K3, K4]
