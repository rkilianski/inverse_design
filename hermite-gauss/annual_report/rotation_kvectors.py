import numpy as np

THETA = 3 * np.pi / 4
PHI = np.arctan(-np.sin(THETA))
ZETA = np.pi/2
R_x = np.array([[1, 0, 0], [0, np.cos(PHI), -np.sin(PHI)], [0, np.sin(PHI), np.cos(PHI)]])
R_y = np.array([[np.cos(THETA), 0, np.sin(THETA)], [0, 1, 0], [-np.sin(THETA), 0, np.cos(THETA)]])
R_z = np.array([[np.cos(ZETA), -np.sin(ZETA), 0], [np.sin(ZETA), np.cos(ZETA), 0], [0, 0, 1]])

K_1 = np.array([1, 0, 0])
K_2 = np.array([0, 1, 0])
K_3 = np.array([0, 0, 1])

rotated_K1 = np.dot(R_x, np.dot(R_y, K_1))
rotated_K2 = np.dot(R_x, np.dot(R_y, K_2))
rotated_K3 = np.dot(R_x, np.dot(R_y, K_3))

V1 = rotated_K1-rotated_K2
V2 = rotated_K2-rotated_K3
V3 = rotated_K3-rotated_K1

zk1 = np.dot(R_z,rotated_K1)
zk2 = np.dot(R_z,rotated_K2)
zk3 = np.dot(R_z,rotated_K3)

xy_rotated_k_vectors = [rotated_K1,rotated_K2,rotated_K3]
z_rotated_k_vectors = [zk1,zk2,zk3]

W1 = zk1 -zk2
W2 = zk2 - zk3
W3 = zk3 - zk1

rotation_matrix = np.dot(R_z, np.dot(R_x, R_y))

rotated_K_vectors = [rotated_K1,rotated_K2,rotated_K3]


k_rot_elements = []
z_k_rot_elements = []

for k_vec in rotated_K_vectors:

    for i in range(3):
        vector = [0, 0, 0]
        vector[i] = k_vec[i]
        k_rot_elements.append(vector)

for k_vec in z_rotated_k_vectors:

    for i in range(3):
        vector = [0, 0, 0]
        vector[i] = k_vec[i]
        z_k_rot_elements.append(vector)


def k_vec_amp(arr):
    amplitudes = []
    for element in arr:
        amplitudes.append(np.linalg.norm(element))
    return amplitudes


k_amps = k_vec_amp(k_rot_elements)
Z_k_amps = k_vec_amp(z_k_rot_elements)

