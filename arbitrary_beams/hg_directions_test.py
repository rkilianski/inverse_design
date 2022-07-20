from matplotlib import pyplot as plt

import module_hg_beam as mhg
import meep as mp
import numpy as np

RESOLUTION = 8
SLICE_AXIS = 2
CHOSEN_POINT = 20

DPML = 2  # thickness of perfectly matched layers (PMLs) around the box
PML_LAYERS = [mp.PML(DPML)]
DT = 5
T = 20
FCEN = 1

CELL_X, CELL_Y, CELL_Z = 10, 10, 10
OBS_X_A, OBS_Y_A, OBS_Z_A = 6, 6, 6  # dimensions of the computational cell, not including PML

sx = CELL_X + 2 * DPML
sy = CELL_Y + 2 * DPML
sz = CELL_Z + 2 * DPML

cell_3d = mp.Vector3(sx, sy, sz)
obs_vol = [OBS_X_A, OBS_Y_A, OBS_Z_A]

SRC_POS_X, SRC_POS_Y, SRC_POS_Z = -3, 0, 0

MATERIAL = mp.Medium(epsilon=1)
M, N = 0, 0
WAVELENGTH = 1
WAIST = 1

beams = []
plots_2D = []


def get_fields(simulation, slice_axis, which_point):
    fields_data = [simulation.get_array(center=mp.Vector3(), size=obs_vol, component=field) for field in
                   [mp.Ex, mp.Ey, mp.Ez]]
    fields_data_elements = [element[1:-1, 1:-1, 1:-1] for element in fields_data]
    fields_2D = [[a[which_point, :, :], a[:, which_point, :], a[:, :, which_point]][slice_axis]
                 for a
                 in fields_data_elements]
    # ex,ey,ez,epsilon
    return fields_2D


K1 = np.array([1, 0, 0])
K2 = np.array([0, 1, 0])
K3 = np.array((0, 0, 1))
K4 = np.array([1, 1, 0])
K5 = np.array([1, -1, 0])
K6 = np.array((-1, 1, 0))
K7 = np.array([-1, -1, 0])
K8 = np.array([-1, 0, 0])
K9 = np.array([0, -1, 0])

P = np.array([0, 0, 1])

k_vecs = [K1, K2, K3, K4, K5, K6, K7, K8, K9]

for vector in k_vecs:
    beams.append(
        mhg.make_hg_beam_any_dir(vector, P, FCEN, WAVELENGTH, [SRC_POS_X, SRC_POS_Y, SRC_POS_Z], obs_vol,
                                 waist=WAIST,
                                 m=0, n=0))

for i in range(9):
    sim = mp.Simulation(
        cell_size=cell_3d,
        sources=beams[i],
        boundary_layers=PML_LAYERS,
        resolution=RESOLUTION,
        geometry=[],
        default_material=MATERIAL,
        force_all_components=True,
        force_complex_fields=True
    )

    sim.run(until=20)

    x, y, z, w = sim.get_array_metadata(center=mp.Vector3(), size=obs_vol)
    [x, y, z] = [coordinate[1:-1] for coordinate in [x, y, z]]
    Ex, Ey, Ez = get_fields(sim, SLICE_AXIS, CHOSEN_POINT)
    e_squared = np.real((Ex * np.conjugate(Ex) + Ey * np.conjugate(Ey) + Ez * np.conjugate(Ez)))
    plots_2D.append(e_squared)

fig, ax = plt.subplots(3, 3, figsize=(12, 12))

ax[0, 0].pcolormesh(x, y, np.transpose(plots_2D[0]), cmap='Spectral', alpha=1)
ax[0, 0].set_title('[1,0,0]]')

ax[0, 1].pcolormesh(x, y, np.transpose(plots_2D[1]), cmap='Spectral', alpha=1)
ax[0, 1].set_title('[0, 1, 0]')

ax[0, 2].pcolormesh(x, y, np.transpose(plots_2D[2]), cmap='Spectral', alpha=1)
ax[0, 2].set_title('[0, 0, 1]')

ax[1, 0].pcolormesh(x, y, np.transpose(plots_2D[3]), cmap='Spectral', alpha=1)
ax[1, 0].set_title('[1, 1, 0]')

ax[1, 1].pcolormesh(x, y, np.transpose(plots_2D[4]), cmap='Spectral', alpha=1)
ax[1, 1].set_title('[1, -1, 0]')

ax[1, 2].pcolormesh(x, y, np.transpose(plots_2D[5]), cmap='Spectral', alpha=1)
ax[1, 2].set_title('[-1, 1, 0]')

ax[2, 0].pcolormesh(x, y, np.transpose(plots_2D[6]), cmap='Spectral', alpha=1)
ax[2, 0].set_title('[-1, -1, 0]')

ax[2, 1].pcolormesh(x, y, np.transpose(plots_2D[7]), cmap='Spectral', alpha=1)
ax[2, 1].set_title('[-1, 0, 0]')

ax[2, 2].pcolormesh(x, y, np.transpose(plots_2D[8]), cmap='Spectral', alpha=1)
ax[2, 2].set_title('[0, -1, 0]')

plt.show()
