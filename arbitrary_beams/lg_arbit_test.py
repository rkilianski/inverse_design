from matplotlib import pyplot as plt

import module_lg_beam_any as mlg
import meep as mp
import numpy as np

RESOLUTION = 6
SLICE_AXIS = 2
CHOSEN_POINT = 20

DPML = 2  # thickness of perfectly matched layers (PMLs) around the box
PML_LAYERS = [mp.PML(DPML)]
DT = 5
T = 100
FCEN = 1

CELL_X, CELL_Y, CELL_Z = 8, 8, 8
OBS_X_A, OBS_Y_A, OBS_Z_A = 6, 6, 6  # dimensions of the computational cell, not including PML

sx = CELL_X + 2 * DPML
sy = CELL_Y + 2 * DPML
sz = CELL_Z + 2 * DPML

cell_3d = mp.Vector3(sx, sy, sz)
OBS_VOL = mp.Vector3(OBS_X_A, OBS_Y_A, OBS_Z_A)

SRC_POS_X, SRC_POS_Y, SRC_POS_Z = -3, 0, 0

MATERIAL = mp.Medium(epsilon=1)
L, P = 0,0
WAVELENGTH = 1
WAIST = 3

plots_2D = []


def get_fields(simulation, slice_axis, which_point):
    fields_data = [simulation.get_array(center=mp.Vector3(), size=OBS_VOL, component=field) for field in
                   [mp.Ex, mp.Ey, mp.Ez]]
    fields_data_elements = [element[1:-1, 1:-1, 1:-1] for element in fields_data]
    fields_2D = [[a[which_point, :, :], a[:, which_point, :], a[:, :, which_point]][slice_axis]
                 for a
                 in fields_data_elements]
    # ex,ey,ez,epsilon
    return fields_2D


K_VEC = np.array([1, 1, 0])
POL_VEC = np.array([0, 0, 1])

beams = mlg.make_lg_beam_any_dir(K_VEC, POL_VEC, FCEN, WAVELENGTH, [sx, sy, sz], OBS_VOL, waist=WAIST,
                                 l=L, p=P)

sim = mp.Simulation(
    cell_size=cell_3d,
    sources=beams,
    boundary_layers=PML_LAYERS,
    resolution=RESOLUTION,
    geometry=[],
    default_material=MATERIAL,
    force_all_components=True,
    force_complex_fields=True
)

sim.run(until=20)

x, y, z, w = sim.get_array_metadata(center=mp.Vector3(), size=OBS_VOL)
[x, y, z] = [coordinate[1:-1] for coordinate in [x, y, z]]
Ex, Ey, Ez = get_fields(sim, SLICE_AXIS, CHOSEN_POINT)
e_squared = np.real((Ex * np.conjugate(Ex) + Ey * np.conjugate(Ey) + Ez * np.conjugate(Ez)))
plots_2D.append(e_squared)

# fig, ax = plt.subplots(1, 1, figsize=(12, 12))

plt.pcolormesh(x, y, np.transpose(plots_2D[0]),cmap='Spectral', alpha=1)




plt.show()
