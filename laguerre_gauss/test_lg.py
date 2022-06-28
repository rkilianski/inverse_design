from matplotlib import pyplot as plt

import module_lg_beam as mlg
import module_hg_beam as mhg
import meep as mp
import numpy as np

RESOLUTION = 8
SLICE_AXIS = 0
CHOSEN_POINT = 20

DPML = 2  # thickness of perfectly matched layers (PMLs) around the box
PML_LAYERS = [mp.PML(DPML)]
DT = 5
T = 20
FCEN = 1

CELL_X, CELL_Y, CELL_Z = 10, 10, 10
OBS_X_A, OBS_Y_A, OBS_Z_A = 8, 8, 8  # dimensions of the computational cell, not including PML

sx = CELL_X + 2 * DPML
sy = CELL_Y + 2 * DPML
sz = CELL_Z + 2 * DPML

cell_3d = mp.Vector3(sx, sy, sz)
obs_vol = mp.Vector3(OBS_X_A, OBS_Y_A, OBS_Z_A)

SRC_POS_X, SRC_POS_Y, SRC_POS_Z = -3, 0, 0

MATERIAL = mp.Medium(epsilon=1)
L, P = 2, 0
WAVELENGTH = 1
WAIST = 1


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


# for i in range(1, 3):
#     for j in range(1, 3):
#         beams.append(
#             mlg.make_lg_beam(FCEN, WAVELENGTH, [0, sy, sz], [SRC_POS_X, SRC_POS_Y, SRC_POS_Z], dir_prop=0, waist=WAIST,
#                              m=i, n=j))
#
# for i in range(4):
#     sim = mp.Simulation(
#         cell_size=cell_3d,
#         sources=beams[i],
#         boundary_layers=PML_LAYERS,
#         resolution=RESOLUTION,
#         geometry=[],
#         default_material=MATERIAL,
#         force_all_components=True,
#         force_complex_fields=True
#     )

beams = mlg.make_lg_beam(FCEN, WAVELENGTH, [0, sy, sz], [SRC_POS_X, SRC_POS_Y, SRC_POS_Z], dir_prop=0, waist=WAIST,
                             m=L, n=P)

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

sim.run(until=T)

x, y, z, w = sim.get_array_metadata(center=mp.Vector3(), size=obs_vol)
[x, y, z] = [coordinate[1:-1] for coordinate in [x, y, z]]
Ex, Ey, Ez = get_fields(sim, SLICE_AXIS, CHOSEN_POINT)
e_squared = np.real((Ex * np.conjugate(Ex) + Ey * np.conjugate(Ey) + Ez * np.conjugate(Ez)))
amp_squared = np.sqrt(Ex*Ex + Ey*Ey + Ez*Ez)
arg = np.angle(amp_squared)
plots_2D.append(e_squared)

fig, ax = plt.subplots(2, 2, figsize=(12, 12))

ax[0, 0].pcolormesh(x, y, arg, cmap='Spectral', alpha=1)
ax[0, 0].set_title(f'LG{L}{P}')

ax[0, 1].pcolormesh(x, y, plots_2D[0], cmap='Spectral', alpha=1)
ax[0, 1].set_title('TEM01')

ax[1, 0].pcolormesh(x, y, plots_2D[0], cmap='Spectral', alpha=1)
ax[1, 0].set_title('TEM02')

ax[1, 1].pcolormesh(x, y, plots_2D[0], cmap='Spectral', alpha=1)
ax[1, 1].set_title('TEM10')

# ax[1, 1].pcolormesh(x, y, plots_2D[4], cmap='Spectral', alpha=1)
# ax[1, 1].set_title('TEM11')
#
# ax[1, 2].pcolormesh(x, y, plots_2D[5], cmap='Spectral', alpha=1)
# ax[1, 2].set_title('TEM12')
#
# ax[2, 0].pcolormesh(x, y, plots_2D[6], cmap='Spectral', alpha=1)
# ax[2, 0].set_title('TEM20')
#
# ax[2, 1].pcolormesh(x, y, plots_2D[7], cmap='Spectral', alpha=1)
# ax[2, 1].set_title('TEM21')
#
# ax[2, 2].pcolormesh(x, y, plots_2D[8], cmap='Spectral', alpha=1)
# ax[2, 2].set_title('TEM22')
#
#
plt.show()
