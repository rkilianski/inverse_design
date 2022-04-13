from matplotlib import pyplot as plt

import module_hg_beam as mhg
import meep as mp
import numpy as np

RESOLUTION = 8
ITERATIONS = 5

SLICE_AXIS = 2
CHOSEN_POINT = 20

DPML = 2  # thickness of perfectly matched layers (PMLs) around the box
PML_LAYERS = [mp.PML(DPML)]
DT = 5
T = 100
FCEN = 1

CELL_X, CELL_Y, CELL_Z = 10, 10, 10
OBS_X_A, OBS_Y_A, OBS_Z_A = 6, 6, 6  # dimensions of the computational cell, not including PML

sx = CELL_X + 2 * DPML
sy = CELL_Y + 2 * DPML
sz = CELL_Z + 2 * DPML

cell_3d = mp.Vector3(sx, sy, sz)
obs_vol = mp.Vector3(OBS_X_A, OBS_Y_A, OBS_Z_A)

SRC_POS_X, SRC_POS_Y, SRC_POS_Z = -3, 0, 0
obs_position_x, obs_position_y, obs_position_z = OBS_X_A / 2, OBS_Y_A / 2, 0

MATERIAL = mp.Medium(epsilon=1)
M, N = 0, 0
WAVELENGTH = 1
WAIST = 1


hg_beam = mhg.make_hg_beam(FCEN, WAVELENGTH, [0, sy, sz], [SRC_POS_X, SRC_POS_Y, SRC_POS_Z], [1, 0, 0], waist=WAIST,
                           m=M, n=N)
sim = mp.Simulation(
    cell_size=cell_3d,
    sources=hg_beam,
    boundary_layers=PML_LAYERS,
    resolution=RESOLUTION,
    geometry=[],
    default_material=MATERIAL,
    force_all_components=True,
    force_complex_fields=True
)

sim.run(until=20)


def get_fields(simulation, slice_axis, which_point):
    fields_data = [simulation.get_array(center=mp.Vector3(), size=obs_vol, component=field) for field in
                   [mp.Ex, mp.Ey, mp.Ez]]
    fields_data_elements = [element[1:-1, 1:-1, 1:-1] for element in fields_data]
    fields_2D = [[a[which_point, :, :], a[:, which_point, :], a[:, :, which_point]][slice_axis]
                 for a
                 in fields_data_elements]
    # ex,ey,ez,epsilon
    return fields_2D


x, y, z, w = sim.get_array_metadata(center=mp.Vector3(), size=obs_vol)
[x, y, z] = [coordinate[1:-1] for coordinate in [x, y, z]]
Ex, Ey, Ez = get_fields(sim, SLICE_AXIS, CHOSEN_POINT)
e_squared = np.transpose(np.real((Ex * np.conjugate(Ex) + Ey * np.conjugate(Ey) + Ez * np.conjugate(Ez))))

fig, ax = plt.subplots(2, 3, figsize=(12, 10))

ax[0, 0].pcolormesh(x, y, np.transpose(np.real(Ex)), cmap='RdBu', alpha=1)
ax[0, 0].set_title('Ex')

ax[0, 1].pcolormesh(x, y, np.transpose(np.real(Ey)), cmap='RdBu', alpha=1)
ax[0, 1].set_title('Ey')

ax[0, 2].pcolormesh(x, y, np.transpose(np.real(Ez)), cmap='RdBu', alpha=1)
ax[0, 2].set_title('Ez')

ax[1, 0].pcolormesh(x, y, np.real(e_squared), cmap='Spectral', alpha=1)
ax[1, 0].set_title('E squared')

plt.show()
