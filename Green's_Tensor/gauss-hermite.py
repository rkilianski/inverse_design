import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy import special

RESOLUTION = 12
ITERATIONS = 5

SLICE_AXIS = 0
CHOSEN_POINT = 20

DPML = 2  # thickness of perfectly matched layers (PMLs) around the box
PML_LAYERS = [mp.PML(DPML)]
DT = 5
T = 100
FCEN = 1

CELL_X, CELL_Y, CELL_Z = 10, 10, 10
OBS_X_A, OBS_Y_A, OBS_Z_A = 10, 10, 10  # dimensions of the computational cell, not including PML

sx = CELL_X + 2 * DPML
sy = CELL_Y + 2 * DPML
sz = CELL_Z + 2 * DPML

cell_3d = mp.Vector3(sx, sy, sz)
obs_vol = mp.Vector3(OBS_X_A, OBS_Y_A, OBS_Z_A)

SOURCE_POSITION_X, SOURCE_POSITION_Y, SOURCE_POSITION_Z = 0, 0, 0
obs_position_x, obs_position_y, obs_position_z = OBS_X_A / 2, OBS_Y_A / 2, 0

MATERIAL = mp.Medium(epsilon=1)


def hermite_amp(x0, y0, w0, fcen, m=2, n=2):
    def _pw_amp(vec):
        prop_dir, x_h, y_h = vec
        x = x_h + x0
        y = y_h + y0
        lam = 2 * np.pi / fcen
        z_R = np.pi * (w0 ** 2) / lam
        w = w0 * np.sqrt(1 + prop_dir / z_R)
        h_n = special.hermite(n)
        h_m = special.hermite(m)
        h_function = 1e3*h_n(np.sqrt(2) * x / w) * h_m(np.sqrt(2) * y / (w ** 2)) * np.exp(
            1j * np.arctan(prop_dir / z_R) * (1 + n + m))

        return h_function

    return _pw_amp


source_hg = [mp.GaussianBeamSource(
    mp.ContinuousSource(frequency=FCEN),
    center=mp.Vector3(-OBS_X_A / 2, 0, 0),
    size=mp.Vector3(0, sy, sz),
    component=mp.ALL_COMPONENTS,
    beam_x0=mp.Vector3(0, 0, 0),
    beam_kdir=mp.Vector3(1, 0, 0),
    beam_w0=0.8,
    beam_E0=mp.Vector3(0, 0, 1),
    amp_func=hermite_amp(0, 0, 0.8, FCEN)),

]

sim = mp.Simulation(
    cell_size=cell_3d,
    sources=source_hg,
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
e_squared = np.real((Ex * np.conjugate(Ex) + Ey * np.conjugate(Ey) + Ez * np.conjugate(Ez)))

fig, ax = plt.subplots(2, 3, figsize=(12, 10))

ax[0, 0].pcolormesh(x, y, np.transpose(np.real(Ex)), cmap='RdBu', alpha=1, vmin=-0.003, vmax=0.003)
ax[0, 0].set_title('Ex')

ax[0, 1].pcolormesh(x, y, np.transpose(np.real(Ey)), cmap='RdBu', alpha=1, vmin=-0.005, vmax=0.005)
ax[0, 1].set_title('Ey')

ax[0, 2].pcolormesh(x, y, np.transpose(np.real(Ez)), cmap='RdBu', alpha=1, vmin=-0.3, vmax=0.3)
ax[0, 2].set_title('Ez')

ax[1, 0].pcolormesh(x, y, np.real(e_squared), cmap='Spectral', alpha=1, vmin=0, vmax=1)
ax[1, 0].set_title('E squared')

plt.show()
