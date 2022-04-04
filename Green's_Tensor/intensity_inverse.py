# Initialize starting geometry
import meep as mp
import numpy as np
from matplotlib import pyplot as plt
import module_3d_wave as m3d

DPML = 2  # thickness of perfectly matched layers (PMLs) around the box
COMP_X, COMP_Y, COMP_Z = [6, 6, 6]  # dimensions of the computational cell, not including PML
COMP_VOL = mp.Vector3(COMP_X, COMP_Y, COMP_Z)
SX, SY, SZ = COMP_X + 2 * DPML, COMP_Y + 2 * DPML, COMP_Z + 2 * DPML  # cell size, including PML
SOURCE_POSITION_X, SOURCE_POSITION_Y, SOURCE_POSITION_Z = -SX / 2, 0, 0
OBS_POSITION_X, OBS_POSITION_Y, OBS_POSITION_Z = 0, -COMP_Y / 2, 0
CELL = mp.Vector3(SX, SY, SZ)
MATERIAL = mp.Medium(epsilon=1)
OMEGA = np.pi  # angular frequency of emitter
PML_LAYERS = [mp.PML(DPML)]
RESOLUTION = 4
PIXEL_SIZE = 1 / RESOLUTION
BLOCK_SIZE = 2 * PIXEL_SIZE
GEOMETRY = []
x_points = []
y_points = []

DT = 5
T = 100
FCEN = 2 / np.pi

SLICE_POSITION = 20
SLICE_AXIS = 2

k_vectors = [[1, 0, 0]]
polarisations = [[0, 0, 1]]




# Calculate the merit function

def df(old_field_arr, adj_field_arr):
    e1, e2, e3, eps = old_field_arr
    a1, a2, a3, eps = adj_field_arr
    merit_function = np.real(a1 * e1 + a2 * e2 + a3 * e3)
    return merit_function


# Produce a current source from one of the sides o the box

DIPOLE_AT_SOURCE = m3d.make_3d_wave(k_vectors, polarisations, FCEN, DT, [SX, SY, SZ], [COMP_X, COMP_Y, COMP_Z], 1)
sim = mp.Simulation(
    cell_size=CELL,
    sources=DIPOLE_AT_SOURCE,
    boundary_layers=PML_LAYERS,
    resolution=RESOLUTION,
    geometry=GEOMETRY,
    default_material=MATERIAL,
    force_all_components=True,
    force_complex_fields=True
)
slice_volume = mp.Volume(center=mp.Vector3(), size=COMP_VOL)

dft_obj = sim.add_dft_fields([mp.Ex, mp.Ey, mp.Ez, mp.Dielectric], FCEN, FCEN, 1, where=slice_volume)
sim.run(until=T)

x, y, z, w = sim.get_array_metadata(center=mp.Vector3(0, 0, 0), size=COMP_VOL)
[x, y, z] = [coordinate[1:-1] for coordinate in [x, y, z]]
slice_v = np.argmin((np.array([x, y, z][SLICE_AXIS]) - SLICE_POSITION) ** 2)


def get_fields(ft):
    Ex, Ey, Ez, eps = [sim.get_dft_array(ft, component=i, num_freq=0) for i in [mp.Ex, mp.Ey, mp.Ez, mp.Dielectric]]
    Ex, Ey, Ez, eps = [a[1:-1, 1:-1, 1:-1] for a in [Ex, Ey, Ez, eps]]
    Ex, Ey, Ez, eps = [[a[slice_v, :, :], a[:, slice_v, :], a[:, :, slice_v]][SLICE_AXIS] for a in [Ex, Ey, Ez, eps]]
    return [Ex, Ey, Ez, eps]


old_field = get_fields(dft_obj)
# Simulate a source from the obs point with amplitude of the E field from the observation point

X_OBS = np.where(np.abs(x - OBS_POSITION_X) < 0.2)[0][0]
Y_OBS = np.where(np.abs(y - OBS_POSITION_Y) < 0.2)[0][0]
Z_OBS = np.where(np.abs(z - OBS_POSITION_Z) < 0.2)[0][0]

DIPOLE_AT_OBS = [

    mp.Source(
        mp.ContinuousSource(FCEN, width=DT, is_integrated=True),
        component=mp.Ex,
        size=mp.Vector3(),
        center=mp.Vector3(OBS_POSITION_X, OBS_POSITION_Y, OBS_POSITION_Z),
        amplitude=np.conjugate(old_field[0][X_OBS, Y_OBS])),

    mp.Source(
        mp.ContinuousSource(FCEN, width=DT, is_integrated=True),
        component=mp.Ey,
        size=mp.Vector3(),
        center=mp.Vector3(OBS_POSITION_X, OBS_POSITION_Y, OBS_POSITION_Z),
        amplitude=np.conjugate(old_field[1][X_OBS, Y_OBS])),

    mp.Source(
        mp.ContinuousSource(FCEN, width=DT, is_integrated=True),
        component=mp.Ez,
        size=mp.Vector3(),
        center=mp.Vector3(OBS_POSITION_X, OBS_POSITION_Y, OBS_POSITION_Z),
        amplitude=np.conjugate(old_field[2][X_OBS, Y_OBS])),
]

sim_adjoint = mp.Simulation(
    cell_size=CELL,
    sources=DIPOLE_AT_OBS,
    boundary_layers=PML_LAYERS,
    resolution=RESOLUTION,
    geometry=GEOMETRY,
    default_material=MATERIAL,
    force_all_components=True,
    force_complex_fields=True,

)
dft_adjoint = sim_adjoint.add_dft_fields([mp.Ex, mp.Ey, mp.Ez, mp.Dielectric], FCEN, FCEN, 1, where=slice_volume)
sim_adjoint.run(until=T)
adjoint_field = get_fields(dft_adjoint)

delta_f = df(old_field, adjoint_field)


########################################################################################################################
#                                 ADDING BLOCKS
########################################################################################################################
points = []


def add_block(first, second):
    block = mp.Block(
        center=mp.Vector3(first - PIXEL_SIZE / 2, second - PIXEL_SIZE / 2, 0),
        size=mp.Vector3(BLOCK_SIZE, BLOCK_SIZE), material=mp.Medium(epsilon=1.3))
    GEOMETRY.append(block)
    return GEOMETRY


def delete_existing(arr):
    for tup in points:
        arr[tup[0], tup[1]] = 0
    return arr


def pick_max(delta):
    """Returns a pair of points (x,y) corresponding to the highest value of the merit function."""
    if len(points) > 0:
        delta = delete_existing(delta)
    max_x, max_y = np.unravel_index(delta.argmax(), delta.shape)
    points.append((max_x, max_y))

    return x[max_x], y[max_y]


x_inclusion, y_inclusion = pick_max(delta_f)
add_block(x_inclusion, y_inclusion)

x_points.append(x_inclusion)
y_points.append(y_inclusion)

for i in range(10):
    sim = mp.Simulation(
        cell_size=CELL,
        sources=DIPOLE_AT_SOURCE,
        boundary_layers=PML_LAYERS,
        resolution=RESOLUTION,
        geometry=GEOMETRY,
        default_material=MATERIAL,
        force_all_components=True,
        force_complex_fields=True
    )

    dft_obj = sim.add_dft_fields([mp.Ex, mp.Ey, mp.Ez, mp.Dielectric], FCEN, FCEN, 1, where=slice_volume)
    sim.run(until=T)
    old_field = get_fields(dft_obj)

    DIPOLE_AT_OBS = [

        mp.Source(
            mp.ContinuousSource(FCEN, width=DT, is_integrated=True),
            component=mp.Ex,
            size=mp.Vector3(),
            center=mp.Vector3(OBS_POSITION_X, OBS_POSITION_Y, OBS_POSITION_Z),
            amplitude=np.conjugate(old_field[0][X_OBS, Y_OBS])),

        mp.Source(
            mp.ContinuousSource(FCEN, width=DT, is_integrated=True),
            component=mp.Ey,
            size=mp.Vector3(),
            center=mp.Vector3(OBS_POSITION_X, OBS_POSITION_Y, OBS_POSITION_Z),
            amplitude=np.conjugate(old_field[1][X_OBS, Y_OBS])),

        mp.Source(
            mp.ContinuousSource(FCEN, width=DT, is_integrated=True),
            component=mp.Ez,
            size=mp.Vector3(),
            center=mp.Vector3(OBS_POSITION_X, OBS_POSITION_Y, OBS_POSITION_Z),
            amplitude=np.conjugate(old_field[2][X_OBS, Y_OBS])),
    ]

    sim_adjoint = mp.Simulation(
        cell_size=CELL,
        sources=DIPOLE_AT_OBS,
        boundary_layers=PML_LAYERS,
        resolution=RESOLUTION,
        geometry=GEOMETRY,
        default_material=MATERIAL,
        force_all_components=True,
        force_complex_fields=True,

    )
    dft_adjoint = sim_adjoint.add_dft_fields([mp.Ex, mp.Ey, mp.Ez, mp.Dielectric], FCEN, FCEN, 1, where=slice_volume)
    sim_adjoint.run(until=T)
    adjoint_field = get_fields(dft_adjoint)

    #  Calculating the merit function df
    delta_f = df(old_field, adjoint_field)
    #  picking the coordinates corresponding to the highest change in df
    x1, y1 = pick_max(delta_f)
    #  updating the geometry
    add_block(x1, y1)

    x_points.append(x1)
    y_points.append(y1)
########################################################################################################################
#                                                   PLOTTING
########################################################################################################################
Ex, Ey, Ez, eps_data = old_field
Ex_a, Ey_a, Ez_a, eps_data_a = adjoint_field
eps_data = np.ma.masked_array(eps_data, eps_data < np.sqrt(1.4))

e_squared_a = np.real((Ex_a * np.conjugate(Ex_a) + Ey_a * np.conjugate(Ey_a) + Ez_a * np.conjugate(Ez_a)))
e_squared = np.real((Ex * np.conjugate(Ex) + Ey * np.conjugate(Ey) + Ez * np.conjugate(Ez)))

plt.scatter(x_points, y_points)
plt.show()

fig, ax = plt.subplots(2, 3, figsize=(12, 10))

ax[0, 0].pcolormesh(x, y, np.transpose(np.real(Ex)))
ax[0, 0].set_title('Ex')
ax[0, 0].pcolormesh(x, y, np.transpose(np.real(eps_data)), cmap='Greys', alpha=1, vmin=0, vmax=4)

ax[0, 1].pcolormesh(x, y, np.transpose(np.real(Ey)))
ax[0, 1].set_title('Ey')
ax[0, 1].pcolormesh(x, y, np.transpose(np.real(eps_data)), cmap='Greys', alpha=1, vmin=0, vmax=4)

ax[0, 2].pcolormesh(x, y, np.transpose(e_squared))
ax[0, 2].set_title('Intensity')
ax[0, 2].pcolormesh(x, y, np.transpose(np.real(eps_data)), cmap='Greys', alpha=1, vmin=0, vmax=4)

i = 0

for ax, Si, name in zip([ax[1, 0], ax[1, 1], ax[1, 2]],
                        [e_squared, e_squared_a, delta_f],
                        ['E field Intensity', 'Adjoint field intensity', "Merit Function"]):
    S_ax = ax.pcolormesh(x, y, np.transpose(Si), vmax=1, vmin=-1, cmap='RdYlBu')
    ax.pcolormesh(x, y, np.transpose(np.real(eps_data)), cmap='Greys', alpha=1, vmin=0, vmax=4)
    ax.set_title(name)
    cbar_ax = fig.add_axes([0.85, 0.1, 0.02, 0.5])
    fig.colorbar(S_ax, cax=cbar_ax)
fig.subplots_adjust(right=0.8)

plt.show()
