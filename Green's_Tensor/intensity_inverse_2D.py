# Initialize starting geometry
import meep as mp
import numpy as np
from matplotlib import pyplot as plt

DPML = 1  # thickness of perfectly matched layers (PMLs) around the box
PML_LAYERS = [mp.PML(DPML)]
OBS_X_A, OBS_Y_A = 10, 10  # dimensions of the computational cell, not including PML
OBS_VOL = mp.Vector3(OBS_X_A, OBS_Y_A)
CELL_X, CELL_Y = 20, 20
CELL = mp.Vector3(CELL_X + 2 * DPML, CELL_Y + 2 * DPML)

SOURCE_POSITION_X, SOURCE_POSITION_Y = -OBS_X_A / 2, 0
OBS_POSITION_X, OBS_POSITION_Y = 1, 3

MATERIAL = mp.Medium(epsilon=1)
OMEGA = np.pi  # angular frequency of emitter

RESOLUTION = 8
PIXEL_SIZE = 1 / RESOLUTION
BLOCK_SIZE = 2 * PIXEL_SIZE
GEOMETRY = []
x_points = []
y_points = []
intensity_at_obs = []
intensity_at_source = []
components = [mp.Ex, mp.Ey, mp.Ez, mp.Dielectric]
e_field_components = [mp.Ex, mp.Ey, mp.Ez]
DT = 5
T = 100
FCEN = 10 / np.pi
ITERATIONS = 5
blocks_added = np.arange(ITERATIONS)


# Calculate the merit function

def df(old_field_arr, adj_field_arr):
    e1, e2, e3, eps1 = old_field_arr
    a1, a2, a3, eps2 = adj_field_arr
    merit_function = np.real(a1 * e1 + a2 * e2 + a3 * e3)
    return merit_function


# Produce a current source from one of the sides o the box

DIPOLE_AT_SOURCE = [
    # mp.Source(
    #     mp.ContinuousSource(FCEN, width=DT, is_integrated=True),
    #     component=mp.Ez,
    #     size=mp.Vector3(),
    #     center=mp.Vector3(-COMP_X/2, 0, 0)),
    mp.Source(
        mp.ContinuousSource(FCEN, width=DT, is_integrated=True),
        component=mp.Ez,
        size=mp.Vector3(),
        center=mp.Vector3(SOURCE_POSITION_X, SOURCE_POSITION_Y),
        amplitude=1),

    # mp.Source(
    #     mp.ContinuousSource(FCEN, width=DT, is_integrated=True),
    #     component=mp.Ez,
    #     size=mp.Vector3(),
    #     center=mp.Vector3(-COMP_X / 2, 0))
]
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
slice_volume = mp.Volume(center=mp.Vector3(), size=OBS_VOL)

dft_obj = sim.add_dft_fields(components, FCEN, FCEN, 1, where=slice_volume)
sim.run(until=T)

x, y, z, w = sim.get_array_metadata(center=mp.Vector3(0, 0, 0), size=OBS_VOL)
[x, y, z] = [coordinate[1:-1] for coordinate in [x, y, z]]


# slice_v = np.argmin((np.array([x, y, z][SLICE_AXIS]) - SLICE_POSITION) ** 2)

def produce_adjoint_field(forward_field):
    dipole_at_obs = []
    for element in range(3):
        dipole_at_obs.append(mp.Source(
            mp.ContinuousSource(FCEN, width=DT, is_integrated=True),
            component=e_field_components[element],
            size=mp.Vector3(),
            center=mp.Vector3(x[X_OBS], y[Y_OBS]),
            amplitude=np.conjugate(forward_field[element][X_OBS, Y_OBS])))
    return dipole_at_obs


def get_fields(simulation, ft):
    Ex, Ey, Ez, eps_data = [simulation.get_dft_array(ft, component=i, num_freq=0) for i in components]
    Ex, Ey, Ez, eps_data = [a[1:-1, 1:-1] for a in [Ex, Ey, Ez, eps_data]]
    # Ex, Ey, Ez = [[a[slice_v, :, :], a[:, slice_v, :], a[:, :, slice_v]][SLICE_AXIS] for a in [Ex, Ey, Ez]]
    return [Ex, Ey, Ez, eps_data]


old_field = get_fields(sim, dft_obj)
# Simulate a source from the obs point with amplitude of the E field from the observation point

X_OBS = np.where(np.abs(x - OBS_POSITION_X) < 0.2)[0][0]
Y_OBS = np.where(np.abs(y - OBS_POSITION_Y) < 0.2)[0][0]
Z_OBS = 0

X_SRC = np.where(np.abs(x - SOURCE_POSITION_X) < 0.2)[0][0]
Y_SRC = np.where(np.abs(y - SOURCE_POSITION_Y) < 0.2)[0][0]

DIPOLE_AT_OBS = produce_adjoint_field(old_field)

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
dft_adjoint = sim_adjoint.add_dft_fields(components, FCEN, FCEN, 1, where=slice_volume)
sim_adjoint.run(until=T)
adjoint_field = get_fields(sim_adjoint, dft_adjoint)
delta_f = df(old_field, adjoint_field)

########################################################################################################################
#                                 ADDING BLOCKS
########################################################################################################################
points = []


def add_block(first, second):
    block = mp.Block(
        center=mp.Vector3(first - PIXEL_SIZE / 2, second - PIXEL_SIZE / 2),
        size=mp.Vector3(BLOCK_SIZE, BLOCK_SIZE, mp.inf), material=mp.Medium(epsilon=1.3))
    GEOMETRY.append(block)
    return GEOMETRY


def exclude_points():
    for x_coord in x:
        for y_coord in y:
            if (x_coord - x[X_OBS]) ** 2 + (y_coord - y[Y_OBS]) ** 2 < 2:
                points.append((x_coord, y_coord))



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


x_points.append(x_inclusion)
y_points.append(y_inclusion)


def intensity_at_point(field, x, y):
    intensity_at_x0 = 0
    for i in range(3):
        intensity_at_x0 += field[i][x, y] * np.conjugate(field[i][x, y])
    return intensity_at_x0


geometry = add_block(x_inclusion, y_inclusion)

for i in range(ITERATIONS):
    sim = mp.Simulation(
        cell_size=CELL,
        sources=DIPOLE_AT_SOURCE,
        boundary_layers=PML_LAYERS,
        resolution=RESOLUTION,
        geometry=geometry,
        default_material=MATERIAL,
        force_all_components=True,
        force_complex_fields=True
    )

    dft_obj = sim.add_dft_fields(components, FCEN, FCEN, 1, where=slice_volume)
    sim.run(until=T)
    old_field = get_fields(sim, dft_obj)

    sim_adjoint = mp.Simulation(
        cell_size=CELL,
        sources=produce_adjoint_field(old_field),
        boundary_layers=PML_LAYERS,
        resolution=RESOLUTION,
        geometry=geometry,
        default_material=MATERIAL,
        force_all_components=True,
        force_complex_fields=True,

    )
    dft_adjoint = sim_adjoint.add_dft_fields(components, FCEN, FCEN, 1, where=slice_volume)
    sim_adjoint.run(until=T)
    adjoint_field = get_fields(sim_adjoint, dft_adjoint)

    # recording intensity at observation point after adding a block
    intensity_at_obs.append(intensity_at_point(old_field, X_OBS, Y_OBS))
    # intensity at the source point
    intensity_at_source.append(intensity_at_point(old_field, X_SRC, Y_SRC))
    #  Calculating the merit function df
    delta_f = df(old_field, adjoint_field)
    #  picking the coordinates corresponding to the highest change in df
    x1, y1 = pick_max(delta_f)
    #  updating the geometry
    geometry = add_block(x1, y1)

    x_points.append(x1)
    y_points.append(y1)

Ex, Ey, Ez, eps_data = old_field
Ex_a, Ey_a, Ez_a, eps_data_a = adjoint_field
eps_data = np.ma.masked_array(eps_data, eps_data < np.sqrt(1.4))

e_squared_a = np.real((Ex_a * np.conjugate(Ex_a) + Ey_a * np.conjugate(Ey_a) + Ez_a * np.conjugate(Ez_a)))
e_squared = np.real((Ex * np.conjugate(Ex) + Ey * np.conjugate(Ey) + Ez * np.conjugate(Ez)))
########################################################################################################################
#                                                   PLOTTING
########################################################################################################################


plt.scatter(x_points, y_points)
plt.show()

fig, ax = plt.subplots(2, 3, figsize=(12, 10))

ax[0, 0].pcolormesh(x, y, np.transpose(np.real(Ex)))
ax[0, 0].set_title('Ex')
ax[0, 0].pcolormesh(x, y, np.transpose(np.real(eps_data)), cmap='Greys', alpha=1, vmin=0, vmax=4)
ax[0, 0].plot(x[X_OBS], y[Y_OBS], 'ro')

ax[0, 1].pcolormesh(x, y, np.transpose(np.real(Ez)))
ax[0, 1].set_title('Ez')
ax[0, 1].pcolormesh(x, y, np.transpose(np.real(eps_data)), cmap='Greys', alpha=1, vmin=0, vmax=4)
ax[0, 1].plot(x[X_OBS], y[Y_OBS], 'ro')

ax[0, 2].plot(blocks_added, intensity_at_obs)
ax[0, 2].set_title('Intensity at x0 after adding a block')

i = 0

for ax, Si, name in zip([ax[1, 0], ax[1, 1], ax[1, 2]],
                        [e_squared, e_squared_a, delta_f],
                        ['E field Intensity', 'Adjoint field intensity', "Merit Function"]):
    S_ax = ax.pcolormesh(x, y, np.transpose(Si), vmax=1, vmin=0, cmap='RdYlBu')
    ax.pcolormesh(x, y, np.transpose(np.real(eps_data)), cmap='Greys', alpha=1, vmin=0, vmax=4)
    ax.set_title(name)
    cbar_ax = fig.add_axes([0.85, 0.1, 0.02, 0.5])
    fig.colorbar(S_ax, cax=cbar_ax)
fig.subplots_adjust(right=0.8)

plt.show()
