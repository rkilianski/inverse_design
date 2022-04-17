# Initialize starting geometry
import meep as mp
import numpy as np
from matplotlib import pyplot as plt

DPML = 2  # thickness of perfectly matched layers (PMLs) around the box
PML_LAYERS = [mp.PML(DPML)]
DT = 5
T = 100
FCEN = 2 / np.pi
ITERATIONS = 10

OBS_X_A, OBS_Y_A = 10, 10  # dimensions of the computational cell, not including PML
OBS_VOL = mp.Vector3(OBS_X_A, OBS_Y_A)
CELL_X, CELL_Y = 20, 20
CELL = mp.Vector3(CELL_X + 2 * DPML, CELL_Y + 2 * DPML)

SOURCE_POSITION_X, SOURCE_POSITION_Y = -OBS_X_A / 2, 0
OBS_POSITION_X, OBS_POSITION_Y = 0, 0

MATERIAL = mp.Medium(epsilon=1)
OMEGA = np.pi  # angular frequency of emitter

RESOLUTION = 8
PIXEL_SIZE = 1 / RESOLUTION
BLOCK_SIZE = 3 * PIXEL_SIZE
GEOMETRY = []
x_points = []
y_points = []
intensity_at_obs = []
intensity_at_source = []

components = [mp.Ex, mp.Ey, mp.Ez, mp.Dielectric]
e_field_components = [mp.Ex, mp.Ey, mp.Ez]

blocks_added = np.arange(ITERATIONS)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# Calculate the merit function

def df(old_field_arr, adj_field_arr):
    e1, e2, e3, eps1 = old_field_arr
    a1, a2, a3, eps2 = adj_field_arr
    merit_function = np.real(a1 * e1 + a2 * e2)
    norm = 1 / (np.amax(merit_function))
    return norm * merit_function


# Produce a current source from one of the sides o the box

def pw_amp(k, x0, delta):
    def _pw_amp(x):
        return delta * np.exp(1j * k.dot(x + x0))

    return _pw_amp


DF = 0.02  # turn-on bandwidth


def pw_maker(freq, kDir, pol, srcPos, srcBox, complex_phase=1):
    kDir = mp.Vector3(kDir[0], kDir[1])
    kVec = kDir.unit().scale(2 * np.pi * freq)
    polNormed = pol / np.sqrt(np.dot(pol, np.conjugate(pol)))
    xPart = mp.Source(mp.ContinuousSource(FCEN, fwidth=DF), component=mp.Ex,
                      center=srcPos, size=srcBox,
                      amp_func=pw_amp(kVec, mp.Vector3(SOURCE_POSITION_X, SOURCE_POSITION_Y), delta=polNormed[0]))
    yPart = mp.Source(mp.ContinuousSource(FCEN, fwidth=DF), component=mp.Ey,
                      center=srcPos, size=srcBox,
                      amp_func=pw_amp(kVec, mp.Vector3(SOURCE_POSITION_X, SOURCE_POSITION_Y), delta=polNormed[1]))
    return xPart, yPart


k1 = [1, 0]

sourceLineX = mp.Vector3(0, CELL_Y)

# Plane wave source for input wave


DIPOLE_AT_SOURCE = pw_maker(freq=FCEN, kDir=[1, 0, 0], pol=[0, 1, 0], srcPos=[SOURCE_POSITION_X, SOURCE_POSITION_Y],
                            srcBox=sourceLineX)

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

sim.run(until=T)

x, y, z, w = sim.get_array_metadata(center=mp.Vector3(), size=OBS_VOL)
[x, y, z] = [coordinate[1:-1] for coordinate in [x, y, z]]


def produce_adjoint_field(forward_field):
    dipole_at_obs = []
    for element in range(3):
        dipole_at_obs.append(mp.Source(
            mp.ContinuousSource(FCEN, width=DT, is_integrated=True),
            component=e_field_components[element],
            size=mp.Vector3(),
            center=mp.Vector3(x[x_obs_index], y[y_obs_index]),
            amplitude=np.conjugate(forward_field[element][x_obs_index, y_obs_index])))
    return dipole_at_obs


def get_fields(simulation):
    fields_data = [simulation.get_array(center=mp.Vector3(), size=OBS_VOL, component=field) for field in components]
    fields_data_elements = [a[1:-1, 1:-1] for a in fields_data]
    # ex,ey,ez,epsilon
    return fields_data_elements


old_field = get_fields(sim)
# Simulate a source from the obs point with amplitude of the E field from the observation point

x_obs_index = find_nearest(x, OBS_POSITION_X)
y_obs_index = find_nearest(y, OBS_POSITION_Y)

x_src_index = find_nearest(x, SOURCE_POSITION_X)
y_src_index = find_nearest(y, SOURCE_POSITION_Y)

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

sim_adjoint.run(until=T)

adjoint_field = get_fields(sim_adjoint)

delta_f = df(old_field, adjoint_field)

########################################################################################################################
#                                 ADDING BLOCKS
########################################################################################################################
points = []


def add_block(first, second):
    block = mp.Block(
        center=mp.Vector3(first, second),
        size=mp.Vector3(BLOCK_SIZE, BLOCK_SIZE, mp.inf), material=mp.Medium(epsilon=1.3))
    GEOMETRY.append(block)
    return GEOMETRY


def exclude_points():
    for x_coord in x:
        for y_coord in y:
            if (x_coord - x[x_obs_index]) ** 2 + (y_coord - y[y_obs_index]) ** 2 < 1 or (
                    x_coord - x[x_src_index]) ** 2 + (y_coord - y[y_src_index]) ** 2 < 1:
                x_index = np.where(x == x_coord)[0][0]
                y_index = np.where(y == y_coord)[0][0]
                points.append((x_index, y_index))


exclude_points()


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


delta_f[x_obs_index:, :] = np.zeros((len(x) - x_obs_index, len(y)))
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

    sim.run(until=T)
    old_field = get_fields(sim)

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

    sim_adjoint.run(until=T)
    adjoint_field = get_fields(sim_adjoint)

    # recording intensity at observation point after adding a block
    intensity_at_obs.append(intensity_at_point(old_field, x_obs_index, y_obs_index))
    # intensity at the source point
    intensity_at_source.append(intensity_at_point(old_field, x_src_index, y_src_index))
    #  Calculating the merit function df
    delta_f = df(old_field, adjoint_field)
    delta_f[x_obs_index:, :] = np.zeros((len(x) - x_obs_index, len(y)))
    delta_f[:5, :] = np.zeros((5, len(y)))
    #  picking the coordinates corresponding to the highest change in df
    x1, y1 = pick_max(delta_f)
    #  updating the geometry

    geometry = add_block(x1, y1)

    x_points.append(x1)
    y_points.append(y1)

    intensity_ratio = np.array(intensity_at_obs) / np.array(intensity_at_source[0])
    intensity_ratio_obs_obs = np.array(intensity_at_obs) / np.array(intensity_at_obs[0])

Ex, Ey, Ez, eps_data = old_field
Ex_a, Ey_a, Ez_a, eps_data_a = adjoint_field
eps_data = np.ma.masked_array(eps_data, eps_data < np.sqrt(1.4))

e_squared_a = np.real((Ex_a * np.conjugate(Ex_a) + Ey_a * np.conjugate(Ey_a) + Ez_a * np.conjugate(Ez_a)))
e_squared_a = (1 / np.amax(e_squared_a)) * e_squared_a

e_squared = np.real((Ex * np.conjugate(Ex) + Ey * np.conjugate(Ey) + Ez * np.conjugate(Ez)))
e_squared = (1 / np.amax(e_squared)) * e_squared

########################################################################################################################
#                                                   PLOTTING
########################################################################################################################


plt.scatter(x_points, y_points)
plt.show()

fig, ax = plt.subplots(2, 3, figsize=(12, 10))

ax[0, 0].pcolormesh(x, y, np.transpose(np.real(Ex)))
ax[0, 0].set_title('Ex')
ax[0, 0].pcolormesh(x, y, np.transpose(np.real(eps_data)), cmap='Greys', alpha=1, vmin=0, vmax=4)
ax[0, 0].plot(x[x_obs_index], y[y_obs_index], 'ro')

ax[0, 1].pcolormesh(x, y, np.transpose(np.real(Ey)))
ax[0, 1].set_title('Ey')
ax[0, 1].pcolormesh(x, y, np.transpose(np.real(eps_data)), cmap='Greys', alpha=1, vmin=0, vmax=4)
ax[0, 1].plot(x[x_obs_index], y[y_obs_index], 'ro')

# ax[0,1].plot(blocks_added, intensity_ratio_obs_obs)
# ax[0, 1].set_title('(I at obs)/(I at obs at t=0) after adding a block')

ax[0, 2].plot(blocks_added, intensity_at_obs)
ax[0, 2].set_title('Intensity after adding a block')

i = 0

for ax, Si, name in zip([ax[1, 0], ax[1, 1], ax[1, 2]],
                        [e_squared, e_squared_a, delta_f],
                        ['E field Intensity', 'Adjoint field intensity', "Merit Function"]):
    S_ax = ax.pcolormesh(x, y, np.transpose(Si), cmap='RdYlBu')
    ax.pcolormesh(x, y, np.transpose(np.real(eps_data)), cmap='Greys', alpha=1, vmin=0, vmax=4)
    ax.set_title(name)
    cbar_ax = fig.add_axes([0.85, 0.1, 0.02, 0.5])
    fig.colorbar(S_ax, cax=cbar_ax)
fig.subplots_adjust(right=0.8)

plt.show()
