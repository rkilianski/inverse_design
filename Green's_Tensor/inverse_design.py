import meep as mp
import numpy as np

points = []

def df(old_field_arr, adj_field_arr):
    e1, e2, e3, eps1 = old_field_arr
    a1, a2, a3, eps2 = adj_field_arr
    merit_function = np.real(a1 * e1 + a2 * e2 + a3 * e3)
    return merit_function

def get_fields(simulation, ft):
    field_data = [simulation.get_dft_array(ft, component=i, num_freq=0) for i in components]
    Ex, Ey, Ez, eps_data = [a[1:-1, 1:-1] for a in field_data]
    return [Ex, Ey, Ez, eps_data]


def generate_forward_field(available_sources):
    simulations = []
    for j in range(x):
        for e_source in available_sources:
            sim = mp.Simulation(
                cell_size=CELL,
                sources=e_source,
                boundary_layers=PML_LAYERS,
                resolution=RESOLUTION,
                geometry=GEOMETRY,
                default_material=MATERIAL,
                force_all_components=True,
                force_complex_fields=True
            )

            dft_obj = sim.add_dft_fields(components, FCEN, FCEN, 1, where=slice_volume)
            sim.run(until=T)
            simulations.append(get_fields(sim, dft_obj))

    return simulations


def generate_adjoint_field(forward_field):
    dipole_at_obs = []
    for element in range(3):
        dipole_at_obs.append(mp.Source(
            mp.ContinuousSource(FCEN, width=DT, is_integrated=True),
            component=e_field_components[element],
            size=mp.Vector3(),
            center=mp.Vector3(x[X_OBS], y[Y_OBS]),
            amplitude=np.conjugate(forward_field[element][X_OBS, Y_OBS])))
    return dipole_at_obs


def add_block(first, second):
    block = mp.Block(
        center=mp.Vector3(first - PIXEL_SIZE / 2, second - PIXEL_SIZE / 2),
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
