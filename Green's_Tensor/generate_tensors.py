import meep as mp
import numpy as np
from joblib import Parallel, delayed

# In other file

SOURCE_POSITION_X = 0
SOURCE_POSITION_Y = 0
SOURCE_POSITION_Z = 0

SIM_BOX_1_X = -0.5
SIM_BOX_1_Y = 0.5

SIM_BOX_2_X = -0.5
SIM_BOX_2_Y = 0.5

SIM_BOX_3_X = -0.5
SIM_BOX_3_Y = 0.5

sim_box = [[SIM_BOX_1_X, SIM_BOX_1_Y], [SIM_BOX_2_X, SIM_BOX_2_Y], [SIM_BOX_3_X, SIM_BOX_3_Y]]
source_position = [SOURCE_POSITION_X, SOURCE_POSITION_Y, SOURCE_POSITION_Z]
DPML = 2  # thickness of perfectly matched layers (PMLs) around the box
w = np.pi  # angular frequency of emitter
gmtry = [mp.Block(mp.Vector3(3, 3, 3),
                  center=mp.Vector3(3, 3, 3),
                  material=mp.Medium(epsilon=1))]
PADDING = 2  # padding between the simulation box and the PML
RESOLUTION = 4


def cell_properties(sim_box, bound_size, pad):
    """Returns 3 Vector3 meep objects specifying:
    *the size of the simulation
    *the centre point of the simulation volume
    *the observational volume"""

    size_data = []
    cell_center = []
    obs_vol = []

    for i in range(3):
        size_data.append(2 * bound_size + pad + sim_box[i][1] - sim_box[i][0])
        cell_center.append((sim_box[i][1] + sim_box[i][0]) / 2)
        obs_vol.append(sim_box[i][1] - sim_box[i][0])

    s1, s2, s3 = size_data
    c1, c2, c3 = cell_center
    v1, v2, v3 = obs_vol

    return mp.Vector3(s1, s2, s3), mp.Vector3(c1, c2, c3), mp.Vector3(v1, v2, v3)


def t_row_j(row_nr, src_pos_in, omega, sim_box, boundary_thickness, res, geom, pad, e_field=True):
    """ Returns a row of a Tensor T, where the subscript row corresponds to i-th element of T_ij.
    Boolean parameter e_field lets user choose the output to be in terms of a Green's Tensor (by default)
    or Electric Field tensor.
     """
    PI = np.pi
    FCEN = omega / (2 * PI)
    DT = 5  # Temporal width of the Gaussian source

    pml_layers = [mp.PML(boundary_thickness)]

    # properties of the simulation cell
    size, center, volume = cell_properties(sim_box, boundary_thickness, pad)

    obs_comp = [mp.Ex, mp.Ey, mp.Ez]  # observation components which correspond the j's of the G_ij
    src_comp = [mp.Ex, mp.Ey, mp.Ez][row_nr]
    src_pos = mp.Vector3(src_pos_in[0], src_pos_in[1], src_pos_in[2])

    # creating a gaussian source and its fourier transform
    gauss_src = [mp.Source(mp.GaussianSource(FCEN, width=DT, is_integrated=True), component=src_comp, center=src_pos)]
    ft_gauss_src = mp.GaussianSource(FCEN, width=DT).fourier_transform(FCEN)

    # performing the simulation
    sim = mp.Simulation(cell_size=size,
                        geometry=geom,
                        geometry_center=center,
                        sources=gauss_src,
                        resolution=res,
                        boundary_layers=pml_layers,
                        force_all_components=True,
                        force_complex_fields=True,
                        eps_averaging=False)

    # setting up a volume where the fields are to be observed

    slice_vol = mp.Volume(center=center, size=volume)
    dft_obj = sim.add_dft_fields(obs_comp, FCEN, FCEN, 1, where=slice_vol)

    sim.run(until_after_sources=10)

    if not e_field:
        # dividing the source back out in order to arrive at the Green's function components
        t_tensor_row = np.array(
            [(1 / (1j * ft_gauss_src * omega)) * sim.get_dft_array(dft_obj, j, 0) for j in obs_comp])
    else:
        # returning Electric Field tensor
        t_tensor_row = np.array([sim.get_dft_array(dft_obj, j, 0) for j in obs_comp])

    return t_tensor_row


def grid_creator():
    size, center, volume = cell_properties(sim_box, DPML, PADDING)
    slice_vol = mp.Volume(center=center, size=volume)
    sim = mp.Simulation(cell_size=size,
                        geometry=gmtry,
                        geometry_center=center,
                        sources=[],
                        resolution=RESOLUTION,
                        force_all_components=True, force_complex_fields=True, eps_averaging=False
                        )
    sim.run(until=0)
    x, y, z, u = sim.get_array_metadata(slice_vol)

    return x, y, z


def t_tensor_constructor(geom, src_pos, omega, simbox, boundary_thickness, pad, res):
    t_no_coords = np.array(Parallel(n_jobs=-1, verbose=0)(delayed(t_row_j)(row_nr=row,
                                                                           src_pos_in=src_pos,
                                                                           omega=omega,
                                                                           sim_box=simbox,
                                                                           res=res,
                                                                           geom=geom,
                                                                           pad=pad,
                                                                           boundary_thickness=boundary_thickness,
                                                                           ) for row in [0, 1, 2]))
    x_grid, y_grid, z_grid = grid_creator()

    t_with_coords = []
    for i in range(len(x_grid)):
        for j in range(len(y_grid)):
            for k in range(len(z_grid)):
                t_with_coords.append([x_grid[i], y_grid[j], z_grid[k], t_no_coords[:, :, i, j, k]])
    return np.array(t_with_coords, dtype=object)


def tensor_comp_extract(x3_static_variable, tensor_data, real=True):
    """static_variable accepts strings: 'x','y','z'; when 'x' is chosen to be the observation point, the other
    'y' and 'z'  become x1 and x2 respectively. If the choice is different, the leftmost 'free' coordinate becomes x1.

    Function returns:
    the tensors at various (x1,x2) coordinates while x3 is kept fixed,
    array of tuples of (x1, x2) coordinates.
    The form of the tensors is a real or imaginary part of the complex entry. This is specified by the boolean real.

    """

    if x3_static_variable == "x":
        column = 0
        x1 = 1
        x2 = 2
    elif x3_static_variable == "y":
        column = 1
        x1 = 0
        x2 = 2
    else:
        column = 2
        x1 = 0
        x2 = 1

    x_3_static_position = tensor_data[0, column]

    tensor_list = []
    coordinates_x1x2 = []
    matrix_of_coordinates = tensor_data[:, 0:3]
    column_of_the_static_coordinate = matrix_of_coordinates[:, column]
    for row in range(len(column_of_the_static_coordinate)):
        if column_of_the_static_coordinate[row] == x_3_static_position:
            tensor_list.append(tensor_data[row, 3])
            pair_x1x2 = (matrix_of_coordinates[row, x1], matrix_of_coordinates[row, x2])
            coordinates_x1x2.append(pair_x1x2)

    if not real:
        tensor_list_complex_part = np.array([tensor.imag for tensor in tensor_list])
    else:
        tensor_list_complex_part = np.array([tensor.real for tensor in tensor_list])

    tensor_x1x2 = np.array(tensor_list_complex_part)

    return tensor_x1x2, coordinates_x1x2


# grid = t_row_j(0, source_position, w, sim_box, DPML, rez, gmtry, pad, False)

T = t_tensor_constructor(gmtry, source_position, w, sim_box, DPML, PADDING, RESOLUTION)

M, xy = tensor_comp_extract(x3_static_variable="z", tensor_data=T)
