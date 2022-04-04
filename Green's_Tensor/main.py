import meep as mp
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import scipy as sp
from scipy import special
from matplotlib.pyplot import figure
import matplotlib.cm as cm

# Input parameters

geom = [mp.Block(mp.Vector3(3, 3, mp.inf),
                 center=mp.Vector3(3, 3),
                 material=mp.Medium(epsilon=1))]

src_pos = [0, 0, 0]
omega = 4
simbox = [6, 6, 10]
res = 5
boundary_thickness = 1
dim = 2

array_length = simbox[0] * res + boundary_thickness


def G(src_pos, omega, simbox, boundary_thickness, res, dim, geom):
    dpml = boundary_thickness  # Thickness of boundary layers which absorb outgoing waves

    resolution = res
    pixel_size = 1 / resolution
    piece_size = 3 * pixel_size

    sx = 2 * dpml + simbox[0]  # X length of simulation area
    sy = 2 * dpml + simbox[1]  # Y length of simulation area
    sz = 0

    cell_size = mp.Vector3(sx, sy, sz)  # Set up the whole volume (including bounding layers) in meep code
    pml_layers = [mp.PML(dpml)]  # Set up the boundary layers in meep code
    c = 1
    pi = np.pi
    fcen = omega / (2 * pi)  # Central frequency of the gaussian source
    dt = 5  # temporal width of the gaussian source

    sourcePos = mp.Vector3(src_pos[0], src_pos[1], src_pos[2])

    # ----Calculate the Green's tensor for a source at the given position----#

    srcList = [mp.Ex, mp.Ey, mp.Ez]

    [srcX, srcY, srcZ] = [[mp.Source(mp.GaussianSource(fcen, width=dt,
                                                       is_integrated=True),
                                     component=i, center=sourcePos)] for i in srcList]

    ft = mp.GaussianSource(fcen, width=dt).fourier_transform(fcen)
    geometry = geom

    simList = []
    for source_ in [srcX, srcY, srcZ]:
        simList.append(mp.Simulation(cell_size=mp.Vector3(sx, sy, sz),
                                     geometry=geometry,
                                     sources=source_,
                                     resolution=resolution,
                                     boundary_layers=pml_layers, force_all_components=True,
                                     force_complex_fields=True))

    sliceVol = mp.Volume(center=mp.Vector3(0, 0, 0),
                         size=mp.Vector3(sx - 2 * dpml, sy - 2 * dpml, sz - 2 * dpml))

    dft_obj_list = [i.add_dft_fields(srcList, fcen, fcen, 1, where=sliceVol) \
                    for i in simList]

    [i.run(until_after_sources=100) for i in simList]

    # (x,y,z,w)=simList[0].get_array_metadata(dft_cell=dft_obj_list[0])
    x = np.linspace(-(simbox[0] + pixel_size) / 2, (simbox[0] + pixel_size) / 2, array_length)
    y = np.linspace(-(simbox[1] + pixel_size) / 2, (simbox[1] + pixel_size) / 2, array_length)
    z = np.linspace(-(simbox[2] + pixel_size) / 2, (simbox[2] + pixel_size) / 2, array_length)

    # ----First index is observation polarisation specified by mp.E_i ,second is source polarisation given
    # by sim_i and dft_obj_i_Slice (which must be the same as each other, therefore are zipped)-----#

    g_data = np.array([[(1 / (1j * ft * omega)) * sim_i.get_dft_array(dft_i, j, 0)
                        for sim_i, dft_i in zip(simList, dft_obj_list)] for j in srcList])

    return [x, y, z, g_data]


G_sim = G(src_pos, omega, simbox, boundary_thickness, res, dim, geom)


def G_reshaper(G_data):
    g_matrices = G_data[-1]
    x = G_data[0]
    y = G_data[1]
    z = G_data[2]
    return np.array([
        [[[x[x_i], y[y_i],
           np.array([
               [
                   g_matrices[i][j][x_i][y_i]
                   for j in [0, 1, 2]]
               for i in [0, 1, 2]])]]
         for y_i in range(len(y))]
        for x_i in range(len(x))
    ]).reshape(-1, 3)


Greens_and_coords = np.array(G_reshaper(G_sim))


# =============================================================================
#
# Extracting the Matrix components of a Green's tensor
#
# =============================================================================

# Function extracting a column of values for an xy entry of the Green's tensor

def G_comp_extract(x, y):
    G_tensor_list = Greens_and_coords[:, 2]

    G_tensor_list_real = np.array([tensor.real for tensor in G_tensor_list])

    G_xy = np.array([G_tensor_list_real[i][x, y] \
                     for i in range(0, array_length * array_length)])
    return G_xy


# =============================================================================
# CALCULATING ANALYTICAL SOLUTION WITH HANKEL FUNCTIONS
# =============================================================================

# coordinates (x,y,z)
Coords = Greens_and_coords[:, 0:2]

# coordinate magnitudes r' - r (Rho), to calculate G_zz
x_coords_y_0 = G_sim[0]
len_x = len(x_coords_y_0)

# G_zz values where y = 0
G_zz_vals = np.array([Greens_and_coords[i, 2][2, 2].real for i in \
                      range(0, len(Greens_and_coords[:, 1])) \
                      if Greens_and_coords[i, 1] == 0])  # this is clutter, CHANGE!

# H =  H(omega*Rho)
Hankel_input = np.multiply(omega, x_coords_y_0)

han = sp.special.hankel1(0, Hankel_input)
# Analytical solution for G_zz
analytical_sol = -0.25 * han.imag

plt.figure(dpi=200)
plt.plot(x_coords_y_0, analytical_sol, color='green', label="Analytic solution",
         linestyle='dotted')
plt.plot(x_coords_y_0, G_zz_vals, color='blue', label="Simulation"
         )
plt.title("Plot of G_zz where y = 0, with a dipole aligned with z ")
plt.xlabel('x coordinate')
plt.legend(loc="upper right")
plt.show()

# =============================================================================
# PLOTTING ELEMENTS OF G
# =============================================================================

x = np.array(G_sim[0])
y = np.array(G_sim[1])

nrows, ncols = array_length, array_length

plt.figure(figsize=(20, 20))
count = 0
for i in [0, 1, 2]:
    for j in [0, 1, 2]:
        count += 1
        grid_ij = G_comp_extract(i, j).reshape((nrows, ncols))

        plt.subplot(3, 3, count)
        plt.imshow(grid_ij, extent=(x.min(), x.max(), y.max(), y.min()),
                   interpolation='nearest', cmap=cm.gist_rainbow, vmin=-0.1, vmax=0.1)
        plt.colorbar(orientation="vertical")

plt.show()
