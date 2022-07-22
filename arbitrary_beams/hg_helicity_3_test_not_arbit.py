"""Script simulating helicity lattice in vacuum using 4 plane waves.  """
import meep as mp
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import module_hg_beam as mhg
import plane_rotator as pr
import set_waves_module as sw
import matplotlib.pyplot as plt

DPML = 2  # thickness of PML layers
COMP_X, COMP_Y, COMP_Z = [8,8,8]  # dimensions of the computational cell, not including PML
SX, SY, SZ = COMP_X + 2 * DPML, COMP_Y + 2 * DPML, COMP_Z + 2 * DPML  # cell size, including PML
CELL = mp.Vector3(SX, SY, SZ)
OBS_VOL = mp.Vector3(6, 6, 6)
PML_LAYERS = [mp.PML(DPML)]
RESOLUTION = 6

FCEN = 2/ np.pi  # pulse center frequency
DF = 0.02  # turn-on bandwidth

WAIST = 10
WAVELENGTH = 1
M, N = 0, 0
########################################################################################################################
# K-VECTORS, E-VECTORS AND ROTATION
########################################################################################################################

K1 = np.array([1, 0, 0])
K2 = np.array([0, 1, 0])
K3 = np.array([0, 0, 1])
E1 = np.array([0, 1, 0])
E2 = np.array([0, 0, 1])
E3 = np.array([1, 0, 0])
k_vectors = [K1, K2, K3]
e_vectors = [E1, E2, E3]

print(k_vectors)
# rotating k vectors and e vectors
k_vectors_r, e_vectors_r = pr.rotate_by_angle(k_vectors, e_vectors, np.pi / 4, 7 * np.pi / 4, prp_to=2)


########################################################################################################################
# SIMULATION
########################################################################################################################
T = 20  # run time

all_waves = mhg.make_multiple_hg_beams(k_vectors_r, e_vectors_r, FCEN, WAVELENGTH, [SX, SY, SZ], OBS_VOL, WAIST, m=M, n=N)

sim = mp.Simulation(
    cell_size=CELL,
    sources=all_waves,
    boundary_layers=PML_LAYERS,
    resolution=RESOLUTION,
    default_material=mp.Medium(index=N),
    force_complex_fields=True
)

sim.run(until=T)

########################################################################################################################
# PLOTS AND METADATA
########################################################################################################################

SLICE_POSITION = 20
SLICE_AXIS = 2

components = [mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz, mp.Dielectric]

Ex, Ey, Ez, Hx, Hy, Hz, eps_data = [sim.get_array(center=mp.Vector3(), size=OBS_VOL, component=i) for i in components]
x, y, z, w = sim.get_array_metadata(center=mp.Vector3(), size=OBS_VOL)

x = x[1:-1]
y = y[1:-1]
z = z[1:-1]

chosen_slice = np.argmin((np.array([x, y, z][SLICE_AXIS]) - SLICE_POSITION) ** 2)

Ex, Ey, Ez, Hx, Hy, Hz, eps_data = [a[1:-1, 1:-1, 1:-1] for a in [Ex, Ey, Ez, Hx, Hy, Hz, eps_data]]

Ex, Ey, Ez, Hx, Hy, Hz, eps_data = [[a[chosen_slice, :, :], a[:, chosen_slice, :], a[:, :, chosen_slice]][SLICE_AXIS]
                                    for a
                                    in [Ex, Ey, Ez, Hx, Hy, Hz, eps_data]]

intensityNorm = 1 / (Ex * np.conjugate(Ex) + Ey * np.conjugate(Ey) + Ez * np.conjugate(Ez))

e_sq = np.real((Ex * np.conjugate(Ex) + Ey * np.conjugate(Ey) + Ez * np.conjugate(Ez)))
h_sq = np.real((Hx * np.conjugate(Hx) + Hy * np.conjugate(Hy) + Hz * np.conjugate(Hz)))
helicity_density = np.imag(intensityNorm * (Ex * np.conjugate(Hx) + Ey * np.conjugate(Hy) + Ez * np.conjugate(Hz)))

fig, ax = plt.subplots(figsize=(12, 12))

im = ax.pcolormesh(x, y, helicity_density, cmap='RdBu', alpha=1, vmin=-1, vmax=1)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(im, cax=cax)
# ax.set_title(f'Helicity Density 4 plane waves')

# ax[1].pcolormesh(x, y, np.transpose(e_sq), cmap='OrRd', alpha=1)
# ax[1].set_title('Intensity')
#
# ax[2].pcolormesh(x, y, np.transpose(e_sq), cmap='RdPu', alpha=1)
# ax[2].set_title('H Squared')

plt.show()
