"""Script simulating helicity lattice in vacuum using 4 plane waves.  """
import meep as mp
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import module_hg_beam as mhg
import plane_rotator as pr
import set_waves_module as sw
import matplotlib.pyplot as plt

DPML = 2  # thickness of PML layers
COMP_X, COMP_Y, COMP_Z = [10,10,10]  # dimensions of the computational cell, not including PML
SX, SY, SZ = COMP_X + 2 * DPML, COMP_Y + 2 * DPML, COMP_Z + 2 * DPML  # cell size, including PML
CELL = mp.Vector3(SX, SY, SZ)
OBS_VOL = mp.Vector3(8,8,8)
PML_LAYERS = [mp.PML(DPML)]
RESOLUTION = 8

FCEN = 2 / np.pi  # pulse center frequency
DF = 0.02  # turn-on bandwidth

WAIST = 10
WAVELENGTH = 1
M, N = 0, 0
########################################################################################################################
# K-VECTORS, E-VECTORS AND ROTATION
########################################################################################################################
C = 1
a1, a2, a3 = 1, 1, 1
T1, T2, T3 = 0, 0, 0
k_vectors, e_vectors = sw.make_3_wave_NI(C, T1, T2, T3, a1, a2, a3)
# rotating k vectors and e vectors
k_vectors, e_vectors = pr.find_angles_and_rotate(k_vectors, e_vectors, prp_to=2)
# k_vectors, e_vectors = pr.rotate_on_axis(k_vectors, e_vectors, np.pi / 4, 2)

########################################################################################################################
# SIMULATION
########################################################################################################################
T = 20  # run time

all_waves = mhg.make_multiple_hg_beams(k_vectors, e_vectors, FCEN, WAVELENGTH, [SX, SY, SZ], OBS_VOL, WAIST, m=M, n=N)

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

im = ax.pcolormesh(x, y, np.transpose(helicity_density), cmap='RdBu', alpha=1, vmin=-1, vmax=1)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(im, cax=cax)

plt.show()
