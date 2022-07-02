"""Script simulating helicity lattice in vacuum using 3 HG beams.  """
import meep as mp
import numpy as np
import module_lg_beam_any as mlg
import matplotlib.pyplot as plt

DPML = 2  # thickness of PML layers
COMP_X, COMP_Y, COMP_Z = [8, 8, 8]  # dimensions of the computational cell, not including PML
SX, SY, SZ = COMP_X + 2 * DPML, COMP_Y + 2 * DPML, COMP_Z + 2 * DPML  # cell size, including PML
CELL = mp.Vector3(SX, SY, SZ)
OBS_VOL = mp.Vector3(6, 6, 6)
PML_LAYERS = [mp.PML(DPML)]
RESOLUTION = 6

L, P = 1, 1
WAIST = 8
WAVELENGTH = 1.4
FCEN = 2 / np.pi  # pulse center frequency
DF = 0.02  # turn-on bandwidth
N = 1  # refractive index of material containing the source

########################################################################################################################
# K-VECTORS
########################################################################################################################
THETA = np.pi / 6
C = 1
K1 = C * np.array([1, 0, 0])
K2 = C * np.array([-np.cos(THETA), -np.sin(THETA), 1])
K3 = C * np.array([-np.cos(THETA), np.sin(THETA), 1])
K4 = C * np.array([np.cos(THETA), -np.sin(THETA), 1])

k_vectors = [K1, K2, K3, K4]

########################################################################################################################
# POLARISATION VECTORS
########################################################################################################################
delta_phi = 0
amp1 = 1
a3 = 1
amp2 = np.conjugate(amp1) * (a3 / np.conjugate(a3)) * np.exp(1j * delta_phi)
amp3 = a3
amp4 = a3 * np.exp(1j * delta_phi)

E1 = C * amp1 * np.array([-np.cos(THETA), -np.sin(THETA), 1])
E2 = C * amp2 * np.array([-np.cos(THETA), -np.sin(THETA), -1])
E3 = C * amp3 * np.array([np.sin(THETA), -np.cos(THETA), 1])
E4 = C * amp4 * np.array([-np.sin(THETA), np.cos(THETA), 1])

e_vectors = [E1, E2, E3, E4]

########################################################################################################################
# SIMULATION
########################################################################################################################
T = 20  # run time

all_waves = mlg.make_multiple_lg_beams(k_vectors, e_vectors, FCEN, WAVELENGTH, [SX, SY, SZ], OBS_VOL, WAIST, l=L, p=P)

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

fig, ax = plt.subplots(3, 3, figsize=(8, 12))

ax[0, 0].pcolormesh(x, y, np.transpose(helicity_density), cmap='RdYlBu', alpha=1, vmin=-1, vmax=1)
ax[0, 0].set_title(f'Helicity Density using beam LG{L}{P}')

ax[0, 1].pcolormesh(x, y, np.transpose(e_sq), cmap='OrRd', alpha=1)
ax[0, 1].set_title('Intensity')

ax[0, 2].pcolormesh(x, y, np.transpose(e_sq), cmap='RdPu', alpha=1)
ax[0, 2].set_title('H Squared')

plt.show()
