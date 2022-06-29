"""Script simulating helicity lattice in vacuum using 4 plane waves. This setup is an example of a lattice exhibiting
 a property of superchirality, regular regions in the plane that reach the max/min value of helicity.  """
import meep as mp
import numpy as np
import module_hg_beam as mhg
import matplotlib.pyplot as plt

DPML = 2  # thickness of PML layers
COMP_X, COMP_Y, COMP_Z = [8, 8, 8]  # dimensions of the computational cell, not including PML
SX, SY, SZ = COMP_X + 2 * DPML, COMP_Y + 2 * DPML, COMP_Z + 2 * DPML  # cell size, including PML
CELL = mp.Vector3(SX, SY, SZ)
OBS_VOL = mp.Vector3(6, 6, 6)
VX, VY, VZ = OBS_VOL
PML_LAYERS = [mp.PML(DPML)]
RESOLUTION = 6

WAIST = 10
WAVELENGTH = 1.5
FCEN = 2 / np.pi  # pulse center frequency
DF = 0.02  # turn-on bandwidth
N = 1  # refractive index of material containing the source

########################################################################################################################
# K-VECTORS
########################################################################################################################
THETA = np.pi / 6
C = np.sqrt(2) / 2
K1 = C * np.array([np.cos(THETA), np.sin(THETA), 1])
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
# test bits
# K_T = np.array([1, 0, 0])
# E_T = np.array([0, 0, 1])
# wave_T = mhg.make_hg_beam_any_dir(K_T, E_T, FCEN, WAVELENGTH, [SX, SY, SZ], [-3, 0, 0], WAIST, m=0, n=0)
########################################################################################################################
# TWO WAVES
########################################################################################################################
K1 = np.array([np.sin(THETA), np.cos(THETA), 0])
K2 = np.array([-np.sin(THETA), np.cos(THETA), 0])

E1 = np.array([0, 0, 1])
E2 = np.array([np.cos(THETA), np.sin(THETA), 0])
########################################################################################################################

# working source coordinates
# first = [-6 * np.tan(THETA) / 2, -3, 0]
# second = [6 * np.tan(THETA) / 2, -6 / 2, 0]

wave_1 = mhg.make_hg_beam_any_dir(K1, E1, FCEN, WAVELENGTH, [SX, SY, SZ], OBS_VOL, WAIST, m=0,
                                  n=0)
wave_2 = mhg.make_hg_beam_any_dir(K2, E2, FCEN, WAVELENGTH, [SX, SY, SZ], OBS_VOL, WAIST,
                                  m=0, n=0)
# wave_3 = mhg.make_hg_beam_any_dir(K3, E3, FCEN, WAVELENGTH, [SX, SY, SZ], [3, -3, -3], WAIST, m=0, n=0)
# wave_4 = mhg.make_hg_beam_any_dir(K4, E4, FCEN, WAVELENGTH, [SX, SY, SZ], [-3, 3, -3], WAIST, m=0, n=0)

waves = [wave_1, wave_2]
# waves=[wave_3]
all_waves = []

for wave in waves:
    for element in wave:
        all_waves.append(element)

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

eps_data = np.ma.masked_array(eps_data, eps_data < np.sqrt(1.4))

intensityNorm = 1 / (Ex * np.conjugate(Ex) + Ey * np.conjugate(Ey) + Ez * np.conjugate(Ez))

ESquared = np.real((Ex * np.conjugate(Ex) + Ey * np.conjugate(Ey) + Ez * np.conjugate(Ez)))
HSquared = np.real((Hx * np.conjugate(Hx) + Hy * np.conjugate(Hy) + Hz * np.conjugate(Hz)))

S0 = np.real(intensityNorm * (Ex * np.conjugate(Ex) + Ey * np.conjugate(Ey)))
S1 = np.real(intensityNorm * (Ex * np.conjugate(Ex) - Ey * np.conjugate(Ey)))
S2 = np.real(intensityNorm * (Ex * np.conjugate(Ey) + Ey * np.conjugate(Ex)))
S3 = np.real(intensityNorm * 1j * (Ex * np.conjugate(Ey) - Ey * np.conjugate(Ex)))

helicity_density = np.imag(intensityNorm * (Ex * np.conjugate(Hx) + Ey * np.conjugate(Hy) + Ez * np.conjugate(Hz)))

fig, ax = plt.subplots(3, 3, figsize=(12, 10))

ax[0, 0].pcolormesh(x, y, np.transpose(np.real(Ex)))
ax[0, 0].set_title('Ex')
ax[0, 0].pcolormesh(x, y, np.transpose(eps_data), cmap='Greys', alpha=1, vmin=0, vmax=4)

ax[0, 1].pcolormesh(x, y, np.transpose(np.real(Ey)))
ax[0, 1].set_title('Ey')
ax[0, 1].pcolormesh(x, y, np.transpose(eps_data), cmap='Greys', alpha=1, vmin=0, vmax=4)

ax[0, 2].pcolormesh(x, y, np.transpose(np.real(Ez)))
ax[0, 2].set_title('Ez')
ax[0, 2].pcolormesh(x, y, np.transpose(eps_data), cmap='Greys', alpha=1, vmin=0, vmax=4)

i = 0

for ax, Si, name in zip([ax[1, 0], ax[1, 1], ax[1, 2], ax[2, 0], ax[2, 1], ax[2, 2]],
                        [ESquared, HSquared, helicity_density, S1, S2, S3],
                        ['E field Intensity', 'H field Intensity', "Helicity density", 'S1', 'S2', 'S3']):
    S_ax = ax.pcolormesh(x, y, np.transpose(Si), vmax=1, vmin=-1, cmap='RdYlBu')
    ax.pcolormesh(x, y, np.transpose(eps_data), cmap='Greys', alpha=1, vmin=0, vmax=4)
    ax.set_title(name)
    cbar_ax = fig.add_axes([0.85, 0.1, 0.02, 0.5])
    fig.colorbar(S_ax, cax=cbar_ax)
fig.subplots_adjust(right=0.8)

plt.show()
#
# plt.pcolormesh(helicityDensity, vmin=-1, vmax=1, cmap='RdYlBu')
# plt.show()
