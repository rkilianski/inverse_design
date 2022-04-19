import meep as mp
import numpy as np
import module_3d_wave as m3d
import rotation_kvectors as rk
import matplotlib.pyplot as plt

SLICE_POSITION = 20
SLICE_AXIS = 2

DPML = 3  # thickness of PML layers
COMP_X, COMP_Y, COMP_Z = [10, 10, 10]  # dimensions of the computational cell, not including PML
SX, SY, SZ = COMP_X + 2 * DPML, COMP_Y + 2 * DPML, COMP_Z + 2 * DPML  # cell size, including PML
CELL = mp.Vector3(SX, SY, SZ)
OBS_VOL = mp.Vector3(8, 8, 8)
PML_LAYERS = [mp.PML(DPML)]
RESOLUTION = 14

THETA_1 = np.pi / 6
FCEN = 2 / np.pi  # pulse center frequency
DF = 0.02  # turn-on bandwidth
N = 1  # refractive index of material containing the source
components = [mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz, mp.Dielectric]

slicePosition = 5
sliceAxis = 2

K1, K2, K3 = rk.z_rotated_k_vectors
k_vectors_3 = [K1, K2, K3]
E1, E2, E3 = K2, K3, K1
e_vectors_3 = [E1, E2, E3]

three_waves = m3d.make_3d_wave(k_vectors_3, e_vectors_3, FCEN, DF, [SX, SY, SZ], [COMP_X, COMP_Y, COMP_Z], N)

sim = mp.Simulation(
    cell_size=CELL,
    sources=three_waves,
    boundary_layers=PML_LAYERS,
    resolution=RESOLUTION,
    default_material=mp.Medium(index=N),
    force_complex_fields=True
)

t = 20  # run time
sim.run(until=t)

Ex, Ey, Ez, Hx, Hy, Hz, eps_data = [sim.get_array(center=mp.Vector3(), size=OBS_VOL, component=i) for i in components]
x, y, z, w = sim.get_array_metadata(center=mp.Vector3(), size=OBS_VOL)

x = x[1:-1]
y = y[1:-1]
z = z[1:-1]

chosenSlice = np.argmin((np.array([x, y, z][sliceAxis]) - slicePosition) ** 2)

Ex, Ey, Ez, Hx, Hy, Hz, eps_data = [a[1:-1, 1:-1, 1:-1] for a in [Ex, Ey, Ez, Hx, Hy, Hz, eps_data]]

Ex, Ey, Ez, Hx, Hy, Hz, eps_data = [[a[chosenSlice, :, :], a[:, chosenSlice, :], a[:, :, chosenSlice]][sliceAxis] for a
                                    in [Ex, Ey, Ez, Hx, Hy, Hz, eps_data]]

eps_data = np.ma.masked_array(eps_data, eps_data < np.sqrt(1.4))

intensityNorm = 1 / (Ex * np.conjugate(Ex) + Ey * np.conjugate(Ey) + Ez * np.conjugate(Ez))

helicity_density_3 = np.imag(intensityNorm * (Ex * np.conjugate(Hx) + Ey * np.conjugate(Hy) + Ez * np.conjugate(Hz)))

########################################################################################################################

########################################################################################################################
# K-VECTORS
########################################################################################################################
THETA_2 = np.pi / 6
C = np.sqrt(2) / 2
K1 = C * np.array([np.cos(THETA_2), np.sin(THETA_2), 1])
K2 = C * np.array([-np.cos(THETA_2), -np.sin(THETA_2), 1])
K3 = C * np.array([-np.cos(THETA_2), np.sin(THETA_2), 1])
K4 = C * np.array([np.cos(THETA_2), -np.sin(THETA_2), 1])

k_vectors_4 = [K1, K2, K3, K4]
########################################################################################################################
# POLARISATION VECTORS
########################################################################################################################
delta_phi = 0
amp1 = 1
a3 = 1
amp2 = np.conjugate(amp1) * (a3 / np.conjugate(a3)) * np.exp(1j * delta_phi)
amp3 = a3
amp4 = a3 * np.exp(1j * delta_phi)

E1 = C * amp1 * np.array([-np.cos(THETA_2), -np.sin(THETA_2), 1])
E2 = C * amp2 * np.array([-np.cos(THETA_2), -np.sin(THETA_2), -1])
E3 = C * amp3 * np.array([np.sin(THETA_2), -np.cos(THETA_2), 1])
E4 = C * amp4 * np.array([-np.sin(THETA_2), np.cos(THETA_2), 1])

e_vectors_4 = [E1, E2, E3, E4]
########################################################################################################################
# SIMULATION
########################################################################################################################
T = 20  # run time

four_waves = m3d.make_3d_wave(k_vectors_4, e_vectors_4, FCEN, DF, [SX, SY, SZ], [COMP_X, COMP_Y, COMP_Z], N)

sim = mp.Simulation(
    cell_size=CELL,
    sources=four_waves,
    boundary_layers=PML_LAYERS,
    resolution=RESOLUTION,
    default_material=mp.Medium(index=N),
    force_complex_fields=True
)

sim.run(until=T)

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

helicity_density_4 = np.imag(intensityNorm * (Ex * np.conjugate(Hx) + Ey * np.conjugate(Hy) + Ez * np.conjugate(Hz)))

########################################################################################################################
# K-VECTORS
########################################################################################################################
THETA_3 = 2 * np.pi / 3
C = 1
K1 = C * np.array([1, 0, 0])
K2 = C * np.array([np.cos(THETA_3), np.sin(THETA_3), 0])
K3 = C * np.array([np.cos(THETA_3), -np.sin(THETA_3), 0])
K4 = C * np.array([-1, 0, 0])
K5 = C * np.array([-np.cos(THETA_3), -np.sin(THETA_3), 0])
K6 = C * np.array([-np.cos(THETA_3), np.sin(THETA_3), 0])

k_vectors_6 = [K1, K2, K3, K4, K5, K6]
########################################################################################################################
# POLARISATION VECTORS
########################################################################################################################

amp1 = 1
amp2 = 2
amp3 = 2
amp4 = -np.conjugate(amp1) * np.sqrt(np.abs(np.cos(2 * THETA_3))) / np.cos(THETA_3)
amp5 = np.conjugate(amp2) / np.sqrt(np.abs(np.cos(2 * THETA_3)))
amp6 = np.conjugate(amp3) / np.sqrt(np.abs(np.cos(2 * THETA_3)))

E1 = amp1 * np.array([0, 0, 1])
E2 = amp2 * np.array([0, 0, 1])
E3 = amp3 * np.array([0, 0, 1])
E4 = amp4 * np.array([0, -1, 0])
E5 = amp5 * np.array([np.sin(THETA_3), -np.cos(THETA_3), 0])
E6 = amp6 * np.array([-np.sin(THETA_3), -np.cos(THETA_3), 0])

e_vectors_6 = [E1, E2, E3, E4, E5, E6]
########################################################################################################################
# SIMULATION
########################################################################################################################
T = 30  # run time

six_waves = m3d.make_3d_wave(k_vectors_6, e_vectors_6, FCEN, DF, [SX, SY, SZ], [COMP_X, COMP_Y, COMP_Z], N)

sim = mp.Simulation(
    cell_size=CELL,
    sources=six_waves,
    boundary_layers=PML_LAYERS,
    resolution=RESOLUTION,
    default_material=mp.Medium(index=N),
    force_complex_fields=True
)

sim.run(until=T)

########################################################################################################################
# PLOTS AND METADATA
########################################################################################################################


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

helicity_density_6 = np.imag(intensityNorm * (Ex * np.conjugate(Hx) + Ey * np.conjugate(Hy) + Ez * np.conjugate(Hz)))

fig, ax = plt.subplots(1, 3, figsize=(10, 10))

ax[0].pcolormesh(x, y, np.transpose(helicity_density_3), cmap='bwr')
ax[0].set_title('3-wave helicity density')

ax[1].pcolormesh(x, y, np.transpose(helicity_density_4), cmap='bwr')
ax[1].set_title('4-wave helicity density')

ax[2].pcolormesh(x, y, np.transpose(helicity_density_6), cmap='bwr')
ax[2].set_title('6-wave helicity density')

plt.show()
