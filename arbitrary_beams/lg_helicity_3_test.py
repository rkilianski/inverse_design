"""Script simulating helicity lattice in vacuum using 3 HG beams.  """
import meep as mp
import numpy as np
import module_lg_beam_any as mlg
import rotation_kvectors as rk
import matplotlib.pyplot as plt
import plane_rotator as pr
import set_waves_module as sw

DPML = 2  # thickness of PML layers
COMP_X, COMP_Y, COMP_Z = [8, 8, 8]  # dimensions of the computational cell, not including PML
SX, SY, SZ = COMP_X + 2 * DPML, COMP_Y + 2 * DPML, COMP_Z + 2 * DPML  # cell size, including PML
CELL = mp.Vector3(SX, SY, SZ)
OBS_VOL = mp.Vector3(6, 6, 6)
VX, VY, VZ = OBS_VOL
PML_LAYERS = [mp.PML(DPML)]
RESOLUTION = 6

L, P = 0,0
WAIST = 20
WAVELENGTH = 1
FCEN = 2/ np.pi  # pulse center frequency
DF = 0.02  # turn-on bandwidth
N = 1  # refractive index of material containing the source

########################################################################################################################
# K-VECTORS, E-VECTORS AND ROTATION
########################################################################################################################
########################################################################################################################
# K-VECTORS, E-VECTORS AND ROTATION
########################################################################################################################
C = 1
a1, a2, a3 = 5, 5 * np.sqrt(2), 5
T1, T2, T3 = 0, 7 * np.pi / 4, 3 * np.pi / 2
k_vectors, e_vectors = sw.make_3_wave_NI(C, T1, T2, T3, a1, a2, a3)
print(k_vectors)
# rotating k vectors and e vectors
# k_vectors, e_vectors = pr.find_angles_and_rotate(k_vectors, e_vectors, prp_to=2)
k_vectors, e_vectors = pr.rotate_on_axis(k_vectors, e_vectors, np.pi / 4, 2)

########################################################################################################################
# SIMULATION
########################################################################################################################
T = 20  # run time
# K1, K2, K3 = rk.z_rotated_k_vectors
# k_vectors = [K1, K2, K3]
#
# E1, E2, E3 = K2, K3, K1
# e_vectors = [E1, E2, E3]

print(k_vectors)

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
