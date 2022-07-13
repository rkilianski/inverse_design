"""Script simulating helicity lattice in vacuum using 4 plane waves.  """
import meep as mp
import numpy as np
import module_3d_wave as m3d
import module_lg_beam_any as mlg
import module_hg_beam as mhg
import plane_rotator as pr
import set_waves_module as sw
import matplotlib.pyplot as plt

DPML = 2  # thickness of PML layers
COMP_X, COMP_Y, COMP_Z = [8, 8, 8]  # dimensions of the computational cell, not including PML
SX, SY, SZ = COMP_X + 2 * DPML, COMP_Y + 2 * DPML, COMP_Z + 2 * DPML  # cell size, including PML
CELL = mp.Vector3(SX, SY, SZ)
OBS_VOL = mp.Vector3(6, 6, 6)
PML_LAYERS = [mp.PML(DPML)]
RESOLUTION = 14

FCEN = 2 / np.pi  # pulse center frequency
DF = 0.02  # turn-on bandwidth
EPS = 1  # refractive index of material containing the source

WAVELENGTH = 1
WAIST = 12
L, P = 0, 0
M, N = 0, 0

########################################################################################################################
# K-VECTORS, E-VECTORS AND ROTATION
########################################################################################################################
C = 1
a1, a2, a3 = 5, 5, 5
T1, T2, T3 = 0, 0, 0
k_vectors, e_vectors = sw.make_3_wave_NI(C, T1, T2, T3, a1, a2, a3)
ROT = 3 * np.pi / 4
print(k_vectors)
# # rotating k vectors and e vectors
# k_vectors, e_vectors = pr.rotate_on_axis(k_vectors, e_vectors, ROT, 2)
k_vectors, e_vectors = pr.find_angles_and_rotate(k_vectors, e_vectors, 2)

########################################################################################################################
# SIMULATION
########################################################################################################################
T = 20  # run time

pw_waves = m3d.make_3d_wave(k_vectors, e_vectors, FCEN, DF, [SX, SY, SZ], OBS_VOL, EPS)
lg_waves = mlg.make_multiple_lg_beams(k_vectors, e_vectors, FCEN, WAVELENGTH, [SX, SY, SZ], OBS_VOL, WAIST, l=L, p=P)
hg_waves = mhg.make_multiple_hg_beams(k_vectors, e_vectors, FCEN, WAVELENGTH, [SX, SY, SZ], OBS_VOL, WAIST, m=M, n=N)

sim = mp.Simulation(
    cell_size=CELL,
    sources=pw_waves,
    boundary_layers=PML_LAYERS,
    resolution=RESOLUTION,
    default_material=mp.Medium(index=N),
    force_complex_fields=True
)

sim2 = mp.Simulation(
    cell_size=CELL,
    sources=hg_waves,
    boundary_layers=PML_LAYERS,
    resolution=RESOLUTION,
    default_material=mp.Medium(index=N),
    force_complex_fields=True
)

sim3 = mp.Simulation(
    cell_size=CELL,
    sources=lg_waves,
    boundary_layers=PML_LAYERS,
    resolution=RESOLUTION,
    default_material=mp.Medium(index=N),
    force_complex_fields=True
)

sim.run(until=T)
sim2.run(until=T)
sim3.run(until=T)
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

helicity_density = np.imag(intensityNorm * (Ex * np.conjugate(Hx) + Ey * np.conjugate(Hy) + Ez * np.conjugate(Hz)))

########################################################################################################################

Ex2, Ey2, Ez2, Hx2, Hy2, Hz2, _ = [sim2.get_array(center=mp.Vector3(), size=OBS_VOL, component=i) for i in components]

Ex2, Ey2, Ez2, Hx2, Hy2, Hz2 = [a[1:-1, 1:-1, 1:-1] for a in [Ex2, Ey2, Ez2, Hx2, Hy2, Hz2]]

Ex2, Ey2, Ez2, Hx2, Hy2, Hz2 = [[a[chosen_slice, :, :], a[:, chosen_slice, :], a[:, :, chosen_slice]][SLICE_AXIS]
                                for a
                                in [Ex2, Ey2, Ez2, Hx2, Hy2, Hz2]]

intensityNorm2 = 1 / (Ex2 * np.conjugate(Ex2) + Ey2 * np.conjugate(Ey2) + Ez2 * np.conjugate(Ez2))

helicity_density2 = np.imag(
    intensityNorm2 * (Ex2 * np.conjugate(Hx2) + Ey2 * np.conjugate(Hy2) + Ez2 * np.conjugate(Hz2)))

########################################################################################################################

Ex3, Ey3, Ez3, Hx3, Hy3, Hz3, _ = [sim3.get_array(center=mp.Vector3(), size=OBS_VOL, component=i) for i in components]

Ex3, Ey3, Ez3, Hx3, Hy3, Hz3 = [a[1:-1, 1:-1, 1:-1] for a in [Ex3, Ey3, Ez3, Hx3, Hy3, Hz3]]

Ex3, Ey3, Ez3, Hx3, Hy3, Hz3 = [[a[chosen_slice, :, :], a[:, chosen_slice, :], a[:, :, chosen_slice]][SLICE_AXIS]
                                for a
                                in [Ex3, Ey3, Ez3, Hx3, Hy3, Hz3]]

intensityNorm3 = 1 / (Ex3 * np.conjugate(Ex3) + Ey3 * np.conjugate(Ey3) + Ez3 * np.conjugate(Ez3))

helicity_density3 = np.imag(
    intensityNorm3 * (Ex3 * np.conjugate(Hx3) + Ey3 * np.conjugate(Hy3) + Ez3 * np.conjugate(Hz3)))

########################################################################################################################
fig, ax = plt.subplots(1, 3, figsize=(12, 12))

ax[0].pcolormesh(x, y, np.transpose(helicity_density), cmap='RdBu', alpha=1, vmin=-1, vmax=1)
ax[0].set_title(f'Helicity Density 3 plane waves')

ax[1].pcolormesh(x, y, helicity_density2, cmap='RdBu', alpha=1, vmin=-1, vmax=1)
ax[1].set_title('Helicity Density 3 HG beams')

ax[2].pcolormesh(x, y, helicity_density3, cmap='RdBu', alpha=1, vmin=-1, vmax=1)
ax[2].set_title('Helicity Density 3 LG beams')

plt.show()
