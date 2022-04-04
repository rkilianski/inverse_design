import meep as mp
import numpy as np
import module_3d_wave as m3d
import rotation_kvectors as rk
import matplotlib.pyplot as plt

DPML = 3  # thickness of PML layers
COMP_X, COMP_Y, COMP_Z = [10, 10, 10]  # dimensions of the computational cell, not including PML
SX, SY, SZ = COMP_X + 2 * DPML, COMP_Y + 2 * DPML, COMP_Z + 2 * DPML  # cell size, including PML
CELL = mp.Vector3(SX, SY, SZ)
OBS_VOL = mp.Vector3(6, 6, 6)
PML_LAYERS = [mp.PML(DPML)]
RESOLUTION = 6

THETA = np.pi / 6
FCEN = 2 / np.pi  # pulse center frequency
DF = 0.02  # turn-on bandwidth
N = 1  # refractive index of material containing the source

K1, K2, K3 = rk.z_rotated_k_vectors
k_vectors = [K1,K2,K3]
E1, E2, E3 = K2, K3, K1
e_vectors = [E1,E2,E3]

multiple_waves = m3d.make_3d_wave(k_vectors, e_vectors, FCEN, DF, [SX, SY, SZ], [COMP_X, COMP_Y, COMP_Z], N)



sim = mp.Simulation(
    cell_size=CELL,
    sources=multiple_waves,
    boundary_layers=PML_LAYERS,
    resolution=RESOLUTION,
    default_material=mp.Medium(index=N),
    force_complex_fields=True
)

t = 20  # run time
sim.run(until=t)

components =[mp.Ex,mp.Ey,mp.Ez,mp.Hx,mp.Hy,mp.Hz,mp.Dielectric]

Ex, Ey, Ez, Hx, Hy, Hz, eps_data = [sim.get_array(center=mp.Vector3(), size=OBS_VOL, component=i) for i in components]
x, y, z, w = sim.get_array_metadata(center=mp.Vector3(), size=OBS_VOL)

x = x[1:-1]
y = y[1:-1]
z = z[1:-1]

slicePosition = 5
sliceAxis = 2

chosenSlice = np.argmin((np.array([x, y, z][sliceAxis]) - slicePosition) ** 2)

Ex, Ey, Ez, Hx, Hy, Hz, eps_data = [a[1:-1, 1:-1, 1:-1] for a in [Ex, Ey, Ez, Hx, Hy, Hz, eps_data]]

Ex, Ey, Ez, Hx, Hy, Hz, eps_data = [[a[chosenSlice, :, :], a[:, chosenSlice, :], a[:, :, chosenSlice]][sliceAxis] for a
                                    in [Ex, Ey, Ez, Hx, Hy, Hz, eps_data]]

eps_data = np.ma.masked_array(eps_data, eps_data < np.sqrt(1.4))

intensityNorm = 1 / (Ex * np.conjugate(Ex) + Ey * np.conjugate(Ey) + Ez * np.conjugate(Ez))

ESquared = np.real((Ex * np.conjugate(Ex) + Ey * np.conjugate(Ey) + Ez * np.conjugate(Ez)))
HSquared = np.real((Hx * np.conjugate(Hx) + Hy * np.conjugate(Hy) + Hz * np.conjugate(Hz)))

S0 = np.real(intensityNorm * (Ex * np.conjugate(Ex) + Ey * np.conjugate(Ey)))
S1 = np.real(intensityNorm * (Ex * np.conjugate(Ex) - Ey * np.conjugate(Ey)))
S2 = np.real(intensityNorm * (Ex * np.conjugate(Ey) + Ey * np.conjugate(Ex)))
S3 = np.real(intensityNorm * 1j * (Ex * np.conjugate(Ey) - Ey * np.conjugate(Ex)))

helicityDensity = np.imag(intensityNorm * (Ex * np.conjugate(Hx) + Ey * np.conjugate(Hy) + Ez * np.conjugate(Hz)))

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
                        [ESquared, HSquared, helicityDensity, S1, S2, S3],
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
