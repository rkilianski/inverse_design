from __future__ import division

import cmath
import math
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import Module_GCalc as GCalc

new_sim = True

s = 20
dpml = 1

cell_size_x = 10
cell_size_y = 10
cell_size_z = 10

obs_size_x = 10
obs_size_y = 10
obs_size_z = 10
obsCellList = [[-x / 2, x / 2] for x in [obs_size_x, obs_size_y, obs_size_z]]
obsCell = mp.Vector3(obs_size_y, obs_size_y, obs_size_z)

cell = mp.Vector3(cell_size_x + 2 * dpml, cell_size_y + 2 * dpml, cell_size_z + 2 * dpml)

pml_layers = [mp.PML(dpml)]
resolution = 4

slicePosition = 0
sliceAxis = 0

src_pos = 0

fcen = 2 / math.pi  # pulse center frequency

eps = 1.5538 ** 2
epsFast = 2
epsSlow = 1
d = 1 / (4 * fcen * (np.sqrt(epsFast) - np.sqrt(epsSlow)))

epsilon_diag = mp.Vector3(epsFast, epsSlow, 1)

QWP1 = mp.Block(center=mp.Vector3(0, 0, 0),
                size=mp.Vector3(5, 5, d), material=mp.Medium(epsilon_diag=epsilon_diag))

cone = mp.Cone(center=mp.Vector3(0, 0, -2),
               radius=4, height=4, material=mp.Medium(epsilon=eps), axis=mp.Vector3(0, 0, 1))

geom = []


def pw_amp(k, x0, delta):
    def _pw_amp(x):
        return delta * cmath.exp(1j * k.dot(x + x0))

    return _pw_amp


df = 0.02  # turn-on bandwidth


def pw_maker(freq, kDir, pol, srcPos, srcBox, complex_phase=1):
    kDir = mp.Vector3(kDir[0], kDir[1], kDir[2])
    kVec = kDir.unit().scale(2 * math.pi * freq)
    polNormed = pol / np.sqrt(np.dot(pol, np.conjugate(pol)))
    xPart = mp.Source(mp.ContinuousSource(fcen, fwidth=df), component=mp.Ex,
                      center=srcPos, size=srcBox,
                      amp_func=pw_amp(kVec, mp.Vector3(x=src_pos), delta=polNormed[0]))
    yPart = mp.Source(mp.ContinuousSource(fcen, fwidth=df), component=mp.Ey,
                      center=srcPos, size=srcBox,
                      amp_func=pw_amp(kVec, mp.Vector3(x=src_pos), delta=polNormed[1]))
    zPart = mp.Source(mp.ContinuousSource(fcen, fwidth=df), component=mp.Ez,
                      center=srcPos, size=srcBox,
                      amp_func=pw_amp(kVec, mp.Vector3(x=src_pos), delta=polNormed[2]))
    return xPart, yPart, zPart


sourceArea_XY = mp.Vector3(cell_size_x, cell_size_y, 0)

sources = pw_maker(freq=fcen, kDir=[0, 0, 1], pol=[0, 1, 0], srcPos=[0, 0, -cell_size_z / 2], srcBox=sourceArea_XY)

sim = mp.Simulation(
    cell_size=cell,
    sources=sources,
    boundary_layers=pml_layers,
    resolution=resolution,
    geometry=geom,
    force_complex_fields=True,
    # k_point = mp.Vector3(0,0,0)
)

if new_sim:
    t = 250  # run time
    sim.run(until=t)

    # x,y,z,w = sim.get_array_metadata(center = mp.Vector3(),size = obsCell)
    Ex = np.array(sim.get_array(center=mp.Vector3(), size=obsCell, component=mp.Ex))
    Ey = np.array(sim.get_array(center=mp.Vector3(), size=obsCell, component=mp.Ey))
    Ez = np.array(sim.get_array(center=mp.Vector3(), size=obsCell, component=mp.Ez))

    Hx = np.array(sim.get_array(center=mp.Vector3(), size=obsCell, component=mp.Hx))
    Hy = np.array(sim.get_array(center=mp.Vector3(), size=obsCell, component=mp.Hy))
    Hz = np.array(sim.get_array(center=mp.Vector3(), size=obsCell, component=mp.Hz))

    eps_data = sim.get_array(center=mp.Vector3(), size=obsCell, component=mp.Dielectric)
    fieldData = Ex, Ey, Ez, Hx, Hy, Hz

    np.save("fieldData", fieldData)
    np.save("eps_data", eps_data)

x, y, z = [GCalc.GMeta(obsCellList, res=resolution) for obsCellList in obsCellList]
fieldData = np.load("fieldData.npy", allow_pickle=True)
eps_data = np.load("eps_data.npy", allow_pickle=True)

Ex, Ey, Ez, Hx, Hy, Hz = fieldData

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

fig, ax = plt.subplots(3, 3, figsize=(12, 10), sharex=True, sharey=True)

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
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.1, 0.02, 0.5])
fig.colorbar(S_ax, cax=cbar_ax)
plt.savefig("plot3D.png")
