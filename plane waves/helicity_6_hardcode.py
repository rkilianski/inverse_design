from __future__ import division

import cmath
import math
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import rotation_kvectors as rk

s = 10 # the size of the computational cell, not including PML
dpml = 3 # thickness of PML layers

sxy = s + 2 * dpml  # cell size, including PML
cell = mp.Vector3(sxy, sxy, sxy)
obs_vol = mp.Vector3(6, 6, 6)
pml_layers = [mp.PML(dpml)]
resolution = 12


# pw-amp is a function that returns the amplitude exp(ik(x+x0)) at a
# given point x.  (We need the x0 because current amplitude functions
# in Meep are defined relative to the center of the current source,
# whereas we want a fixed origin.)  Actually, it is a function of k
# and x0 that returns a function of x ...
def pw_amp(k, x0):
    def _pw_amp(x):
        return cmath.exp(1j * k.dot(x + x0))

    return _pw_amp



fcen = 2 / np.pi  # pulse center frequency
df = 0.02  # turn-on bandwidth
n = 1  # refractive index of material containing the source


def k_vector(arr):
    k1dir = mp.Vector3(arr[0], arr[1], arr[2])  # direction of k (length is irrelevant)
    k = k1dir.unit().scale(2 * math.pi * fcen * n)  # k with correct length
    return k


########################################################################################################################
# K-VECTORS
########################################################################################################################
THETA = 2 * np.pi / 3
C = 1
K1 = C * np.array([1, 0, 0])
K2 = C * np.array([np.cos(THETA), np.sin(THETA), 0])
K3 = C * np.array([np.cos(THETA), -np.sin(THETA), 0])
K4 = C * np.array([-1, 0, 0])
K5 = C * np.array([-np.cos(THETA), -np.sin(THETA), 0])
K6 = C * np.array([-np.cos(THETA), np.sin(THETA), 0])

k_vectors = [K1, K2, K3, K4, K5, K6]
########################################################################################################################
# POLARISATION VECTORS
########################################################################################################################

amp1 = 1
amp2 = 1
amp3 = 1
amp4 = -np.conjugate(amp1) * np.sqrt(np.abs(np.cos(2 * THETA))) / np.cos(THETA)
amp5 = np.conjugate(amp2) / np.sqrt(np.abs(np.cos(2 * THETA)))
amp6 = np.conjugate(amp3) / np.sqrt(np.abs(np.cos(2 * THETA)))

E1 = amp1 * np.array([0, 0, 1])
E2 = amp2 * np.array([0, 0, 1])
E3 = amp3 * np.array([0, 0, 1])
E4 = amp4 * np.array([0, -1, 0])
E5 = amp5 * np.array([np.sin(THETA), -np.cos(THETA), 0])
E6 = amp6 * np.array([-np.sin(THETA), -np.cos(THETA), 0])

e_vectors = [E1, E2, E3, E4, E5, E6]

sources = [
    # #k1
    mp.Source(
        mp.ContinuousSource(fcen, fwidth=df),
        component=mp.Ez,
        size=mp.Vector3(0, sxy, sxy),
        center=mp.Vector3(-s / 2, 0, 0),
        amp_func=pw_amp(k_vector(K1), mp.Vector3(-s/ 2, 0, 0)),
        amplitude=E1[2]
    ),

    # # k2
    mp.Source(
        mp.ContinuousSource(fcen, fwidth=df),
        component=mp.Ez,
        size=mp.Vector3(0, sxy, sxy),
        center=mp.Vector3(s / 2, 0, 0),
        amp_func=pw_amp(k_vector(K2), mp.Vector3(s / 2, 0, 0)),
        amplitude=0.5*E2[2]
    ),
    mp.Source(
        mp.ContinuousSource(fcen, fwidth=df),
        component=mp.Ez,
        size=mp.Vector3(sxy, 0, sxy),
        center=mp.Vector3(0, -s/2, 0),
        amp_func=pw_amp(k_vector(K2), mp.Vector3(0, -s/2, 0)),
        amplitude=0.5 * E2[2]
    ),

    # k3
    mp.Source(
        mp.ContinuousSource(fcen, fwidth=df),
        component=mp.Ez,
        size=mp.Vector3(0, sxy, sxy),
        center=mp.Vector3(s / 2, 0, 0),
        amp_func=pw_amp(k_vector(K3), mp.Vector3(s / 2, 0, 0)),
        amplitude=0.5 * E3[2]
    ),
    mp.Source(
        mp.ContinuousSource(fcen, fwidth=df),
        component=mp.Ez,
        size=mp.Vector3(sxy, 0, sxy),
        center=mp.Vector3(0, s / 2, 0),
        amp_func=pw_amp(k_vector(K3), mp.Vector3(0, s / 2, 0)),
        amplitude=0.5 * E3[2]
    ),
    # k4

    mp.Source(
        mp.ContinuousSource(fcen, fwidth=df),
        component=mp.Ey,
        size=mp.Vector3(0, sxy, sxy),
        center=mp.Vector3(s / 2, 0, 0),
        amp_func=pw_amp(k_vector(K4), mp.Vector3(s / 2, 0, 0)),
        amplitude=E4[1]
    ),
    # k5

    mp.Source(
        mp.ContinuousSource(fcen, fwidth=df),
        component=mp.Ey,
        size=mp.Vector3(0, sxy, sxy),
        center=mp.Vector3(-s / 2, 0, 0),
        amp_func=pw_amp(k_vector(K5), mp.Vector3(-s / 2, 0, 0)),
        amplitude=0.5 * E5[1]
    ),
    mp.Source(
        mp.ContinuousSource(fcen, fwidth=df),
        component=mp.Ex,
        size=mp.Vector3(sxy, 0, sxy),
        center=mp.Vector3(0, s / 2, 0),
        amp_func=pw_amp(k_vector(K5), mp.Vector3(0, s / 2, 0)),
        amplitude=0.5 * E5[0]
    ),

    # k6

    mp.Source(
        mp.ContinuousSource(fcen, fwidth=df),
        component=mp.Ey,
        size=mp.Vector3(0, sxy, sxy),
        center=mp.Vector3(-s / 2, 0, 0),
        amp_func=pw_amp(k_vector(K6), mp.Vector3(-s / 2, 0, 0)),
        amplitude=0.5 * E6[1]
    ),
    mp.Source(
        mp.ContinuousSource(fcen, fwidth=df),
        component=mp.Ex,
        size=mp.Vector3(sxy, 0, sxy),
        center=mp.Vector3(0, -s / 2, 0),
        amp_func=pw_amp(k_vector(K6), mp.Vector3(0, -s / 2, 0)),
        amplitude=0.5 * E6[0]
    ),

]

sim = mp.Simulation(
    cell_size=cell,
    sources=sources,
    boundary_layers=pml_layers,
    resolution=resolution,
    default_material=mp.Medium(index=n),
    force_complex_fields=True
)

t = 20 # run time
sim.run(until=t)

Ex = np.array(sim.get_array(center=mp.Vector3(), size=obs_vol, component=mp.Ex))
Ey = np.array(sim.get_array(center=mp.Vector3(), size=obs_vol, component=mp.Ey))
Ez = np.array(sim.get_array(center=mp.Vector3(), size=obs_vol, component=mp.Ez))

Hx = np.array(sim.get_array(center=mp.Vector3(), size=obs_vol, component=mp.Hx))
Hy = np.array(sim.get_array(center=mp.Vector3(), size=obs_vol, component=mp.Hy))
Hz = np.array(sim.get_array(center=mp.Vector3(), size=obs_vol, component=mp.Hz))

eps_data = sim.get_array(center=mp.Vector3(), size=obs_vol, component=mp.Dielectric)

x, y, z, w = sim.get_array_metadata(center=mp.Vector3(), size=obs_vol)
x = x[1:-1]
y = y[1:-1]
z = z[1:-1]

slicePosition = 20
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
    S_ax = ax.pcolormesh(x, y, np.transpose(Si), vmax=1.25, vmin=-1.25, cmap='RdYlBu')
    ax.pcolormesh(x, y, np.transpose(eps_data), cmap='Greys', alpha=1, vmin=0, vmax=4)
    ax.set_title(name)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.1, 0.02, 0.5])
fig.colorbar(S_ax, cax=cbar_ax)
plt.show()

# plt.pcolormesh(helicityDensity,vmin=-1,vmax=1, cmap='RdYlBu')
# plt.show()
