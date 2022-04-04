from __future__ import division

import cmath
import math
import meep as mp
import numpy as np
import matplotlib.pyplot as plt


# https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


dpml = 1
cell_size_x = 20
cell_size_y = 20
obs_size_x = 10
obs_size_y = 10
obsCellList = [[-x / 2, x / 2] for x in [obs_size_x, obs_size_y]]
obsCell = mp.Vector3(obs_size_y, obs_size_y)

cell = mp.Vector3(cell_size_x + 2 * dpml, cell_size_y + 2 * dpml)

pml_layers = [mp.PML(dpml)]
resolution = 12
blockSize = 1 / 4
src_pos = -cell_size_x / 2

FCEN = 10 / math.pi  # pulse center frequency
DT = 5
eps = 1.3


def pw_amp(k, x0, delta):
    def _pw_amp(x):
        return delta * cmath.exp(1j * k.dot(x + x0))

    return _pw_amp


df = 0.02  # turn-on bandwidth


def pw_maker(freq, kDir, pol, srcPos, srcBox, complex_phase=1):
    kDir = mp.Vector3(kDir[0], kDir[1])
    kVec = kDir.unit().scale(2 * math.pi * freq)
    polNormed = pol / np.sqrt(np.dot(pol, np.conjugate(pol)))
    xPart = mp.Source(mp.ContinuousSource(FCEN, fwidth=df), component=mp.Ex,
                      center=srcPos, size=srcBox,
                      amp_func=pw_amp(kVec, mp.Vector3(x=src_pos), delta=polNormed[0]))
    yPart = mp.Source(mp.ContinuousSource(FCEN, fwidth=df), component=mp.Ey,
                      center=srcPos, size=srcBox,
                      amp_func=pw_amp(kVec, mp.Vector3(x=src_pos), delta=polNormed[1]))
    return xPart, yPart


k1 = [1, 0]

sourceLineX = mp.Vector3(0, cell_size_y, 0)

# Plane wave source for input wave

sourceForward = mp.Source(
        mp.ContinuousSource(FCEN, width=DT, is_integrated=True),
        component=mp.Ex,
        size=mp.Vector3(),
        center=mp.Vector3(-cell_size_x/2, 0)
        ),

# Continuous source for fictitious dipole, with amplitude found from the forward sim

r0 = [3, 0]


def sourceAdjoint(amp):
    return [mp.Source(src=mp.ContinuousSource(FCEN),
                      center=mp.Vector3(x=r0[0], y=r0[1], z=0),
                      component=mp.Ex,
                      amplitude=amp[0]),
            mp.Source(src=mp.ContinuousSource(FCEN),
                      center=mp.Vector3(x=r0[0], y=r0[1], z=0),
                      component=mp.Ey,
                      amplitude=amp[1])]


def simFn(source, name):
    t = 100  # run time

    sim = mp.Simulation(
        cell_size=cell,
        sources=source,
        boundary_layers=pml_layers,
        resolution=resolution,
        # material_function = lens,
        geometry=geom,
        force_complex_fields=True,
        # k_point = mp.Vector3(0,0,0)
    )

    sim.run(until=t)

    x, y, z, w = sim.get_array_metadata(center=mp.Vector3(), size=obsCell)
    Ex = np.array(sim.get_array(center=mp.Vector3(), size=obsCell, component=mp.Ex))
    Ey = np.array(sim.get_array(center=mp.Vector3(), size=obsCell, component=mp.Ey))
    Ez = np.array(sim.get_array(center=mp.Vector3(), size=obsCell, component=mp.Ez))

    eps_data = sim.get_array(center=mp.Vector3(), size=obsCell, component=mp.Dielectric)
    fieldData = Ex, Ey, Ez
    metaData = x, y, z, w

    np.save("fieldData" + name, fieldData)
    np.save("metaData" + name, metaData)
    np.save("eps_data" + name, eps_data)


def geomGen(pos):
    return mp.Block(mp.Vector3(blockSize, blockSize, mp.inf),
                    center=mp.Vector3(pos[0], pos[1]),
                    material=mp.Medium(epsilon=eps))


def optregion(pos):
    return pos[0] < 0


geom = []

for i in range(10):
    simFn(sourceForward, "Forward")
    x, y, z, w = np.load("metaDataForward.npy", allow_pickle=True)
    fieldDataForward = np.load("fieldDataForward.npy", allow_pickle=True)
    x0 = find_nearest(x, r0[0])
    y0 = find_nearest(y, r0[1])
    EAtObs = ([np.conjugate(fieldDataForward[i][x0, y0]) for i in [0, 1, 2]])
    simFn(sourceAdjoint(EAtObs), "Adjoint")

    fieldDataForward = np.load("fieldDataForward.npy", allow_pickle=True)
    fieldDataAdjoint = np.load("fieldDataAdjoint.npy", allow_pickle=True)
    eps_data = np.load("eps_dataForward.npy", allow_pickle=True)
    x, y, z, w = np.load("metaDataForward.npy", allow_pickle=True)

    Ex, Ey, Ez = fieldDataForward
    ExA, EyA, EzA = fieldDataAdjoint

    dF = np.real(Ex * ExA + Ey * EyA)

    eps_data = np.ma.masked_array(eps_data, eps_data < np.sqrt(1.4))

    ESquared = np.real((Ex * np.conjugate(Ex) + Ey * np.conjugate(Ey) + Ez * np.conjugate(Ez)))

    fig, ax = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

    ax[0, 0].pcolormesh(x, y, np.transpose(np.real(Ey)))
    ax[0, 0].set_title('Ey')
    ax[0, 0].pcolormesh(x, y, np.transpose(eps_data), cmap='Greys', alpha=0.9, vmin=0, vmax=5 * eps)

    ax[0, 1].pcolormesh(x, y, np.transpose(ESquared))
    ax[0, 1].set_title('Intensity' + str(EAtObs @ np.conjugate(EAtObs)))
    ax[0, 1].pcolormesh(x, y, np.transpose(eps_data), cmap='Greys', alpha=0.9, vmin=0, vmax=5 * eps)
    ax[0, 1].plot(r0[0], r0[1], 'ro')

    ax[1, 0].pcolormesh(x, y, np.transpose(np.real(EyA)))
    ax[1, 0].set_title('EyA')
    ax[1, 0].pcolormesh(x, y, np.transpose(eps_data), cmap='Greys', alpha=0.9, vmin=0, vmax=5 * eps)

    ax[1, 1].pcolormesh(x, y, np.transpose(np.real(dF)))
    ax[1, 1].set_title('dF')

    plt.savefig("plt" + str(i) + ".png")
    plt.close()
    maxX = int(len(x) / 2)
    minX = int(len(x) / 4)

    dFReduced = dF[minX:maxX, :]
    xReduced = x[minX:maxX]
    yReduced = y
    ind = np.unravel_index(np.argmax(dFReduced, axis=None), dFReduced.shape)
    maxPos = [xReduced[ind[0]], yReduced[ind[1]]]
    geom.append(geomGen(maxPos))
