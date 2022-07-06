"""Script simulating helicity lattice in vacuum using 3 HG beams.  """
import meep as mp
import numpy as np
import module_lg_beam_any as mlg
import plane_rotator as pr
import set_waves_module as sw
from matplotlib import pyplot as plt, animation

DPML = 2  # thickness of PML layers
COMP_X, COMP_Y, COMP_Z = [8, 8, 8]  # dimensions of the computational cell, not including PML
SX, SY, SZ = COMP_X + 2 * DPML, COMP_Y + 2 * DPML, COMP_Z + 2 * DPML  # cell size, including PML
CELL = mp.Vector3(SX, SY, SZ)
OBS_VOL = mp.Vector3(6, 6, 6)
PML_LAYERS = [mp.PML(DPML)]
RESOLUTION = 12

WAIST = 12
WAVELENGTH = 1.4
FCEN = 2 / np.pi  # pulse center frequency
DF = 0.02  # turn-on bandwidth
L, P = 0, 0

########################################################################################################################
# K-VECTORS, E-VECTORS AND ROTATION
########################################################################################################################
C = 1
a1, a2, a3 = 1, 1, 1
T1, T2, T3 = 0, 0, 0
k_vectors, e_vectors = sw.make_3_wave_NI(C, T1, T2, T3, a1, a2, a3)
print(k_vectors)
# rotating k vectors and e vectors
k_vectors, e_vectors = pr.find_angles_and_rotate(k_vectors, e_vectors, prp_to=2)

########################################################################################################################
# SIMULATION
########################################################################################################################
T = 20  # run time

all_waves = []
helicity_anim = []
wavelengths = np.arange(1.0, 2, 0.1)
ITERATIONS = len(wavelengths)
SLICE_POSITION = 20
SLICE_AXIS = 2

for i in range(ITERATIONS):

    all_waves = mlg.make_multiple_lg_beams(k_vectors, e_vectors, FCEN, wavelengths[i], [SX, SY, SZ], OBS_VOL, WAIST, l=L, p=P)

    sim = mp.Simulation(
        cell_size=CELL,
        sources=all_waves,
        boundary_layers=PML_LAYERS,
        resolution=RESOLUTION,
        default_material=mp.Medium(index=1),
        force_complex_fields=True
    )

    sim.run(until=T)

    components = [mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz]

    Ex, Ey, Ez, Hx, Hy, Hz = [sim.get_array(center=mp.Vector3(), size=OBS_VOL, component=i) for i in
                              components]
    x, y, z, w = sim.get_array_metadata(center=mp.Vector3(), size=OBS_VOL)

    x = x[1:-1]
    y = y[1:-1]
    z = z[1:-1]

    chosen_slice = np.argmin((np.array([x, y, z][SLICE_AXIS]) - SLICE_POSITION) ** 2)

    Ex, Ey, Ez, Hx, Hy, Hz = [a[1:-1, 1:-1, 1:-1] for a in [Ex, Ey, Ez, Hx, Hy, Hz]]

    Ex, Ey, Ez, Hx, Hy, Hz = [
        [a[chosen_slice, :, :], a[:, chosen_slice, :], a[:, :, chosen_slice]][SLICE_AXIS]
        for a
        in [Ex, Ey, Ez, Hx, Hy, Hz]]

    intensityNorm = 1 / (Ex * np.conjugate(Ex) + Ey * np.conjugate(Ey) + Ez * np.conjugate(Ez))

    helicity_density = np.imag(intensityNorm * (Ex * np.conjugate(Hx) + Ey * np.conjugate(Hy) + Ez * np.conjugate(Hz)))

    helicity_anim.append(helicity_density)

########################################################################################################################
# PLOTS AND METADATA
########################################################################################################################

plt.rcParams["figure.figsize"] = [8.00, 8.00]
plt.rcParams["figure.autolayout"] = True
fig_a, ax_a = plt.subplots()

intns = ax_a.pcolormesh(x, y, np.transpose(helicity_anim[0]), vmax=1, vmin=-1, cmap='RdYlBu')
fig_a.colorbar(intns)


def animate(i):
    intns.set_array(np.transpose(helicity_anim[i][:, :]).flatten())
    ax_a.set_title(f"Wavelength:: {wavelengths[i]}")


anim = animation.FuncAnimation(fig_a, animate, interval=200, frames=ITERATIONS)
anim.save(f'Intensity animation up to {ITERATIONS} frames.gif')
plt.show()
