"""Script simulating helicity lattice in vacuum using 3 HG beams.  """
import meep as mp
import numpy as np
import module_hg_beam as mhg
import rotation_kvectors as rk
from matplotlib import pyplot as plt, animation

DPML = 2  # thickness of PML layers
COMP_X, COMP_Y, COMP_Z = [8, 8, 8]  # dimensions of the computational cell, not including PML
SX, SY, SZ = COMP_X + 2 * DPML, COMP_Y + 2 * DPML, COMP_Z + 2 * DPML  # cell size, including PML
CELL = mp.Vector3(SX, SY, SZ)
OBS_VOL = mp.Vector3(6, 6, 6)
VX, VY, VZ = OBS_VOL
PML_LAYERS = [mp.PML(DPML)]
RESOLUTION = 10

WAIST = 12
WAVELENGTH = 1.4
FCEN = 2 / np.pi  # pulse center frequency
DF = 0.02  # turn-on bandwidth
N = 1  # refractive index of material containing the source

########################################################################################################################
# SIMULATION
########################################################################################################################
T = 20  # run time
K1, K2, K3 = rk.z_rotated_k_vectors
k_vectors = [K1, K2, K3]
E1, E2, E3 = K2, K3, K1
e_vectors = [E1, E2, E3]

all_waves = []
helicity_anim = []
wavelengths = np.arange(0.8, 2, 0.1)
ITERATIONS = len(wavelengths)
SLICE_POSITION = 20
SLICE_AXIS = 2

for i in range(ITERATIONS):

    all_waves = []

    wave_1 = mhg.make_hg_beam_any_dir(K1, E1, FCEN, wavelengths[i], [SX, SY, SZ], OBS_VOL, WAIST, m=0, n=0)
    wave_2 = mhg.make_hg_beam_any_dir(K2, E2, FCEN, wavelengths[i], [SX, SY, SZ], OBS_VOL, WAIST, m=0, n=0)
    wave_3 = mhg.make_hg_beam_any_dir(K3, E3, FCEN, wavelengths[i], [SX, SY, SZ], OBS_VOL, WAIST, m=0, n=0)
    waves = [wave_1, wave_2, wave_3]
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
# ax_a.plot(x[x_obs_index], y[y_obs_index], 'ro')
fig_a.colorbar(intns)


def animate(i):
    intns.set_array(np.transpose(helicity_anim[i][:, :]).flatten())
    ax_a.set_title(f"Wavelength:: {wavelengths[i]}")


anim = animation.FuncAnimation(fig_a, animate, interval=100, frames=ITERATIONS)
anim.save(f'Intensity animation up to {ITERATIONS} frames.gif')
plt.show()
