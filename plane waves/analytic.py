
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import rotation_kvectors as rk

THETA = 3 * np.pi / 4
PHI = np.arctan(-np.sin(THETA))
ZETA = np.pi/2

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

xx, yy = np.meshgrid(np.linspace(-10, 10, 1000), np.linspace(-10, 10, 1000))

rot_M = rk.rotation_matrix

X = xx
Y = yy

E1 = np.array([0,1,0])
E2 = np.array([0,0,1])
E3 = np.array([1,0,0])
H1 = np.array([0,0,1])
H2 = np.array([1,0,0])
H3 = np.array([0,1,0])
k1 = np.array([1,0,0])
k2 = np.array([0,1,0])
k3 = np.array([0,0,1])

vectors = [E1,E2,E3,H1,H2,H3,k1,k2,k3]
E_vectors = vectors[:3]
H_vectors = vectors[3:6]
k_vectors = vectors[6:9]

r = np.array([1,1,1])
hl =0

rotated_helicity = -2.0000174792 * np.sin(1.41422 * yy) - 2.0000174792 * np.sin(1.22475 * xx - 0.70711 * yy) \
       + 2.0000174792 * np.sin(1.22475 * xx + 0.70711 * yy)

helicity = -2 * np.sin(xx - yy) + 2 * np.sin(xx) - 2 * np.sin(yy)

fig = plt.figure()

ax1 = fig.add_subplot(121)
ax1.imshow(helicity, cmap="plasma", interpolation='nearest', origin='lower', extent=[-10, 10, -10, 10])

ax2 = fig.add_subplot(122)
ax2.imshow(rotated_helicity, cmap="plasma", interpolation='nearest', origin='lower', extent=[-10, 10, -10, 10])

plt.show()

