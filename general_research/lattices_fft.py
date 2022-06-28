import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 50)
y = np.linspace(-10, 10, 50)
wavelength = 500

centre = int((len(x) - 1) / 2)
coords_left_half = (
    (x, y) for x in range(len(x)) for y in range(centre + 1)
)

xx, yy = np.meshgrid(x, y)

arg = 2 * np.pi / wavelength

z = np.cos(2*np.pi* xx)
sampl = x[1]-x[0]
print(sampl)
def calculate_distance_from_centre(coords, centre):
    # Distance from centre is √(x^2 + y^2)
    return np.sqrt(
        (coords[0] - centre) ** 2 + (coords[1] - centre) ** 2
    )


def calculate_2dft(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)


coords_left_half = sorted(
    coords_left_half,
    key=lambda x: calculate_distance_from_centre(x, centre)
)


def find_symmetric_coordinates(coords, centre):
    return (centre + (centre - coords[0]),
            centre + (centre - coords[1]))


ft = calculate_2dft(z)
plt.subplot(121)
plt.imshow(z)
plt.subplot(122)
FT = abs(ft)
ind = np.unravel_index(np.argmax(ft.real, axis=None), ft.real.shape)
print(ind)
print(ft.real[ind])
plt.imshow(FT)

freqx = np.fft.fftfreq(len(x))
freqy = np.fft.fftfreq(len(y))
# print(np.fft.fftfreq(FT[:, 8]))
# plt.xlim([480, 520])
# plt.ylim([520, 480])  # Note, order is reversed for y
plt.show()

# for coef, freq in zip(ftx, freq):
#     if coef:
#         print('{c:>6} * exp(2 pi i t * {f})'.format(c=coef,
#                                                     f=freq))
print(freqx[ind[0]], freqy[ind[1]])

