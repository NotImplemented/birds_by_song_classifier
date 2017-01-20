import numpy
import math

import matplotlib.pyplot

# using definition
def fourier_series(sample):

    n = len(sample)
    result = numpy.zeros(n, dtype=complex)

    for m in range(n):

        coeff = complex()
        for p in range(n):
            coeff += sample[p] * numpy.exp(complex(0.0, -2.0 * math.pi * m * p / n))

        result[m] = coeff

    return result


sample_size = 1024
sines = 8
sines_step = 16

t = []
f = []

for x in range(sample_size):

    t.append(x / float(sample_size))
    value = 0

    for i in range(sines):
        value += math.sin(2.0 * (sines_step * i + 1.0) * math.pi * x / float(sample_size))

    f.append(value)

matplotlib.pyplot.ion()

matplotlib.pyplot.figure(1)
matplotlib.pyplot.subplot(211)

matplotlib.pyplot.plot(t, f, 'r')
matplotlib.pyplot.xlabel('t')
matplotlib.pyplot.ylabel('f(t)')
matplotlib.pyplot.pause(0.05)

fourier_series = fourier_series(f)

fourier_absolute_values = []

for x in range(sample_size):
    fourier_absolute_values.append(numpy.abs(fourier_series[x]))

matplotlib.pyplot.subplot(212)
matplotlib.pyplot.plot(t, fourier_absolute_values, 'g')

matplotlib.pyplot.ylabel('fourier_series')
matplotlib.pyplot.pause(0.05)

matplotlib.pyplot.show()