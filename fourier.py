import numpy
import numpy.fft
import math

import matplotlib.pyplot

# using definition
def fourier_series(sample):

    n = len(sample)
    result = numpy.zeros(n)

    for m in range(n):

        coeff = complex()
        for p in range(n):
            coeff += sample[p] * numpy.exp(complex(0.0, -2.0 * math.pi * m * p / n))

        result[m] = numpy.abs(coeff)

    return result

def fourier_series_test():

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
    matplotlib.pyplot.subplot(311)

    matplotlib.pyplot.plot(t, f, 'r')
    matplotlib.pyplot.xlabel('t')
    matplotlib.pyplot.ylabel('f(t)')
    matplotlib.pyplot.pause(0.05)

    fourier_series = fourier_series(f)

    matplotlib.pyplot.subplot(312)
    matplotlib.pyplot.plot(t, fourier_series, 'g')

    fourier_values = numpy.abs(numpy.fft.fft(f))

    matplotlib.pyplot.subplot(313)
    matplotlib.pyplot.plot(t, fourier_values)

    matplotlib.pyplot.ylabel('fourier_series')
    matplotlib.pyplot.pause(0.05)

    matplotlib.pyplot.show()