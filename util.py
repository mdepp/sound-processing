"""
Contains utilities (fetching frequencies, etc.)
"""
from collections import namedtuple
import numpy as np

Wave = namedtuple('Wave', ['frequency', 'amplitude']) # Stores a wave in frequency domain

def expandWaves(waves):
    """
    This function expands a list of waves of complex amplitude to a list of
    waves of purely real and purely imaginary amplitudes.
    """
    for w in waves:
        yield Wave(w.frequency, np.real(w.amplitude))
        yield Wave(w.frequency, -1.j*np.imag(w.amplitude))

def extractFrequencies(signal, sampling_rate):
    """
    When given a (real) signal at a certain sampling rate, this function uses a
    discrete fourier transform to decompose the signal into sinusoidal waves,
    and returns a list containing frequency, amplitude pairs for each wave. The
    amplitude can be real or imaginary -- if it is real, it is the cosine
    Fourier coefficient, and if it is imaginary, it is the sine Fourier
    coefficient.
    Arguments:
        * signal: a list of real numbers representing samples taken of some
                  input signal
        * sampling_rate: the number of samples of the signal taken per second
    """
    num_samples = len(signal) # Number of samples taken of the signal
    signal_length = num_samples/sampling_rate # Length (in seconds) of signal
    
    # Get a list of all possible (positive) frequencies in the signal, which
    # is indexed the same as the amplitudes calculated later
    frequencies = np.fft.fftfreq(num_samples, 1.0/sampling_rate)[:num_samples//2]
    # Calculate amplitudes of positive (and zero) frequencies. We divide by
    # num_samples to normalize the results, and multiply by 2 since we are
    # ignoring negative frequencies, which have a one-to-one correspondance to
    # positive frequencies in the real signal case.
    amplitudes = np.fft.fft(signal)[:num_samples//2]/num_samples*2
    amplitudes[0] *= 0.5 # zero frequency has no negative

    return expandWaves(Wave(frequencies[i], amplitudes[i]) for i in range(len(frequencies)))
