import numpy as np
import matplotlib.pyplot as plt
from scipy import fft
from scipy.signal import square

from util import extractFrequencies

def getSignalValue(waves, period, time):
    val = 0
    for frequency, amplitude in waves:
        real = np.real(amplitude)
        imag = np.imag(amplitude)
        if np.isclose(imag, 0): # Wave has real amplitude
            val += real*np.cos(frequency*2*np.pi/period*time)
        elif np.isclose(real, 0): # Wave has imaginary amplitude
            val += imag*np.sin(frequency*2*np.pi/period*time)
        else: # Wave has no amplitude
            pass
    return val

def writeToFile(data, filename):
    from scipy.io.wavfile import write
    scaled = np.int16(data/np.max(np.abs(data)) * 32767)
    write(filename, len(data), scaled)

def main():
    sampling_rate = 10000
    sampling_interval = 1.0/sampling_rate
    t_axis = np.arange(0, 2, sampling_interval)
    signal_frequency = 440
    #signal = np.sin(2 * np.pi * signal_frequency * t_axis) + np.cos(2*np.pi*signal_frequency*2*t_axis)
    signal = square(t_axis*2*np.pi*signal_frequency) + np.sin(2*np.pi*signal_frequency*t_axis) + np.sin(2*np.pi*signal_frequency*t_axis*1.333)

    plt.subplot(2, 1, 1)
    plt.plot(t_axis, signal, 'k-')
    plt.xlabel('time')
    plt.ylabel('amplitude')

    plt.subplot(2,1,2)
    num_samples = len(signal) # Total number of samples in signal
    signal_length = num_samples*sampling_interval # Length of signal in seconds

    
    waves = extractFrequencies(signal, sampling_rate)

    # Display the dominant frequencies
    dominant_waves = list(w for w in waves if abs(w.amplitude) > 0.2)

    for frequency, amplitude in dominant_waves:
        print('{f} Hz : {a}'.format(f=frequency, a=amplitude))

    period = 1
    reconstructed = [getSignalValue(dominant_waves, period, t) for t in t_axis]

    plt.plot(t_axis, signal, 'k-')
    plt.plot(t_axis, reconstructed, 'r-')

    plt.xlabel('time')
    plt.ylabel('amplitude')

    writeToFile(signal, 'signal.wav')
    writeToFile(reconstructed, 'reconstructed.wav')

    plt.show()      

main()
