"""Audio equalizing capabilities for amodem."""

import itertools

import numpy as np

from . import dsp
from . import sampling
from . import levinson


class Equalizer:

    def __init__(self, config):
        self.carriers = config.carriers
        self.omegas = 2 * np.pi * np.array(config.frequencies) / config.Fs
        self.Nfreq = config.Nfreq
        self.Nsym = config.Nsym

    def train_symbols(self, length, constant_prefix=32):
        def zadoff_chu(length, u=1):
            n = np.arange(length)
            return np.exp(-1j * np.pi * u * n * (n + 1) / length)

        symbols = []
        for i in range(self.Nfreq):
            zc = zadoff_chu(length, u=i+1)
            symbols.append(zc)
        
        symbols = np.array(symbols).T
        symbols[:constant_prefix, :] = 1
        return symbols

    def modulator(self, symbols):
        gain = 1.0 / len(self.carriers)
        result = []
        for s in symbols:
            result.append(np.dot(s, self.carriers))
        result = np.concatenate(result).real * gain
        assert np.max(np.abs(result)) <= 1
        return result

    def demodulator(self, signal, size):
        signal = itertools.chain(signal, itertools.repeat(0))
        symbols = dsp.Demux(sampler=sampling.Sampler(signal),
                            omegas=self.omegas, Nsym=self.Nsym)
        return np.array(list(itertools.islice(symbols, size)))


equalizer_length = 2000
silence_length = 50
prefix = [1]*equalizer_length + [0]*silence_length


def train(signal, expected, order, lookahead=0):
    padding = np.zeros(lookahead)
    assert len(signal) == len(expected)
    x = np.concatenate([signal, padding])
    y = np.concatenate([padding, expected])

    N = order + lookahead  # filter length
    Rxx = np.zeros(N)
    Rxy = np.zeros(N)
    for i in range(N):
        Rxx[i] = np.dot(x[i:], x[:len(x)-i])
        Rxy[i] = np.dot(y[i:], x[:len(x)-i])
    return levinson.solver(t=Rxx, y=Rxy)
