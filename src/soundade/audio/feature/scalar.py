from itertools import chain

import librosa
import maad
import numpy as np
import pandas as pd
import soundfile
from findiff import FinDiff

from soundade.audio.feature import Feature


def spectral_flux(y=None, sr=48000, S=None, n_fft=2048, hop_length=512, use_finite_difference=True,
                  **kwargs):  # , sr=22050, S=None, n_fft=2048, hop_length=512, freq=None, win_length=None, window='hann', center=True, pad_mode='constant')
    # Compute spectrogram
    # D = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, )  # STFT of y
    # D_mag, _ = librosa.magphase(D)

    if S is None:
        # noverlap = n_fft - hop_length
        S, tf, fn, extent = maad.sound.spectrogram(np.pad(y, pad_width=n_fft // 2, mode='constant'), sr, nperseg=n_fft,
                                                   noverlap=n_fft - hop_length, mode='amplitude')
        # S, _ = librosa.magphase(Sxx)

    # Normalize, column-wise
    S = np.subtract(S, S.min(axis=0).reshape(1, -1))  # Subtract min
    S = np.divide(S, S.max(axis=0).reshape(1, -1))

    if not use_finite_difference:
        diff = S[:, 1:] - S[:, :-1]
    else:
        dx = FinDiff(1, 1)
        diff = dx(S)

    dist = np.linalg.norm(diff, axis=0)

    return np.mean(dist)


def __spectrogram(y=None, frame_length=2048, hop_length=512, pad_mode='constant', spec_mode='amplitude', sr=48000,
                  nperseg=1024, noverlap=None, **kwargs):
    return maad.sound.spectrogram(y, sr, nperseg=nperseg, noverlap=noverlap, mode=spec_mode)


def zero_crossing_rate(y, frame_length=2048, hop_length=512, center=True, **kwargs):
    return np.mean(
        librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length, center=center))


def spectral_centroid(y=None, sr=48000, S=None, n_fft=2048, hop_length=512, freq=None, win_length=None, window='hann',
                      center=True, pad_mode='constant', **kwargs):
    return np.mean(librosa.feature.spectral_centroid(y=y, sr=sr, S=S, n_fft=n_fft, hop_length=hop_length, freq=freq,
                                                     win_length=win_length, window=window, center=center,
                                                     pad_mode=pad_mode))


def root_mean_square(y=None, S=None, frame_length=2048, hop_length=512, center=True, pad_mode='constant', **kwargs):
    return np.mean(librosa.feature.rms(y=y, S=S, frame_length=frame_length, hop_length=hop_length, center=center,
                                       pad_mode=pad_mode))


def acoustic_evenness_index(y=None, flim=(0, 20000), spectrograms=None, frame_length=2048, hop_length=512,
                            mode='constant', sr=48000, nperseg=1024, noverlap=None, **kwargs):
    assert (y is not None) or (spectrograms is not None)

    fmin, fmax = flim

    if spectrograms is None:
        spectrograms = __spectrogram(y, frame_length, hop_length, pad_mode=mode, sr=sr, nperseg=nperseg,
                                     noverlap=noverlap)

    Sxx, _, fn, _ = spectrograms

    return maad.features.acoustic_eveness_index(Sxx, fn, fmin=fmin, fmax=fmax)


def bioacoustic_index(y=None, flim=(2000, 15000), spectrograms=None, frame_length=2048, hop_length=512, mode='constant',
                      sr=48000, nperseg=1024, noverlap=None, **kwargs):
    assert (y is not None) or (spectrograms is not None)

    if spectrograms is None:
        spectrograms = __spectrogram(y, frame_length, hop_length, pad_mode=mode, sr=sr, nperseg=nperseg,
                                     noverlap=noverlap)

    Sxx, _, fn, _ = spectrograms

    return maad.features.bioacoustics_index(Sxx, fn, flim)


def acoustic_complexity_index(y=None, spectrograms=None, frame_length=2048, hop_length=512, mode='constant', sr=48000,
                              nperseg=1024, noverlap=None, **kwargs):
    assert (y is not None) or (spectrograms is not None)

    if spectrograms is None:
        spectrograms = __spectrogram(y, frame_length, hop_length, pad_mode=mode, sr=sr, nperseg=nperseg,
                                     noverlap=noverlap)

    Sxx, _, fn, _ = spectrograms

    return maad.features.acoustic_complexity_index(Sxx)[2]


def spectral_entropy(y=None, power_spectrograms=None, frame_length=2048, hop_length=512, mode='constant', sr=48000,
                     nperseg=1024, noverlap=None, **kwargs):
    assert (y is not None) or (power_spectrograms is not None)

    if power_spectrograms is None:
        # Requires the power spectrogram, so compute it here.
        power_spectrograms = __spectrogram(y, frame_length, hop_length, pad_mode=mode, spec_mode='psd', sr=sr,
                                           nperseg=nperseg, noverlap=noverlap)

    Sxx_power, _, _, _ = power_spectrograms

    return maad.features.frequency_entropy(Sxx_power)[0]


def temporal_entropy(y=None, Nt=512, frame_length=2048, hop_length=512, mode='constant', **kwargs):
    assert (y is not None)

    return maad.features.temporal_entropy(y, Nt=Nt)


FRAME_LENGTH, HOP_LENGTH = 2048, 512
# TODO THIS IS BAD. Don't hardcode this.
SAMPLING_RATE = 48000

frame_hop_dict = {'n_fft': FRAME_LENGTH, 'frame_length': FRAME_LENGTH, 'hop_length': HOP_LENGTH}
zcr = Feature('zero crossing rate', zero_crossing_rate, **frame_hop_dict)
sc = Feature('spectral centroid', spectral_centroid, sr=SAMPLING_RATE, **frame_hop_dict)  # TODO Wrong sampling rate
rms = Feature('root mean square', root_mean_square, **frame_hop_dict)
sf = Feature('spectral flux', spectral_flux, **frame_hop_dict)
aei = Feature('acoustic evenness index', acoustic_evenness_index, **frame_hop_dict)
bi = Feature('bioacoustic index', bioacoustic_index, **frame_hop_dict)
aci = Feature('acoustic complexity index', acoustic_complexity_index, **frame_hop_dict)
hf = Feature('spectral entropy', spectral_entropy, **frame_hop_dict)
ht = Feature('temporal entropy', temporal_entropy, **frame_hop_dict)
# TODO This should be a factory that takes parameters
Features = [zcr, sc, rms, sf, aei, bi, aci, hf, ht]


def extract_all(f, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH, mode='constant', nperseg=1024,
                noverlap=None, **kwargs):
    audio, sr = soundfile.read(f)

    spectrograms = __spectrogram(y=audio, frame_length=frame_length, hop_length=hop_length, pad_mode=mode, sr=sr,
                                 nperseg=nperseg, noverlap=noverlap, **kwargs)

    features = []
    for feature in Features:
        comp = feature.compute(audio, spectrograms=spectrograms, frame_length=frame_length, hop_length=hop_length,
                               pad_mode=mode, sr=sr, nperseg=nperseg, noverlap=noverlap, **kwargs)
        # comp = feature.compute(audio, **kwargs)
        row = dict(zip(chain(['file', 'feature', 'path'], [str(i) for i in range(comp.size)]),
                       chain([f.name, feature.name, str(f.parent), *comp.flatten().tolist()])))
        features.append(row)

    return pd.DataFrame(features)
