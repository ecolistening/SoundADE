from itertools import chain

import librosa
import maad
import maad.sound
import numpy as np
import pandas as pd
from findiff import FinDiff

from soundade.audio.feature import Feature
from soundade.audio.feature.utils import pad_window

FRAME_LENGTH, HOP_LENGTH = 16000, 4000


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

    return dist


def do_spectrogram(y=None, frame_length=2048, hop_length=512, pad_mode='constant', spec_mode='amplitude', sr=48000,
                  nperseg=1024, noverlap=None, **kwargs):
    # Split the audio file into windows then return a list of spectrograms of each window
    windowed = pad_window(y, frame_length, hop_length, mode=pad_mode)

    try:
        return [maad.sound.spectrogram(w, sr, nperseg=nperseg, noverlap=noverlap, mode=spec_mode) for w in windowed]
    except IndexError as e:
        raise AttributeError(f'Frame length ({frame_length}) must be greater than 1.5*nperseg ({1.5*nperseg}).')


def zero_crossing_rate(y, frame_length=2048, hop_length=512, center=True, **kwargs):
    return librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length, center=center)


def spectral_centroid(y=None, sr=48000, S=None, n_fft=2048, hop_length=512, freq=None, win_length=None, window='hann',
                      center=True, pad_mode='constant', **kwargs):
    return librosa.feature.spectral_centroid(y=y, sr=sr, S=S, n_fft=n_fft, hop_length=hop_length, freq=freq,
                                             win_length=win_length, window=window, center=center, pad_mode=pad_mode)


def root_mean_square(y=None, S=None, frame_length=2048, hop_length=512, center=True, pad_mode='constant', **kwargs):
    return librosa.feature.rms(y=y, S=S, frame_length=frame_length, hop_length=hop_length, center=center,
                               pad_mode=pad_mode)


def acoustic_evenness_index(y=None, flim=(0, 20000), spectrograms=None, frame_length=2048, hop_length=512,
                            mode='constant', sr=48000, nperseg=1024, noverlap=None, **kwargs):
    assert (y is not None) or (spectrograms is not None)

    fmin, fmax = flim

    if spectrograms is None:
        spectrograms = do_spectrogram(y, frame_length, hop_length, pad_mode=mode, sr=sr, nperseg=nperseg,
                                     noverlap=noverlap)

    return np.array(
        [maad.features.acoustic_eveness_index(Sxx, fn, fmin=fmin, fmax=fmax) for Sxx, _, fn, _ in spectrograms])


def bioacoustic_index(y=None, flim=(2000, 15000), spectrograms=None, frame_length=2048, hop_length=512, mode='constant',
                      sr=48000, nperseg=1024, noverlap=None, **kwargs):
    assert (y is not None) or (spectrograms is not None)

    if spectrograms is None:
        spectrograms = do_spectrogram(y, frame_length, hop_length, pad_mode=mode, sr=sr, nperseg=nperseg,
                                     noverlap=noverlap)

    return np.array([maad.features.bioacoustics_index(Sxx, fn, flim) for Sxx, _, fn, _ in spectrograms])


def acoustic_complexity_index(y=None, spectrograms=None, frame_length=2048, hop_length=512, mode='constant', sr=48000,
                              nperseg=1024, noverlap=None, **kwargs):
    assert (y is not None) or (spectrograms is not None)

    if spectrograms is None:
        spectrograms = do_spectrogram(y, frame_length, hop_length, pad_mode=mode, sr=sr, nperseg=nperseg,
                                     noverlap=noverlap)

    return np.array([maad.features.acoustic_complexity_index(Sxx)[2] for Sxx, _, _, _ in spectrograms])


def spectral_entropy(y=None, power_spectrograms=None, frame_length=2048, hop_length=512, mode='constant', sr=48000,
                     nperseg=1024, noverlap=None, **kwargs):
    assert (y is not None) or (power_spectrograms is not None)

    if power_spectrograms is None:
        # Requires the power spectrogram, so compute it here.
        power_spectrograms = do_spectrogram(y, frame_length, hop_length, pad_mode=mode, spec_mode='psd', sr=sr,
                                           nperseg=nperseg, noverlap=noverlap)

    return np.array([maad.features.frequency_entropy(Sxx_power)[0] for Sxx_power, _, _, _ in power_spectrograms])


def temporal_entropy(y=None, Nt=512, frame_length=2048, hop_length=512, mode='constant', **kwargs):
    assert (y is not None)

    windowed = pad_window(y, frame_length, hop_length, mode=mode)

    return np.array([maad.features.temporal_entropy(w, Nt=Nt) for w in windowed])


SAMPLING_RATE = 48000

frame_hop_dict = {'n_fft': FRAME_LENGTH, 'frame_length': FRAME_LENGTH, 'hop_length': HOP_LENGTH}
zcr = Feature('zero crossing rate', zero_crossing_rate, **frame_hop_dict)
sc = Feature('spectral centroid', spectral_centroid, sr=SAMPLING_RATE, **frame_hop_dict)
rms = Feature('root mean square', root_mean_square, **frame_hop_dict)
sf = Feature('spectral flux', spectral_flux, **frame_hop_dict)
aei = Feature('acoustic evenness index', acoustic_evenness_index, **frame_hop_dict)
bi = Feature('bioacoustic index', bioacoustic_index, **frame_hop_dict)
aci = Feature('acoustic complexity index', acoustic_complexity_index, **frame_hop_dict)
hf = Feature('spectral entropy', spectral_entropy, **frame_hop_dict)
ht = Feature('temporal entropy', temporal_entropy, **frame_hop_dict)
Features = [zcr, sc, rms, sf, aei, bi, aci, hf, ht]


def extract_all(f, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH, mode='constant', sr=48000, nperseg=1024,
                noverlap=None, **kwargs):
    audio, sr = librosa.load(f, sr=SAMPLING_RATE)

    spectrograms = do_spectrogram(y=audio, frame_length=frame_length, hop_length=hop_length, pad_mode=mode, sr=sr,
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
