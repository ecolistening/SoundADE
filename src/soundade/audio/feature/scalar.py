"""
This module contains functions for computing various audio features.

Functions:
- spectral_flux: Compute the spectral flux of an audio signal.
- spectrogram: Compute the spectrogram of an audio signal.
- zero_crossing_rate: Compute the zero crossing rate of an audio signal.
- spectral_centroid: Compute the spectral centroid of an audio signal.
- root_mean_square: Compute the root mean square of an audio signal.
- acoustic_evenness_index: Compute the acoustic evenness index of an audio signal.
- bioacoustic_index: Compute the bioacoustic index of an audio signal.
- acoustic_complexity_index: Compute the acoustic complexity index of an audio signal.
- spectral_entropy: Compute the spectral entropy of an audio signal.
- temporal_entropy: Compute the temporal entropy of an audio signal.
- extract_all: Extract all audio features from a given audio file.

Classes:
- Feature: Represents an audio feature and provides a method for computing the feature.

Variables:
- FRAME_LENGTH: The frame length used for computing audio features.
- HOP_LENGTH: The hop length used for computing audio features.
- SAMPLING_RATE: The sampling rate of the audio signal.

Features: A list of Feature objects representing all the available audio features.
"""
import librosa
import maad
import maad.sound
import maad.features
import numpy as np
import pandas as pd
import soundfile

from itertools import chain
from findiff import FinDiff
from numpy.typing import NDArray
from typing import Any, Tuple

from soundade.audio.feature import Feature

def spectrogram(
    y: NDArray,
    sr: int,
    hop_length: int,
    n_fft: int,
    pad_mode: str = None,
    window: str = "hann",
    **kwargs: Any,
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    if pad_mode is not None:
        y = np.pad(y, pad_width=n_fft // 2, mode=pad_mode)
    Sxx, tn, fn, extent = maad.sound.spectrogram(
        y,
        sr,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        mode="complex",
        window=window,
    )
    return np.abs(Sxx), tn, fn, extent

def spectral_flux(
    y: NDArray | None = None,
    S: Tuple[NDArray, NDArray, NDArray, NDArray] | None = None,
    sr: int | None = None,
    use_finite_difference: bool = True,
    **kwargs: Any,
) -> np.float32:
    assert (y is not None and sr is not None) or (S is not None)

    if S is None:
        S = spectrogram(y, sr, **kwargs)
    Sxx, _, fn, _ = S

    # Normalize, column-wise
    Sxx = np.subtract(Sxx, Sxx.min(axis=0).reshape(1, -1))  # Subtract min
    Sxx = np.divide(Sxx, Sxx.max(axis=0).reshape(1, -1))

    if not use_finite_difference:
        diff = Sxx[:, 1:] - Sxx[:, :-1]
    else:
        dx = FinDiff(1, 1)
        diff = dx(Sxx)

    return np.mean(np.linalg.norm(diff, axis=0))

def zero_crossing_rate(
    y: NDArray | None = None,
    frame_length: int | None = None,
    hop_length: int | None = None,
    center: bool = True,
    **kwargs: Any,
) -> np.float32:
    assert y is not None
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length, center=center)
    return np.mean(zcr)

def spectral_centroid(
    y: NDArray = None,
    S: Tuple[NDArray, NDArray, NDArray, NDArray] = None,
    sr: int | None = None,
    n_fft: int | None = None,
    hop_length: int | None = None,
    window: str | None = None,
    pad_mode: str | None = None,
    freq: NDArray | None = None,
    **kwargs: Any,
) -> np.float32:
    assert (y is not None and sr is not None) or (S is not None and freq is not None)

    if S is None:
        S = spectrogram(y, sr, n_fft=n_fft, hop_length=hop_length, window=window, pad_mode=pad_mode)
    Sxx, _, freq, _ = S

    sc = librosa.feature.spectral_centroid(S=Sxx, freq=freq)
    return np.mean(sc)

def root_mean_square(
    y: NDArray = None,
    S: Tuple[NDArray, NDArray, NDArray, NDArray] = None,
    sr: int | None = None,
    frame_length: int | None = None,
    n_fft: int | None = None,
    hop_length: int | None = None,
    window: str | None = None,
    pad_mode: str | None = None,
    center: bool = True,
    **kwargs: Any,
) -> np.float32:
    assert (y is not None and sr is not None) or (S is not None)

    if S is None:
        S = spectrogram(y, sr, n_fft=n_fft, hop_length=hop_length, window=window, pad_mode=pad_mode)
    Sxx, _, _, _ = S

    rms = librosa.feature.rms(y=y, S=S, hop_length=hop_length, frame_length=frame_length, center=center)
    return np.mean(rms)

def acoustic_evenness_index(
    y: NDArray | None = None,
    S: Tuple[NDArray, NDArray, NDArray, NDArray] = None,
    sr: int | None = None,
    n_fft: int | None = None,
    hop_length: int | None = None,
    window: str | None = None,
    pad_mode: str | None = None,
    aei_flim: Tuple[int, int] = (0, 20_000),
    bin_step: int | None = None,
    db_threshold: int | None = None,
    R_compatible: bool = False,
    **kwargs: Any,
) -> np.float32:
    assert (y is not None and sr is not None) or (S is not None)
    assert (aei_flim is not None and bin_step is not None and db_threshold is not None)

    if R_compatible:
        y = y - np.mean(y)
        S = spectrogram(y, sr, n_fft=sr // 10, hop_length=sr // 10, window=window, pad_mode=pad_mode)
        Sxx, _, freq, _ = S
        return maad.features.acoustic_eveness_index(Sxx, freq, fmin=0, fmax=10_000, dB_threshold=-47)

    if S is None:
        S = spectrogram(y, sr, n_fft=n_fft, hop_length=hop_length, window=window, pad_mode=pad_mode)
    Sxx, _, freq, _ = S

    fmin, fmax = aei_flim
    return maad.features.acoustic_eveness_index(Sxx, freq, fmin=fmin, fmax=fmax, bin_step=bin_step, dB_threshold=db_threshold)

def bioacoustic_index(
    y: NDArray | None = None,
    S: Tuple[NDArray, NDArray, NDArray, NDArray] = None,
    sr: int | None = None,
    n_fft: int | None = None,
    hop_length: int | None = None,
    window: str | None = None,
    pad_mode: str | None = None,
    bi_flim: Tuple[int, int] | None = (2000, 15_000),
    R_compatible: bool = False,
    **kwargs: Any,
) -> np.float32:
    assert (y is not None and sr is not None) or (S is not None)

    if S is None:
        S = spectrogram(y, sr, n_fft=n_fft, hop_length=hop_length, window=window, pad_mode=pad_mode)
    Sxx, _, fn, _ = S

    fmin, fmax = bi_flim
    fmax = min(fmax, sr // 2)

    return maad.features.bioacoustics_index(Sxx, fn, flim=(fmin, fmax), R_compatible=R_compatible)

def acoustic_complexity_index(
    y: NDArray | None = None,
    S: Tuple[NDArray, NDArray, NDArray, NDArray] = None,
    sr: int | None = None,
    n_fft: int | None = None,
    hop_length: int | None = None,
    window: str | None = None,
    pad_mode: str | None = None,
    **kwargs: Any,
) -> np.float32:
    assert (y is not None and sr is not None) or (S is not None)

    if S is None:
        S = spectrogram(y, sr, n_fft=n_fft, hop_length=hop_length, window=window, pad_mode=pad_mode)
    Sxx, _, fn, _ = S

    ACI_xx, ACI_per_bin, ACI_sum = maad.features.acoustic_complexity_index(Sxx)
    return ACI_sum

def spectral_entropy(
    y: NDArray | None = None,
    S: Tuple[NDArray, NDArray, NDArray, NDArray] = None,
    sr: int | None = None,
    n_fft: int | None = None,
    hop_length: int | None = None,
    window: str | None = None,
    pad_mode: str | None = None,
    **kwargs: Any,
) -> np.float32:
    assert (y is not None and sr is not None) or (S is not None)

    if S is None:
        S = spectrogram(y, sr, n_fft=n_fft, hop_length=hop_length, window=window, pad_mode=pad_mode)

    Sxx, _, _, _ = S
    Sxx = np.power(Sxx, 2)

    return maad.features.frequency_entropy(Sxx)[0]

def temporal_entropy(
    y: NDArray | None = None,
    frame_length: int | None = None,
    **kwargs: Any,
) -> np.float32:
    assert y is not None and frame_length is not None

    return maad.features.temporal_entropy(y, Nt=frame_length)

Features = [
    Feature('zero crossing rate', zero_crossing_rate, center=True),
    Feature('spectral centroid', spectral_centroid),
    Feature('root mean square', root_mean_square, pad_mode="constant"),
    Feature('spectral flux', spectral_flux, pad_mode="constant"),
    Feature('acoustic evenness index', acoustic_evenness_index),
    Feature('bioacoustic index', bioacoustic_index),
    Feature('acoustic complexity index', acoustic_complexity_index),
    Feature('spectral entropy', spectral_entropy),
    Feature('temporal entropy', temporal_entropy),
]
