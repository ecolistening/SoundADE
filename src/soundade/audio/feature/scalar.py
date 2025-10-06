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
    R_compatible: bool = False,
    **kwargs: Any,
) -> np.float32:
    assert (y is not None and sr is not None) or (S is not None)

    if S is None:
        S = spectrogram(y, sr, **kwargs)
    Sxx, _, fn, _ = S
    Sxx_pow = Sxx ** 2

    Z = Sxx_pow / Sxx_pow.sum(axis=0)

    if R_compatible:
        diff = Z[:, 1:] - Z[:, :-1]
        flux = np.linalg.norm(diff, ord=2, axis=0)
        flux = np.insert(flux, 0, 0)
    else:
        dx = FinDiff(1, 1)
        diff = dx(Z)
        flux = np.linalg.norm(diff, ord=2, axis=0)

    return np.mean(flux)

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

    fmin, fmax = aei_flim

    if R_compatible:
        # seewave compatible AEI requires DC offset on spectrogram with 10 windows across the wav file
        y = y - np.mean(y)
        # the FFT is hard-coded for seewave AEI
        S = spectrogram(y, sr, n_fft=sr // 10, hop_length=sr // 10, window=window, pad_mode=pad_mode)
        Sxx, _, freq, _ = S
        return maad.features.acoustic_eveness_index(Sxx, freq, fmin=fmin, fmax=fmax, dB_threshold=db_threshold)

    if S is None:
        S = spectrogram(y, sr, n_fft=n_fft, hop_length=hop_length, window=window, pad_mode=pad_mode)
    Sxx, _, freq, _ = S

    return maad.features.acoustic_eveness_index(Sxx, freq, fmin=fmin, fmax=fmax, bin_step=bin_step, dB_threshold=db_threshold)

def bioacoustic_index_soundecology(
    Sxx: NDArray,
    sr: int,
    f_min: float,
    f_max: float,
) -> np.float32:
    """
    Replicates the original implementation for testing purposes:

    https://rdrr.io/cran/soundecology/src/R/bioacoust_index.R

    Oddities / errors to note about the soundecology / seewave implementation:

    1) Seewave / soundecology drop the upper bin (nyquist) from the spectrogram
    2) Seewave's meandB function uses an incorrect constant value when computing the mean in decibels
    3) Soundecology subtracts minimum to account for negative values before integration
    """
    # map to decibels (correctly)
    S_db = 20 * np.log10(Sxx / Sxx.max())
    # calculate mean spectrum
    # NB: seewave error: 'A' should be 20 since the spectrogram is an amplitude spectrogram, see seewave's meandB function
    A = 10
    S_mean = A * np.log10(np.mean(10**(S_db / A), axis=1))
    # extract relevant frequency bins
    bin_spacing = len(S_mean) / (sr // 2)
    # NB: soundecology rounds down to drop the nyquist bin
    f_min_idx, f_max_idx  = int(f_min * bin_spacing), int(f_max * bin_spacing)
    S_mean_seg = S_mean[f_min_idx:f_max_idx]
    # NB: soundecology error: subtract the minimum value to account for negative values (dB): its a bit odd!
    S_mean_seg_norm = S_mean_seg - S_mean_seg.min()
    return sum(S_mean_seg_norm * bin_spacing)

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

    if R_compatible:
        return bioacoustic_index_soundecology(Sxx, sr, f_min=fmin, f_max=fmax)
    else:
        return maad.features.bioacoustics_index(Sxx, fn, flim=(fmin, fmax))

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
    sr: int | None = None,
    n_fft: int | None = None,
    hop_length: int | None = None,
    window: str | None = None,
    pad_mode: str | None = None,
    compatibility: str = "QUT",
    **kwargs: Any,
) -> np.float32:
    assert y is not None and sr is not None
    S, freq = maad.sound.spectrum(y, sr, nperseg=n_fft, noverlap=n_fft - hop_length, nfft=n_fft, window=window, scaling="density")
    Hf, _ = maad.features.frequency_entropy(S, compatibility=compatibility)
    return Hf

def temporal_entropy(
    y: NDArray | None = None,
    frame_length: int | None = None,
    compatibility: str = "QUT",
    mode: str = "fast",
    **kwargs: Any,
) -> np.float32:
    assert y is not None and frame_length is not None
    Ht = maad.features.temporal_entropy(y, Nt=frame_length, compatibility=compatibility, mode=mode)
    return Ht

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
