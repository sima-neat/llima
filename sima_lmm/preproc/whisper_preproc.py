# Copied from https://github.com/openai/whisper/blob/main/whisper/audio.py with some modifications.

import json
import numpy as np
from functools import lru_cache
from pathlib import Path
from scipy import signal
import av
"""
Script for audio preprocessing for whisper encoder using Numpy.
"""

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3000 frames in a mel spectrogram input


def load_audio(file: str, sr: int = SAMPLE_RATE):
    container = av.open(file)
    resampler = av.AudioResampler(format='s16', layout='mono', rate=sr)

    parts = []
    for frame in container.decode(audio=0):
        frame.pts = None
        for resampled in resampler.resample(frame):
            parts.append(resampled.to_ndarray().flatten())
            
    audio_int16 = np.concatenate(parts)
    return audio_int16.astype(np.float32) / 32768.0


def pad_or_trim(array: np.ndarray, length: int = N_SAMPLES, *, axis: int = -1) -> np.ndarray:
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """

    if array.shape[axis] > length:
        array = array.take(indices=range(length), axis=axis)

    if array.shape[axis] < length:
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(n_mels: int, hf_preprocessor_config_json_file: Path | None = None) -> np.ndarray:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
            mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
        )
    """
    if hf_preprocessor_config_json_file is None:
        assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"
        mel_filters_file = Path(__file__).parent / "mel_filters.npz"
        with np.load(mel_filters_file, allow_pickle=False) as f:
            return f[f"mel_{n_mels}"]
    else:
        with open(hf_preprocessor_config_json_file, "r") as f:
            preprocessor_config = json.load(f)
        mel_filters = preprocessor_config["mel_filters"]
        return np.asarray(mel_filters, dtype=np.float32)


stft_window = signal.get_window("hann", N_FFT, fftbins=True)


def log_mel_spectrogram(
    audio: np.ndarray, n_mels: int = 80, hf_preprocessor_config_json_file: Path | None = None,
    stft_style: str = "scipy"
):
    """
    Compute the log-Mel spectrogram of

    Args:
        audio: A NumPy array containing the audio waveform in 16 kHz.
        n_mels: The number of Mel-frequency filters, only 80 is supported.

    Returns:
        A Tensor that contains the Mel spectrogram.
    """

    if stft_style == "scipy":
        *_, stft = signal.stft(
            x=audio,
            fs=SAMPLE_RATE,
            window=stft_window,
            nperseg=N_FFT,
            noverlap=N_FFT - HOP_LENGTH,
            return_onesided=True,
            padded=True,
        )
    else:
        assert stft_style == "numpy"
        stft = stft_numpy(audio, N_FFT, HOP_LENGTH)
    magnitudes = np.abs(stft[..., :-1]) ** 2

    filters = mel_filters(n_mels, hf_preprocessor_config_json_file)
    mel_spec = magnitudes.T @ filters.T

    res = mel_spec
    np.maximum(res, 1e-10, out=res)
    np.log10(res, out=res)
    np.maximum(res, res.max() - 8.0, out=res)
    np.add(res, 4, out=res)
    np.divide(res, 4, out=res)
    return res


def stft_numpy(
    signal: np.ndarray,
    window_size: int = 400,
    hop_size: int = 160,
    pad_mode: str = 'reflect',
    window_type: str = 'hann',
    center: bool = True
) -> np.ndarray:
    # Ensure the signal is a numpy array
    signal = np.asarray(signal)

    # Create the window function (Hann window)
    if window_type == 'hann':
        window = np.hanning(window_size)
    else:
        raise ValueError(f"Unsupported window type: {window_type}")

    # Padding the signal if needed (based on 'center' and 'pad_mode')
    if center:
        pad_left = window_size // 2
        pad_right = window_size // 2
        signal = np.pad(signal, (pad_left, pad_right), mode=pad_mode)

    # Calculate the number of frames
    num_frames = (len(signal) - window_size) // hop_size + 1

    # Prepare the result array (complex)
    stft_result = np.zeros((num_frames, window_size // 2 + 1), dtype=np.complex64)

    for i in range(num_frames):
        start_idx = i * hop_size
        end_idx = start_idx + window_size
        frame = signal[start_idx:end_idx] * window

        # Compute real FFT, which returns half the bins (positive frequencies)
        fft_result = np.fft.rfft(frame)

        # Compute the FFT of the frame
        stft_result[i, :] = fft_result

    return stft_result.T


def preprocess_audio(
    audio: np.ndarray, hf_preprocessor_config_json_file: Path | None
) -> np.ndarray:
    # Pad/trim it to fit 30 seconds.
    audio = pad_or_trim(audio)
    # Calc log mel spectrogram
    mel = log_mel_spectrogram(
        audio, hf_preprocessor_config_json_file=hf_preprocessor_config_json_file
    )
    return mel


def load_and_preprocess_numpy(
    audio_file: Path, hf_preprocessor_config_json_file: Path | None,
    out_dtype: type | str = np.float32
) -> np.ndarray:
    # Load audio and preprocess audio using numpy.
    audio = load_audio(audio_file)
    mel = preprocess_audio(audio, hf_preprocessor_config_json_file)
    return np.expand_dims(mel.astype(out_dtype), axis=(0, 1))
