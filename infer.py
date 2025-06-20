import os
import math
import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
from pesq import pesq
from pystoi.stoi import stoi
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable

# Initialization
sr = 16000
n_fft_1, n_fft_2 = 512, 256
hop_size = 128
pre_emphasis = 0.97
results = []
ms = []

a = 1
if a == 1:
    clean_dir = "clean_testset_wav"
    noise_dir = "noisy_testset_wav"
    output_dir = "6.output-RNN-TEST"
elif a == 2:
    clean_dir = "clean_test"
    noise_dir = "noisy_test"
    output_dir = "6.output-h-aid"
os.makedirs(output_dir, exist_ok=True)

@register_keras_serializable()
def extract_stats(x):
    mean = tf.reduce_mean(x, axis=-1, keepdims=True)  # shape: (B, T, 1)
    std = tf.math.reduce_std(x, axis=-1, keepdims=True)
    max_ = tf.reduce_max(x, axis=-1, keepdims=True)
    min_ = tf.reduce_min(x, axis=-1, keepdims=True)
    stats = tf.concat([mean, std, max_, min_], axis=-1)  # shape: (B, T, 4)
    return tf.concat([x, stats], axis=-1)  # shape: (B, T, feat_dim + 4)

@register_keras_serializable()
def band_attention_fn(tensor):
    # tensor shape: [B, T, C]
    B = tf.shape(tensor)[0]
    T = tf.shape(tensor)[1]
    C = tf.shape(tensor)[2]
    band_size = tf.cast(C // 8, tf.int32)

    # reshape to [B, T, num_bands, band_size]
    bands = tf.reshape(tensor, [B, T, 8, band_size])

    # calculate attention weights: [B, 1, num_bands, 1]
    band_weights = tf.reduce_mean(bands, axis=1, keepdims=True)         # [B, 1, num_bands, band_size]
    band_weights = tf.reduce_mean(band_weights, axis=-1, keepdims=True) # [B, 1, num_bands, 1]
    band_weights = tf.sigmoid(band_weights)

    # apply attention
    bands = bands * band_weights

    # reshape back to [B, T, C]
    out = tf.reshape(bands, [B, T, C])
    return out

def predict(test_noisy):
    pre_emphasis = 0.97
    emphasized = np.append(test_noisy[0], test_noisy[1:] - pre_emphasis * test_noisy[:-1])
    stft1 = librosa.stft(emphasized, n_fft=n_fft_1, hop_length=hop_size)
    stft2 = librosa.stft(emphasized, n_fft=n_fft_2, hop_length=hop_size)
    log_mag1 = np.log1p(np.abs(stft1))
    log_mag2 = np.log1p(np.abs(stft2))

    min_T = min(log_mag1.shape[1], log_mag2.shape[1])
    log_mag1 = log_mag1[:, :min_T]
    log_mag2 = log_mag2[:, :min_T]
    phase = np.angle(stft1[:, :min_T])  # phase of noisy signal
    combined_log_mag = np.concatenate([log_mag1.T, log_mag2.T], axis=-1)

    mean = np.mean(combined_log_mag)
    std = np.std(combined_log_mag) + 1e-8
    log_mag_norm = (combined_log_mag - mean) / std
    log_mag_norm = log_mag_norm.reshape(-1, 1, num_freq_bins)

    pred_mask = model.predict(log_mag_norm, verbose=0)[:, 0, :]
    combined_mag = np.expm1(combined_log_mag)  # inverse of log1p
    estimated_mag = combined_mag * pred_mask   # apply mask in linear domain
    estimated_mag1 = estimated_mag[:, :log_mag1.shape[0]].T  # transpose back to [freq, time]
    estimated_stft = estimated_mag1 * np.exp(1j * phase)
    denoised_audio = librosa.istft(estimated_stft, hop_length=hop_size)
    denoised_audio /= np.max(np.abs(denoised_audio) + 1e-8)

    return denoised_audio

def calculate_psnr_mel(original_signal, denoised_signal, sr=16000, n_mels=128, n_fft=512, hop_length=128):
    min_len = min(len(original_signal), len(denoised_signal))
    original_signal = original_signal[:min_len]
    denoised_signal = denoised_signal[:min_len]

    mel_original = librosa.feature.melspectrogram(y=original_signal, sr=sr, n_mels=n_mels,
                                                   n_fft=n_fft, hop_length=hop_length)
    mel_denoised = librosa.feature.melspectrogram(y=denoised_signal, sr=sr, n_mels=n_mels,
                                                   n_fft=n_fft, hop_length=hop_length)
    min_frames = min(mel_original.shape[1], mel_denoised.shape[1])
    mel_original = mel_original[:, :min_frames]
    mel_denoised = mel_denoised[:, :min_frames]

    log_mel_original = np.log1p(mel_original)
    log_mel_denoised = np.log1p(mel_denoised)

    mse = np.mean((log_mel_original - log_mel_denoised) ** 2)
    if mse == 0:
        return float('inf'), 0
    psnr = 20 * math.log10(np.max(log_mel_original) / math.sqrt(mse))
    return psnr, mse

def compute_stft_features(signal):
    pre_emphasis = 0.97
    emphasized = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    stft1 = librosa.stft(emphasized, n_fft=n_fft_1, hop_length=hop_size, window="hann")
    stft2 = librosa.stft(emphasized, n_fft=n_fft_2, hop_length=hop_size, window="hann")
    log_mag1 = np.log1p(np.abs(stft1))
    log_mag2 = np.log1p(np.abs(stft2))
    return np.concatenate([log_mag1.T, log_mag2.T], axis=-1)

clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith(".wav")])
noise_files = sorted([f for f in os.listdir(noise_dir) if f.endswith(".wav")])
X_train = []

for clean_file in clean_files:
    noise_file = clean_file.replace("M_", "H_")
    if noise_file in noise_files:
        clean, _ = librosa.load(os.path.join(clean_dir, clean_file), sr=sr)
        noise, _ = librosa.load(os.path.join(noise_dir, noise_file), sr=sr)
        clean /= np.max(np.abs(clean) + 1e-8)
        noise /= np.max(np.abs(noise) + 1e-8)
        x = compute_stft_features(noise)
        X_train.append(x)

X_train = np.vstack(X_train)
mean, std = np.mean(X_train), np.std(X_train)
num_freq_bins = X_train.shape[-1]

model = load_model("model.h5", compile=False, custom_objects={
    'mse': MeanSquaredError,
    'band_attention_fn': band_attention_fn,
    'stat_concat': extract_stats
})
model.summary()

for i, (clean_fname, noise_fname) in enumerate(zip(clean_files, noise_files)):
    clean_path = os.path.join(clean_dir, clean_fname)
    noise_path = os.path.join(noise_dir, noise_fname)

    test_noisy, _ = librosa.load(noise_path, sr=sr)
    test_clean, _ = librosa.load(clean_path, sr=sr)

    # start = time.perf_counter()
    denoised_audio = predict(test_noisy)
    # end = time.perf_counter()
    # ms.append((end - start) / 20 * 1000)
    # print(f"delay: {(end - start) / 20 * 1000:.2f} ms")

    psnr, mse = calculate_psnr_mel(test_clean, denoised_audio, sr=sr)
    min_len = min(len(test_clean), len(denoised_audio))
    pesq_score = pesq(sr, test_clean[:min_len], denoised_audio[:min_len], 'wb')
    stoi_score = stoi(test_clean[:min_len], denoised_audio[:min_len], sr, extended=False)

    print(f"[{i+1}] {noise_fname} -> PSNR: {psnr:.2f} dB | MSE: {mse:.6f} | PESQ: {pesq_score:.4f} | STOI: {stoi_score:.4f}")
    results.append((noise_fname, psnr, mse, pesq_score, stoi_score))

    if a == 2:
        sf.write(os.path.join(output_dir, str(noise_path[11:])), denoised_audio, sr)
    elif a == 1:
        sf.write(os.path.join(output_dir, str(noise_path[18:])), denoised_audio, sr)

avg_psnr = np.mean([r[1] for r in results])
avg_mse = np.mean([r[2] for r in results])
avg_pesq = np.mean([r[3] for r in results])
avg_stoi = np.mean([r[4] for r in results])

print(f"\navg PSNR: {avg_psnr:.2f} dB | avg MSE: {avg_mse:.6f} | avg PESQ: {avg_pesq:.4f} | avg STOI: {avg_stoi:.4f}")
