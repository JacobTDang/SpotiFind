"""
Audio Transforms Module

This module handles all audio preprocessing and enhancement to improve
similarity matching between different recording sources (studio vs microphone).
"""
import numpy as np
import scipy.signal
import scipy.ndimage
from typing import Tuple, Dict, Optional, Literal
import logging
from dataclasses import dataclass
from enum import Enum
import warnings

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION AND TYPES
# =============================================================================

class AudioSource(Enum):
    """Types of audio sources we handle"""
    STUDIO = "studio"
    MICROPHONE = "microphone"
    UNKNOWN = "unknown"

@dataclass
class AudioQualityMetrics:
    """Container for audio quality analysis results"""
    snr_db: float
    frequency_range_hz: Tuple[float, float]
    dynamic_range_db: float
    noise_floor_db: float
    spectral_centroid_hz: float
    source_type: AudioSource
    confidence: float

@dataclass
class ProcessingConfig:
    """Configuration for different processing pipelines"""
    # Frequency processing
    highpass_cutoff: float = 80
    lowpass_cutoff: float = 8000
    spectral_smoothing_kernel: int = 5

    # Dynamic range
    noise_gate_db: float = -40
    dynamic_expansion_ratio: float = 1.2

    # Enhancement
    harmonic_enhancement: bool = True
    noise_reduction_strength: float = 0.3

# =============================================================================
# STAGE 1: AUDIO QUALITY ANALYSIS
# =============================================================================

def analyze_audio_quality(audio: np.ndarray, sr: int) -> AudioQualityMetrics:
    """
    Analyze audio to determine source type and quality metrics.
    This determines which processing pipeline to use.
    """
    # Calculate SNR
    snr_db = _calculate_snr(audio)

    # Analyze frequency content
    freq_range = _analyze_frequency_range(audio, sr)

    # Calculate dynamic range
    dynamic_range_db = _calculate_dynamic_range(audio)

    # Estimate noise floor
    noise_floor_db = _estimate_noise_floor(audio)

    # Calculate spectral centroid (brightness measure)
    spectral_centroid = _calculate_spectral_centroid(audio, sr)

    # Classify source type based on metrics
    source_type, confidence = _classify_audio_source(
        snr_db, freq_range, dynamic_range_db, spectral_centroid
    )

    return AudioQualityMetrics(
        snr_db=snr_db,
        frequency_range_hz=freq_range,
        dynamic_range_db=dynamic_range_db,
        noise_floor_db=noise_floor_db,
        spectral_centroid_hz=spectral_centroid,
        source_type=source_type,
        confidence=confidence
    )

def _calculate_snr(audio: np.ndarray) -> float:
    """Calculate signal-to-noise ratio in dB using segment-based analysis"""
    if len(audio) == 0:
        return -np.inf

    # Remove DC offset first
    audio = audio - np.mean(audio)
    abs_audio = np.abs(audio)

    # Use segment-based analysis for robust SNR calculation
    segment_length = min(1024, len(audio) // 10)  # At least 10 segments

    if segment_length < 64:
        # Fallback to percentile method for very short audio
        signal_level = np.percentile(abs_audio, 95)
        noise_level = np.percentile(abs_audio, 5)

        if noise_level <= 0:
            return 60.0

        return float(20 * np.log10(signal_level / noise_level))

    # Split into segments and calculate RMS for each
    num_segments = len(audio) // segment_length
    segments = abs_audio[:num_segments * segment_length].reshape(num_segments, segment_length)
    segment_rms = np.sqrt(np.mean(segments**2, axis=1))

    # Signal: 90th percentile of segment RMS (music content)
    # Noise: 10th percentile of segment RMS (background noise)
    signal_power = np.percentile(segment_rms, 90)
    noise_power = np.percentile(segment_rms, 10)

    if noise_power <= 0:
        return 60.0

    snr_db = 10 * np.log10(signal_power / noise_power)
    return float(np.clip(snr_db, -20, 80))  # Reasonable bounds

def _analyze_frequency_range(audio: np.ndarray, sr: int) -> Tuple[float, float]:
    """Determine effective frequency range using FFT and -3dB points"""
    if len(audio) < 256:
        return (0.0, float(sr / 2))  # Nyquist frequency

    # Calculate power spectrum
    fft = np.fft.rfft(audio)
    power_spectrum = np.abs(fft) ** 2

    # Smooth the spectrum to avoid noise artifacts
    if len(power_spectrum) > 10:
        # Apply moving average smoothing
        kernel_size = max(3, len(power_spectrum) // 100)
        kernel = np.ones(kernel_size) / kernel_size
        power_spectrum = np.convolve(power_spectrum, kernel, mode='same')

    # Convert to dB scale
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        power_db = 10 * np.log10(power_spectrum + 1e-10)

    # Find maximum and -3dB points
    max_db = np.max(power_db)
    threshold_db = max_db - 3.0  # -3dB point

    # Frequency bins
    freqs = np.fft.rfftfreq(len(audio), d=1/sr)

    # Find first and last bins above threshold
    above_threshold = power_db >= threshold_db

    if not np.any(above_threshold):
        return (0.0, float(sr / 2))

    # Find frequency range
    indices = np.where(above_threshold)[0]
    low_freq = float(freqs[indices[0]])
    high_freq = float(freqs[indices[-1]])

    # Ensure reasonable bounds
    low_freq = max(low_freq, 20.0)  # Minimum audible frequency
    high_freq = min(high_freq, sr / 2)  # Nyquist limit

    return (low_freq, high_freq)

def _calculate_dynamic_range(audio: np.ndarray) -> float:
    """Calculate dynamic range as difference between RMS of loudest and quietest sections"""
    if len(audio) == 0:
        return 0.0

    # Use frame-based analysis
    frame_length = min(2048, len(audio) // 20)  # At least 20 frames

    if frame_length < 128:
        # Fallback for very short audio
        abs_audio = np.abs(audio)
        loud_level = np.percentile(abs_audio, 95)
        quiet_level = np.percentile(abs_audio, 5)

        if quiet_level <= 0:
            return 60.0

        return float(20 * np.log10(loud_level / quiet_level))

    # Split into overlapping frames
    frame_shift = frame_length // 2
    frames = []

    for i in range(0, len(audio) - frame_length, frame_shift):
        frame = audio[i:i + frame_length]
        frames.append(frame)

    if len(frames) < 2:
        return 20.0  # Default reasonable value

    frames = np.array(frames)

    # Calculate RMS for each frame
    frame_rms = np.sqrt(np.mean(frames**2, axis=1))

    # Remove zero/very small values
    frame_rms = frame_rms[frame_rms > 1e-8]

    if len(frame_rms) < 2:
        return 20.0

    # Dynamic range: difference between loudest and quietest RMS values
    loud_rms = np.percentile(frame_rms, 95)
    quiet_rms = np.percentile(frame_rms, 5)

    if quiet_rms <= 0:
        return 60.0

    dynamic_range = 20 * np.log10(loud_rms / quiet_rms)
    return float(np.clip(dynamic_range, 0, 80))

def _estimate_noise_floor(audio: np.ndarray) -> float:
    """Estimate noise floor level using statistical analysis"""
    if len(audio) == 0:
        return -60.0

    abs_audio = np.abs(audio)

    # Remove outliers and use lower percentiles
    # Noise floor is typically in the bottom 10-20% of amplitude values
    noise_samples = abs_audio[abs_audio <= np.percentile(abs_audio, 20)]

    if len(noise_samples) == 0:
        return -60.0

    # RMS of noise samples
    noise_rms = np.sqrt(np.mean(noise_samples**2))

    if noise_rms <= 0:
        return -60.0

    # Convert to dB relative to full scale (assuming audio is normalized to [-1, 1])
    noise_floor_db = 20 * np.log10(noise_rms)

    return float(np.clip(noise_floor_db, -80, 0))

def _calculate_spectral_centroid(audio: np.ndarray, sr: int) -> float:
    """Calculate spectral centroid (measure of brightness)"""
    if LIBROSA_AVAILABLE:
        try:
            # Use librosa's optimized implementation
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            return float(np.mean(centroid))
        except Exception:
            pass  # Fall back to manual implementation

    # Manual implementation using FFT
    if len(audio) < 256:
        return float(sr / 4)  # Default to quarter of sample rate

    # Calculate magnitude spectrum
    fft = np.fft.rfft(audio)
    magnitude = np.abs(fft)

    # Frequency bins
    freqs = np.fft.rfftfreq(len(audio), d=1/sr)

    # Calculate weighted average frequency (spectral centroid)
    if np.sum(magnitude) == 0:
        return float(sr / 4)

    centroid = np.sum(freqs * magnitude) / np.sum(magnitude)

    # Ensure reasonable bounds
    centroid = np.clip(centroid, 20, sr / 2)

    return float(centroid)

def _classify_audio_source(snr_db: float, freq_range: Tuple[float, float],
                          dynamic_range_db: float, spectral_centroid: float) -> Tuple[AudioSource, float]:
    """Classify audio source based on quality metrics"""

    low_freq, high_freq = freq_range

    # Studio recording characteristics:
    # - High SNR (> 25 dB)
    # - Full frequency range (< 100 Hz low, > 10 kHz high)
    # - Good dynamic range (> 20 dB)

    studio_score = 0.0
    microphone_score = 0.0

    # SNR scoring
    if snr_db > 30:
        studio_score += 0.4
    elif snr_db > 20:
        studio_score += 0.2
        microphone_score += 0.1
    elif snr_db > 10:
        microphone_score += 0.3
    else:
        microphone_score += 0.4

    # Frequency range scoring
    if low_freq < 100 and high_freq > 10000:
        studio_score += 0.3
    elif low_freq < 200 and high_freq > 8000:
        studio_score += 0.1
        microphone_score += 0.1
    else:
        microphone_score += 0.3

    # Dynamic range scoring
    if dynamic_range_db > 25:
        studio_score += 0.2
    elif dynamic_range_db > 15:
        studio_score += 0.1
        microphone_score += 0.1
    else:
        microphone_score += 0.2

    # Spectral centroid (brightness) - studio recordings often more balanced
    if 1000 <= spectral_centroid <= 4000:
        studio_score += 0.1
    elif spectral_centroid > 5000 or spectral_centroid < 500:
        microphone_score += 0.1

    # Make decision
    if studio_score > microphone_score and studio_score > 0.6:
        return AudioSource.STUDIO, min(0.95, studio_score)
    elif microphone_score > 0.4:
        return AudioSource.MICROPHONE, min(0.95, microphone_score)
    else:
        return AudioSource.UNKNOWN, 0.5

# =============================================================================
# STAGE 2: FREQUENCY DOMAIN COMPENSATION
# =============================================================================

def apply_spectral_whitening(audio: np.ndarray, sr: int, strength: float = 0.5) -> np.ndarray:
    """Apply spectral whitening to normalize frequency response"""
    if not LIBROSA_AVAILABLE:
        return _spectral_whitening_manual(audio, sr, strength)

    try:
        # Use librosa's STFT for high-quality processing
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude, phase = np.abs(stft), np.angle(stft)

        # Calculate spectral envelope (smoothed magnitude)
        # Use median filtering to get smooth envelope
        envelope = scipy.ndimage.median_filter(magnitude, size=(5, 1))
        envelope = np.maximum(envelope, np.max(envelope) * 0.01)  # Avoid division by zero

        # Apply whitening: reduce spectral variations
        whitened_magnitude = magnitude / (envelope ** strength)

        # Reconstruct signal
        whitened_stft = whitened_magnitude * np.exp(1j * phase)
        whitened_audio = librosa.istft(whitened_stft, hop_length=512)

        return whitened_audio[:len(audio)]  # Ensure same length

    except Exception as e:
        logger.warning(f"Librosa spectral whitening failed: {e}, using manual method")
        return _spectral_whitening_manual(audio, sr, strength)

def _spectral_whitening_manual(audio: np.ndarray, sr: int, strength: float) -> np.ndarray:
    """Manual implementation of spectral whitening using scipy"""
    try:
        # Use scipy's STFT
        f, t, stft = scipy.signal.stft(audio, fs=sr, nperseg=2048, noverlap=1536)
        magnitude, phase = np.abs(stft), np.angle(stft)

        # Simple spectral envelope estimation
        # Smooth across frequency dimension
        envelope = scipy.ndimage.uniform_filter1d(magnitude, size=5, axis=0)
        envelope = np.maximum(envelope, np.max(envelope) * 0.01)

        # Apply whitening
        whitened_magnitude = magnitude / (envelope ** strength)

        # Reconstruct
        whitened_stft = whitened_magnitude * np.exp(1j * phase)
        _, whitened_audio = scipy.signal.istft(whitened_stft, fs=sr, nperseg=2048, noverlap=1536)

        return whitened_audio[:len(audio)]

    except Exception as e:
        logger.warning(f"Manual spectral whitening failed: {e}")
        return audio  # Return original if processing fails

def apply_frequency_response_correction(audio: np.ndarray, sr: int,
                                      source_type: AudioSource) -> np.ndarray:
    """Apply frequency response correction based on detected source type"""
    if source_type == AudioSource.MICROPHONE:
        return _apply_microphone_eq(audio, sr)
    elif source_type == AudioSource.STUDIO:
        return _apply_studio_eq(audio, sr)
    else:
        return audio

def _apply_microphone_eq(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply EQ curve to compensate for microphone limitations"""
    try:
        # Design filters to compensate for typical microphone issues
        nyquist = sr / 2

        # 1. High-pass filter to remove rumble (below 80 Hz)
        high_cutoff = min(80, nyquist * 0.1)
        sos_hp = scipy.signal.butter(2, high_cutoff / nyquist, btype='high', output='sos')
        audio = scipy.signal.sosfilt(sos_hp, audio)

        # 2. Gentle bass boost (100-300 Hz) that mics typically miss
        if sr > 1000:  # Only if sample rate allows
            bass_freq = min(200, nyquist * 0.2)
            bass_bandwidth = min(100, nyquist * 0.1)

            # Simple peaking filter approximation
            w0 = 2 * np.pi * bass_freq / sr
            Q = bass_freq / bass_bandwidth
            A = np.sqrt(1.5)  # +1.5 dB boost

            # Biquad coefficients for peaking filter
            alpha = np.sin(w0) / (2 * Q)
            cos_w0 = np.cos(w0)

            b0 = 1 + alpha * A
            b1 = -2 * cos_w0
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * cos_w0
            a2 = 1 - alpha / A

            # Apply filter
            audio = scipy.signal.lfilter([b0/a0, b1/a0, b2/a0], [1, a1/a0, a2/a0], audio)

        # 3. Slight high-frequency boost (4-8 kHz) for presence
        if sr > 16000:
            presence_freq = min(6000, nyquist * 0.7)
            sos_shelf = scipy.signal.butter(1, presence_freq / nyquist, btype='high', output='sos')
            boosted = scipy.signal.sosfilt(sos_shelf, audio)
            audio = audio + 0.1 * boosted  # Gentle boost

        return audio

    except Exception as e:
        logger.warning(f"Microphone EQ failed: {e}")
        return audio

def _apply_studio_eq(audio: np.ndarray, sr: int) -> np.ndarray:
    """Light EQ for studio recordings - mainly just cleanup"""
    try:
        # Very gentle high-pass to remove any DC offset or subsonic content
        if sr > 200:
            sos_hp = scipy.signal.butter(1, 20 / (sr/2), btype='high', output='sos')
            audio = scipy.signal.sosfilt(sos_hp, audio)

        return audio

    except Exception as e:
        logger.warning(f"Studio EQ failed: {e}")
        return audio

# =============================================================================
# STAGE 3: DYNAMIC RANGE PROCESSING
# =============================================================================

def normalize_dynamic_range(audio: np.ndarray, source_type: AudioSource,
                           target_range_db: float = 20) -> np.ndarray:
    """Normalize dynamic range to improve consistency between sources"""
    if source_type == AudioSource.MICROPHONE:
        audio = _expand_compressed_dynamics(audio, target_range_db)

    # Apply gentle compression to all sources
    audio = _apply_gentle_compression(audio)

    return audio

def _expand_compressed_dynamics(audio: np.ndarray, target_range_db: float) -> np.ndarray:
    """Attempt to restore dynamic range to compressed audio"""
    try:
        # Calculate envelope using Hilbert transform
        analytic_signal = scipy.signal.hilbert(audio)
        envelope = np.abs(analytic_signal)

        # Smooth the envelope
        if len(envelope) > 100:
            envelope = scipy.ndimage.uniform_filter1d(envelope, size=len(envelope)//100)

        # Avoid division by zero
        envelope = np.maximum(envelope, np.max(envelope) * 0.001)

        # Apply expansion: enhance quiet parts relative to loud parts
        # This is a simple upward expansion
        expansion_ratio = 1.2
        normalized_env = envelope / np.max(envelope)
        expansion_factor = normalized_env ** (1/expansion_ratio - 1)

        # Apply expansion
        expanded_audio = audio * (1 + 0.3 * expansion_factor)

        return expanded_audio

    except Exception as e:
        logger.warning(f"Dynamic expansion failed: {e}")
        return audio

def _apply_gentle_compression(audio: np.ndarray) -> np.ndarray:
    """Apply gentle compression for consistency"""
    try:
        # Simple soft-knee compressor
        threshold = 0.7  # Compression threshold
        ratio = 3.0      # Compression ratio

        # Calculate envelope for gain reduction
        envelope = np.abs(audio)
        if len(envelope) > 50:
            envelope = scipy.ndimage.uniform_filter1d(envelope, size=50)

        # Apply compression above threshold
        gain_reduction = np.ones_like(envelope)
        over_threshold = envelope > threshold

        if np.any(over_threshold):
            excess = envelope[over_threshold] - threshold
            compressed_excess = excess / ratio
            gain_reduction[over_threshold] = (threshold + compressed_excess) / envelope[over_threshold]

        # Smooth gain changes to avoid artifacts
        if len(gain_reduction) > 10:
            gain_reduction = scipy.ndimage.uniform_filter1d(gain_reduction, size=10)

        compressed_audio = audio * gain_reduction

        return compressed_audio

    except Exception as e:
        logger.warning(f"Gentle compression failed: {e}")
        return audio

def remove_agc_artifacts(audio: np.ndarray, sr: int) -> np.ndarray:
    """Remove automatic gain control artifacts common in mobile recordings"""
    try:
        # AGC artifacts appear as sudden gain changes
        # Calculate short-term RMS to detect sudden changes
        frame_length = int(0.1 * sr)  # 100ms frames

        if frame_length >= len(audio):
            return audio

        frame_rms = []
        for i in range(0, len(audio) - frame_length, frame_length // 2):
            frame = audio[i:i + frame_length]
            rms = np.sqrt(np.mean(frame**2))
            frame_rms.append(rms)

        frame_rms = np.array(frame_rms)

        if len(frame_rms) < 3:
            return audio

        # Detect sudden changes in RMS (potential AGC artifacts)
        rms_ratio = frame_rms[1:] / (frame_rms[:-1] + 1e-8)
        sudden_changes = (rms_ratio > 2.0) | (rms_ratio < 0.5)

        if not np.any(sudden_changes):
            return audio  # No AGC artifacts detected

        # Simple artifact reduction: smooth the gain variations
        smoothed_rms = scipy.ndimage.uniform_filter1d(frame_rms, size=3)
        gain_correction = smoothed_rms / (frame_rms + 1e-8)

        # Apply correction
        corrected_audio = audio.copy()
        for i, correction in enumerate(gain_correction):
            start_idx = i * frame_length // 2
            end_idx = min(start_idx + frame_length, len(audio))
            corrected_audio[start_idx:end_idx] *= correction

        return corrected_audio

    except Exception as e:
        logger.warning(f"AGC artifact removal failed: {e}")
        return audio

# =============================================================================
# STAGE 4: NOISE AND ARTIFACT REMOVAL
# =============================================================================

def reduce_background_noise(audio: np.ndarray, sr: int, strength: float = 0.3) -> np.ndarray:
    """Reduce background noise using spectral subtraction"""
    try:
        # Use STFT for frequency-domain noise reduction
        if LIBROSA_AVAILABLE:
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        else:
            f, t, stft = scipy.signal.stft(audio, fs=sr, nperseg=2048, noverlap=1536)

        magnitude, phase = np.abs(stft), np.angle(stft)

        # Estimate noise spectrum from quietest frames
        frame_energy = np.sum(magnitude**2, axis=0)
        noise_threshold = np.percentile(frame_energy, 20)  # Bottom 20% are likely noise
        noise_frames = frame_energy <= noise_threshold

        if np.any(noise_frames):
            noise_spectrum = np.mean(magnitude[:, noise_frames], axis=1, keepdims=True)
        else:
            noise_spectrum = np.percentile(magnitude, 10, axis=1, keepdims=True)

        # Spectral subtraction
        alpha = strength * 2.0  # Over-subtraction factor
        cleaned_magnitude = magnitude - alpha * noise_spectrum

        # Ensure we don't create artifacts
        cleaned_magnitude = np.maximum(cleaned_magnitude, 0.1 * magnitude)

        # Reconstruct signal
        cleaned_stft = cleaned_magnitude * np.exp(1j * phase)

        if LIBROSA_AVAILABLE:
            cleaned_audio = librosa.istft(cleaned_stft, hop_length=512)
        else:
            _, cleaned_audio = scipy.signal.istft(cleaned_stft, fs=sr, nperseg=2048, noverlap=1536)

        return cleaned_audio[:len(audio)]

    except Exception as e:
        logger.warning(f"Noise reduction failed: {e}")
        return audio

def remove_codec_artifacts(audio: np.ndarray, sr: int) -> np.ndarray:
    """Remove compression artifacts from lossy codecs"""
    try:
        # Codec artifacts often appear as high-frequency noise and pre-echo
        # Apply gentle low-pass filtering to remove aliasing artifacts

        if sr > 16000:
            # Remove content above 0.9 * Nyquist to eliminate aliasing
            cutoff = 0.9 * sr / 2
            sos = scipy.signal.butter(4, cutoff / (sr/2), btype='low', output='sos')
            audio = scipy.signal.sosfilt(sos, audio)

        # Light smoothing to reduce quantization noise
        if len(audio) > 10:
            audio = scipy.ndimage.uniform_filter1d(audio, size=3, mode='nearest')

        return audio

    except Exception as e:
        logger.warning(f"Codec artifact removal failed: {e}")
        return audio

def reduce_room_reverb(audio: np.ndarray, sr: int) -> np.ndarray:
    """Reduce room reverb using simple dereverberation"""
    try:
        # Simple approach: high-pass filtering and envelope sharpening
        # This won't eliminate reverb but will reduce its impact

        # High-pass filter to reduce low-frequency reverb
        if sr > 400:
            sos_hp = scipy.signal.butter(2, 100 / (sr/2), btype='high', output='sos')
            audio = scipy.signal.sosfilt(sos_hp, audio)

        # Envelope sharpening to reduce reverb tail
        envelope = np.abs(scipy.signal.hilbert(audio))
        if len(envelope) > 20:
            # Sharpen the envelope to reduce reverb decay
            sharpened_env = envelope ** 1.2
            gain_factor = sharpened_env / (envelope + 1e-8)
            audio = audio * gain_factor

        return audio

    except Exception as e:
        logger.warning(f"Reverb reduction failed: {e}")
        return audio

# =============================================================================
# STAGE 5: PERCEPTUAL ALIGNMENT
# =============================================================================

def enhance_harmonics(audio: np.ndarray, sr: int) -> np.ndarray:
    """Enhance harmonic content that microphones often lose"""
    try:
        # Simple harmonic enhancement using frequency domain processing
        fft = np.fft.rfft(audio)
        magnitude, phase = np.abs(fft), np.angle(fft)

        # Enhance harmonic peaks
        # This is a simplified approach - real harmonic enhancement is complex
        freqs = np.fft.rfftfreq(len(audio), d=1/sr)

        # Gentle boost around common musical frequencies
        musical_freqs = [220, 440, 880, 1760]  # A notes
        for freq in musical_freqs:
            if freq < sr / 2:
                freq_idx = np.argmin(np.abs(freqs - freq))
                width = max(1, len(freqs) // 100)
                start_idx = max(0, freq_idx - width)
                end_idx = min(len(magnitude), freq_idx + width)
                magnitude[start_idx:end_idx] *= 1.1  # Small boost

        # Reconstruct
        enhanced_fft = magnitude * np.exp(1j * phase)
        enhanced_audio = np.fft.irfft(enhanced_fft, n=len(audio))

        return enhanced_audio

    except Exception as e:
        logger.warning(f"Harmonic enhancement failed: {e}")
        return audio

def ensure_mono_consistency(audio: np.ndarray) -> np.ndarray:
    """Convert to mono in a consistent way"""
    if audio.ndim > 1:
        if audio.shape[1] == 2:  # Stereo
            # Preserve center channel, attenuate sides slightly
            mono = 0.7 * np.mean(audio, axis=1) + 0.3 * audio[:, 0]
        else:
            # Multi-channel: simple average
            mono = np.mean(audio, axis=1)
        return mono
    return audio

def align_temporal_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """Remove timing jitter and align temporal features"""
    try:
        # Simple temporal alignment using cross-correlation
        # This helps with phase alignment issues from different recording devices

        if len(audio) < sr:  # Less than 1 second, skip processing
            return audio

        # Split audio into segments and align them
        segment_length = sr // 4  # 250ms segments
        num_segments = len(audio) // segment_length

        if num_segments < 2:
            return audio

        aligned_audio = audio.copy()

        # Use first segment as reference
        reference_segment = audio[:segment_length]

        for i in range(1, num_segments):
            start_idx = i * segment_length
            end_idx = start_idx + segment_length

            if end_idx > len(audio):
                break

            segment = audio[start_idx:end_idx]

            # Find best alignment using cross-correlation
            correlation = scipy.signal.correlate(reference_segment, segment, mode='full')
            max_corr_idx = np.argmax(np.abs(correlation))

            # Calculate shift (small shifts only)
            shift = max_corr_idx - len(reference_segment) + 1
            shift = np.clip(shift, -segment_length//10, segment_length//10)  # Limit shift

            # Apply shift if significant
            if abs(shift) > 2:
                if shift > 0:
                    aligned_segment = np.concatenate([np.zeros(shift), segment[:-shift]])
                else:
                    aligned_segment = np.concatenate([segment[-shift:], np.zeros(-shift)])

                aligned_audio[start_idx:end_idx] = aligned_segment

        return aligned_audio

    except Exception as e:
        logger.warning(f"Temporal alignment failed: {e}")
        return audio

# =============================================================================
# STAGE 6: FEATURE-SPACE NORMALIZATION
# =============================================================================

def adaptive_windowing_preprocess(audio: np.ndarray, sr: int,
                                 quality_metrics: AudioQualityMetrics) -> np.ndarray:
    """Apply adaptive preprocessing based on audio quality"""
    if quality_metrics.snr_db < 15:
        # Low quality - apply more aggressive smoothing
        audio = _apply_heavy_smoothing(audio, sr)
    elif quality_metrics.snr_db > 25:
        # High quality - preserve detail
        audio = _apply_light_smoothing(audio, sr)
    else:
        # Medium quality - moderate smoothing
        audio = _apply_medium_smoothing(audio, sr)

    return audio

def _apply_heavy_smoothing(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply heavy smoothing for noisy recordings"""
    try:
        # Multiple passes of light smoothing to avoid artifacts
        smoothed = audio.copy()

        # Temporal smoothing
        if len(smoothed) > 20:
            smoothed = scipy.ndimage.uniform_filter1d(smoothed, size=5, mode='nearest')

        # Spectral smoothing using FFT
        if len(smoothed) > 256:
            fft = np.fft.rfft(smoothed)
            magnitude, phase = np.abs(fft), np.angle(fft)

            # Smooth magnitude spectrum
            if len(magnitude) > 10:
                magnitude = scipy.ndimage.uniform_filter1d(magnitude, size=7, mode='nearest')

            # Reconstruct
            smoothed_fft = magnitude * np.exp(1j * phase)
            smoothed = np.fft.irfft(smoothed_fft, n=len(audio))

        return smoothed

    except Exception as e:
        logger.warning(f"Heavy smoothing failed: {e}")
        return audio

def _apply_light_smoothing(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply light smoothing for clean recordings"""
    try:
        # Very gentle smoothing to preserve detail
        if len(audio) > 6:
            smoothed = scipy.ndimage.uniform_filter1d(audio, size=3, mode='nearest')
            # Blend with original to preserve detail
            return 0.7 * audio + 0.3 * smoothed
        return audio

    except Exception as e:
        logger.warning(f"Light smoothing failed: {e}")
        return audio

def _apply_medium_smoothing(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply medium smoothing for moderate quality recordings"""
    try:
        if len(audio) > 10:
            smoothed = scipy.ndimage.uniform_filter1d(audio, size=5, mode='nearest')
            # Blend with original
            return 0.6 * audio + 0.4 * smoothed
        return audio

    except Exception as e:
        logger.warning(f"Medium smoothing failed: {e}")
        return audio

def apply_context_aware_processing(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply processing based on detected content type"""
    content_type = _detect_content_type(audio, sr)

    if content_type == "vocal":
        audio = _enhance_for_vocals(audio, sr)
    elif content_type == "instrumental":
        audio = _enhance_for_instruments(audio, sr)
    # If "mixed" or unknown, apply no special processing

    return audio

def _detect_content_type(audio: np.ndarray, sr: int) -> str:
    """Detect whether audio is primarily vocal or instrumental"""
    try:
        # Use spectral characteristics to detect content type

        # Calculate spectral features
        spectral_centroid = _calculate_spectral_centroid(audio, sr)

        # Analyze frequency distribution
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(audio), d=1/sr)

        # Vocal energy is typically concentrated in 100-4000 Hz
        vocal_range_mask = (freqs >= 100) & (freqs <= 4000)
        vocal_energy = np.sum(magnitude[vocal_range_mask])

        # Instrumental energy spans wider range
        instrumental_range_mask = (freqs >= 50) & (freqs <= 8000)
        instrumental_energy = np.sum(magnitude[instrumental_range_mask])

        # Calculate energy ratios
        total_energy = np.sum(magnitude)
        if total_energy > 0:
            vocal_ratio = vocal_energy / total_energy

            # Vocal characteristics:
            # - High energy in vocal range (100-4000 Hz)
            # - Spectral centroid in mid-range (500-3000 Hz)
            if vocal_ratio > 0.6 and 500 <= spectral_centroid <= 3000:
                return "vocal"
            # Instrumental characteristics:
            # - More spread out energy
            # - Spectral centroid could be anywhere
            elif vocal_ratio < 0.4:
                return "instrumental"

        return "mixed"  # Default for unclear cases

    except Exception as e:
        logger.warning(f"Content type detection failed: {e}")
        return "mixed"

def _enhance_for_vocals(audio: np.ndarray, sr: int) -> np.ndarray:
    """Vocal-specific enhancement"""
    try:
        # Enhance vocal presence range (1-4 kHz)
        if sr > 8000:
            # Design a gentle peaking filter for vocal presence
            nyquist = sr / 2
            center_freq = min(2500, nyquist * 0.7)  # 2.5 kHz presence boost

            # Simple high-shelf filter for presence
            sos = scipy.signal.butter(1, center_freq / nyquist, btype='high', output='sos')
            enhanced = scipy.signal.sosfilt(sos, audio)

            # Blend with original (subtle enhancement)
            audio = 0.85 * audio + 0.15 * enhanced

        # Reduce sibilance (harsh 's' sounds around 6-8 kHz)
        if sr > 12000:
            sibilance_freq = min(7000, nyquist * 0.8)
            sos_desib = scipy.signal.butter(2, sibilance_freq / nyquist, btype='low', output='sos')
            desissed = scipy.signal.sosfilt(sos_desib, audio)

            # Gentle de-essing
            audio = 0.9 * audio + 0.1 * desissed

        return audio

    except Exception as e:
        logger.warning(f"Vocal enhancement failed: {e}")
        return audio

def _enhance_for_instruments(audio: np.ndarray, sr: int) -> np.ndarray:
    """Instrumental-specific enhancement"""
    try:
        # Enhance clarity and detail for instruments

        # Gentle high-frequency enhancement for detail
        if sr > 8000:
            nyquist = sr / 2
            detail_freq = min(5000, nyquist * 0.6)

            # High-shelf filter for instrumental detail
            sos = scipy.signal.butter(1, detail_freq / nyquist, btype='high', output='sos')
            enhanced = scipy.signal.sosfilt(sos, audio)

            # Subtle enhancement
            audio = 0.9 * audio + 0.1 * enhanced

        # Enhance bass definition for rhythm instruments
        if sr > 1000:
            bass_freq = min(150, nyquist * 0.1)
            sos_bass = scipy.signal.butter(1, bass_freq / nyquist, btype='high', output='sos')
            bass_enhanced = scipy.signal.sosfilt(sos_bass, audio)

            # Very subtle bass enhancement
            audio = 0.95 * audio + 0.05 * bass_enhanced

        return audio

    except Exception as e:
        logger.warning(f"Instrumental enhancement failed: {e}")
        return audio

# =============================================================================
# MAIN PIPELINE FUNCTIONS
# =============================================================================

def process_studio_recording(audio: np.ndarray, sr: int) -> np.ndarray:
    """Minimal processing pipeline for high-quality studio recordings"""
    try:
        # Light processing to ensure consistency
        audio = ensure_mono_consistency(audio)
        audio = _apply_studio_eq(audio, sr)
        audio = _apply_light_smoothing(audio, sr)
        audio = remove_codec_artifacts(audio, sr)  # Remove any digital artifacts

        return audio

    except Exception as e:
        logger.error(f"Studio processing pipeline failed: {e}")
        return ensure_mono_consistency(audio)  # Minimal fallback

def process_microphone_recording(audio: np.ndarray, sr: int,
                               quality_metrics: AudioQualityMetrics) -> np.ndarray:
    """Aggressive processing pipeline for microphone recordings"""
    try:
        # Stage 1: Basic cleanup
        audio = ensure_mono_consistency(audio)
        audio = reduce_background_noise(audio, sr, strength=0.4)

        # Stage 2: Frequency compensation
        audio = apply_frequency_response_correction(audio, sr, AudioSource.MICROPHONE)
        audio = apply_spectral_whitening(audio, sr, strength=0.6)

        # Stage 3: Dynamic processing
        audio = normalize_dynamic_range(audio, AudioSource.MICROPHONE)
        audio = remove_agc_artifacts(audio, sr)

        # Stage 4: Enhancement
        audio = enhance_harmonics(audio, sr)
        audio = adaptive_windowing_preprocess(audio, sr, quality_metrics)

        # Stage 5: Final cleanup
        audio = reduce_room_reverb(audio, sr)
        audio = align_temporal_features(audio, sr)
        audio = apply_context_aware_processing(audio, sr)

        return audio

    except Exception as e:
        logger.error(f"Microphone processing pipeline failed: {e}")
        # Minimal fallback processing
        try:
            audio = ensure_mono_consistency(audio)
            audio = _apply_microphone_eq(audio, sr)
            return audio
        except:
            return ensure_mono_consistency(audio)

def transform_audio_for_similarity(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Main entry point for audio transformation pipeline.
    Automatically routes to appropriate processing based on detected quality.
    """
    try:
        # Input validation
        if not validate_audio_input(audio, sr):
            logger.warning("Invalid audio input detected")
            return audio

        # Stage 1: Analyze audio quality
        quality_metrics = analyze_audio_quality(audio, sr)

        logger.info(f"Detected audio source: {quality_metrics.source_type.value} "
                    f"(confidence: {quality_metrics.confidence:.2f}, SNR: {quality_metrics.snr_db:.1f} dB)")

        # Stage 2: Route to appropriate pipeline
        if quality_metrics.source_type == AudioSource.STUDIO:
            processed_audio = process_studio_recording(audio, sr)
        elif quality_metrics.source_type == AudioSource.MICROPHONE:
            processed_audio = process_microphone_recording(audio, sr, quality_metrics)
        else:
            # Unknown source - use conservative microphone processing
            logger.info("Unknown audio source, using microphone processing pipeline")
            processed_audio = process_microphone_recording(audio, sr, quality_metrics)

        # Stage 3: Final normalization
        processed_audio = _final_normalization(processed_audio)

        # Log processing results
        logger.info(f"Audio processing complete. Original length: {len(audio)}, "
                   f"Processed length: {len(processed_audio)}")

        return processed_audio

    except Exception as e:
        logger.error(f"Audio transformation pipeline failed: {e}")
        # Ultimate fallback - just normalize and return
        try:
            return _final_normalization(ensure_mono_consistency(audio))
        except:
            return audio

def _final_normalization(audio: np.ndarray) -> np.ndarray:
    """Final normalization step"""
    try:
        # Remove any DC offset
        audio = audio - np.mean(audio)

        # Ensure audio is in [-1, 1] range
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95  # Leave small headroom

        # Remove any remaining outliers
        audio = np.clip(audio, -1.0, 1.0)

        return audio

    except Exception as e:
        logger.warning(f"Final normalization failed: {e}")
        return np.clip(audio, -1.0, 1.0)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_audio_input(audio: np.ndarray, sr: int) -> bool:
    """Validate audio input before processing"""
    try:
        if len(audio) == 0:
            logger.warning("Empty audio array")
            return False
        if sr <= 0:
            logger.warning(f"Invalid sample rate: {sr}")
            return False
        if np.all(audio == 0):
            logger.warning("Audio contains only zeros")
            return False
        if not np.all(np.isfinite(audio)):
            logger.warning("Audio contains non-finite values")
            return False
        if np.max(np.abs(audio)) > 100:  # Unreasonably large values
            logger.warning("Audio values are unexpectedly large")
            return False

        return True

    except Exception as e:
        logger.error(f"Audio validation failed: {e}")
        return False

def get_processing_recommendations(quality_metrics: AudioQualityMetrics) -> Dict[str, str]:
    """Get human-readable recommendations for audio quality"""
    recommendations = {}

    if quality_metrics.snr_db < 10:
        recommendations["noise"] = "Consider recording in a quieter environment"

    if quality_metrics.frequency_range_hz[1] < 8000:
        recommendations["frequency"] = "Try using a better microphone for fuller frequency response"

    if quality_metrics.dynamic_range_db < 10:
        recommendations["dynamics"] = "Audio appears heavily compressed - avoid automatic gain control"

    if quality_metrics.spectral_centroid_hz > 5000:
        recommendations["brightness"] = "Audio seems very bright - might be overdriven or have microphone issues"

    if quality_metrics.spectral_centroid_hz < 500:
        recommendations["dullness"] = "Audio seems dull - check microphone placement and frequency response"

    if quality_metrics.confidence < 0.6:
        recommendations["uncertainty"] = "Audio quality is difficult to classify - results may vary"

    return recommendations

# =============================================================================
# TESTING AND DEBUGGING FUNCTIONS
# =============================================================================

def test_audio_transform_pipeline():
    """Test the complete audio transformation pipeline with synthetic data"""
    print("Testing Audio Transform Pipeline...")

    # Create test signals
    sr = 22050
    duration = 2.0  # 2 seconds
    t = np.linspace(0, duration, int(sr * duration))

    # Test 1: Clean sine wave (should be classified as studio)
    clean_sine = np.sin(2 * np.pi * 440 * t)  # 440 Hz

    print("\n1. Testing with clean sine wave:")
    processed_clean = transform_audio_for_similarity(clean_sine, sr)

    # Test 2: Noisy sine wave (should be classified as microphone)
    noisy_sine = clean_sine + 0.1 * np.random.normal(0, 1, len(clean_sine))

    print("\n2. Testing with noisy sine wave:")
    processed_noisy = transform_audio_for_similarity(noisy_sine, sr)

    # Test 3: Complex music-like signal
    # Multiple harmonics + amplitude modulation + noise
    complex_signal = (
        np.sin(2 * np.pi * 220 * t) +
        0.5 * np.sin(2 * np.pi * 440 * t) +
        0.3 * np.sin(2 * np.pi * 880 * t)
    )
    complex_signal *= (0.7 + 0.3 * np.sin(2 * np.pi * 2 * t))  # AM
    complex_signal += 0.05 * np.random.normal(0, 1, len(complex_signal))

    print("\n3. Testing with complex music-like signal:")
    processed_complex = transform_audio_for_similarity(complex_signal, sr)

    print("\nPipeline testing complete!")

    return {
        'clean': processed_clean,
        'noisy': processed_noisy,
        'complex': processed_complex
    }

def analyze_processing_effects(original: np.ndarray, processed: np.ndarray, sr: int) -> Dict:
    """Analyze the effects of audio processing"""
    try:
        # Calculate quality metrics before and after
        original_metrics = analyze_audio_quality(original, sr)
        processed_metrics = analyze_audio_quality(processed, sr)

        # Calculate additional metrics
        rms_change = np.sqrt(np.mean(processed**2)) / np.sqrt(np.mean(original**2))
        spectral_change = _calculate_spectral_centroid(processed, sr) / _calculate_spectral_centroid(original, sr)

        return {
            'snr_improvement': processed_metrics.snr_db - original_metrics.snr_db,
            'dynamic_range_change': processed_metrics.dynamic_range_db - original_metrics.dynamic_range_db,
            'rms_ratio': rms_change,
            'spectral_centroid_ratio': spectral_change,
            'original_source': original_metrics.source_type.value,
            'processed_source': processed_metrics.source_type.value
        }

    except Exception as e:
        logger.error(f"Processing analysis failed: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    # Run tests if module is executed directly
    test_results = test_audio_transform_pipeline()
