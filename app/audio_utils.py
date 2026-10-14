import numpy as np
import math
from scipy.signal import butter, lfilter, fftconvolve, spectrogram
import wave
import os

def generate_waveform(t, freq, waveform_type):
    if waveform_type == 'Sine':
        return np.sin(2 * np.pi * freq * t)
    elif waveform_type == 'Square':
        return np.sign(np.sin(2 * np.pi * freq * t))
    elif waveform_type == 'Sawtooth':
        return 2 * (t * freq - np.floor(0.5 + t * freq))
    elif waveform_type == 'Triangle':
        return 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - 1
    else:
        return np.sin(2 * np.pi * freq * t)

def generate_waveform_from_phase(phase, waveform_type):
    if waveform_type == 'Sine':
        return np.sin(phase)
    elif waveform_type == 'Square':
        return np.sign(np.sin(phase))
    elif waveform_type == 'Sawtooth':
        return 2 * (phase / (2 * np.pi) - np.floor(0.5 + phase / (2 * np.pi)))
    elif waveform_type == 'Triangle':
        return 2 * np.abs(2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))) - 1
    else:
        return np.sin(phase)

def fm_synthesis_block(num_frames, sr, p, state, dynamic_params=None):
    """
    Generate a block of FM audio. p has:
      carrier_freq, modulator_freq, modulation_index,
      carrier_waveform, modulator_waveform,
      lfo_freq, lfo_depth,
      attack, decay, sustain, release,
      amplitude
    state has 'phase_c', 'phase_m', 'lfo_phase', 'sample_count'
    dynamic_params (optional) can contain 'modulation_index_envelope', 'envelope'
    """
    c_freq = p['carrier_freq']
    m_freq = p['modulator_freq']
    m_index = p['modulation_index']
    c_wf = p['carrier_waveform']
    m_wf = p['modulator_waveform']
    lfo_f = p['lfo_freq']
    lfo_d = p['lfo_depth']
    amplitude = p['amplitude']
    sustain_level = p['sustain'] # Renamed to avoid conflict with ADSR sustain

    block_t = (np.arange(num_frames, dtype=np.float32) + state['sample_count']) / sr
    state['sample_count'] += num_frames

    # LFO
    lfo_phase = 2.0 * np.pi * lfo_f * block_t + state['lfo_phase']
    lfo_sig = np.sin(lfo_phase) * lfo_d

    # Dynamic modulation index from envelope if provided
    current_mod_index = m_index
    if dynamic_params and 'modulation_index_envelope' in dynamic_params:
        # Assuming dynamic_params['modulation_index_envelope'] is pre-calculated for the whole duration
        # Need to sample the correct part for this block
        start_idx = state['sample_count'] - num_frames
        end_idx = state['sample_count']
        if end_idx > len(dynamic_params['modulation_index_envelope']):
            # Handle end of envelope or loop
            env_segment = dynamic_params['modulation_index_envelope'][start_idx:]
            env_segment = np.pad(env_segment, (0, num_frames - len(env_segment)), 'edge')
        else:
            env_segment = dynamic_params['modulation_index_envelope'][start_idx:end_idx]
        current_mod_index = m_index * env_segment

    # ADSR envelope
    current_envelope = np.ones(num_frames)
    if dynamic_params and 'envelope' in dynamic_params:
        start_idx = state['sample_count'] - num_frames
        end_idx = state['sample_count']
        if end_idx > len(dynamic_params['envelope']):
            env_segment = dynamic_params['envelope'][start_idx:]
            env_segment = np.pad(env_segment, (0, num_frames - len(env_segment)), 'edge')
        else:
            env_segment = dynamic_params['envelope'][start_idx:end_idx]
        current_envelope = env_segment

    # slightly modulate carrier freq
    dynamic_carrier = c_freq + c_freq * (0.01 * lfo_sig)

    # build modulator signal
    dt = np.arange(num_frames, dtype=np.float32) / sr
    ph_m0 = state['phase_m']
    phase_m_arr = ph_m0 + 2.0 * np.pi * m_freq * dt
    mod_arr = generate_waveform_from_phase(phase_m_arr, m_wf) * current_mod_index

    out = np.zeros(num_frames, dtype=np.float32)
    ph_c = state['phase_c']
    for i in range(num_frames):
        freq_here = dynamic_carrier[i]
        ph_c += (2.0 * np.pi * freq_here / sr) + mod_arr[i]
        out[i] = generate_waveform_from_phase(ph_c, c_wf)

    # update oscillator phases
    state['phase_c'] = ph_c % (2.0 * math.pi)
    last_phase_m = phase_m_arr[-1] + 2.0 * math.pi * (m_freq / sr)
    state['phase_m'] = last_phase_m % (2.0 * math.pi)
    last_lfo = lfo_phase[-1] + 2.0 * math.pi * lfo_f / sr
    state['lfo_phase'] = last_lfo % (2.0 * math.pi)

    # apply amplitude and sustain
    out *= amplitude * sustain_level * current_envelope

    # convert to stereo
    stereo = np.column_stack((out, out)).astype(np.float32)
    return stereo

def adsr_envelope(num_samples, framerate, attack, decay, sustain, release):
    envelope = np.zeros(num_samples)
    attack_samples = int(framerate * attack)
    decay_samples = int(framerate * decay)
    release_samples = int(framerate * release)
    sustain_samples = num_samples - attack_samples - decay_samples - release_samples
    sustain_samples = max(sustain_samples, 0)

    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    else:
        envelope[0] = 1

    start = attack_samples
    end = attack_samples + decay_samples
    if decay_samples > 0:
        envelope[start:end] = np.linspace(1, sustain, decay_samples)
    else:
        envelope[start] = sustain

    start = end
    end = start + sustain_samples
    if sustain_samples > 0:
        envelope[start:end] = sustain

    if release_samples > 0:
        envelope[-release_samples:] = np.linspace(sustain, 0, release_samples)
    else:
        envelope[-1] = 0

    return envelope

def convolution_reverb(wave_data, impulse_response):
    if impulse_response is None or len(impulse_response) == 0:
        return wave_data
    # Ensure impulse_response is 1D for fftconvolve
    if impulse_response.ndim > 1:
        impulse_response = impulse_response.flatten()
    
    # Ensure wave_data is 1D for fftconvolve if it's stereo
    if wave_data.ndim > 1:
        # Apply to each channel separately or convert to mono for IR
        # For simplicity, let's apply to mono sum and then re-stereoize
        mono_wave_data = np.mean(wave_data, axis=1)
        convolved_mono = fftconvolve(mono_wave_data, impulse_response, mode='full')
        convolved_mono = convolved_mono[:len(mono_wave_data)]
        return np.column_stack((convolved_mono, convolved_mono)) # Convert back to stereo
    else:
        convolved_data = fftconvolve(wave_data, impulse_response, mode='full')
        convolved_data = convolved_data[:len(wave_data)]
        return convolved_data

def reverb_effect(wave_data, decay_factor=0.5, mix=0.5, framerate=44100):
    # Implement a simple Schroeder reverb using a comb filter
    delay_times = [0.0297, 0.0371, 0.0411]  # Delay times in seconds
    gains = [decay_factor, decay_factor * 0.8, decay_factor * 0.6]  # Gains for each delay

    reverb_signal = np.zeros_like(wave_data)
    for delay_time, gain in zip(delay_times, gains):
        delay_samples = int(delay_time * framerate)
        delayed_wave = np.zeros_like(wave_data)
        if delay_samples < wave_data.shape[0]:
            delayed_wave[delay_samples:] = wave_data[:-delay_samples]
            reverb_signal += delayed_wave * gain

    reverb_signal = np.clip(reverb_signal, -1, 1)
    return wave_data * (1 - mix) + reverb_signal * mix

def delay_effect(wave_data, delay_time=0.5, decay_factor=0.5, framerate=44100):
    delay_samples = int(delay_time * framerate)
    delayed_wave = np.zeros_like(wave_data)
    if delay_samples < wave_data.shape[0]:
        delayed_wave[delay_samples:] = wave_data[:-delay_samples]
    return wave_data + delayed_wave * decay_factor

def block_low_pass_filter(data, color_mod_data, base_cutoff_freq, color_filter_scale, framerate, block_size=1024):
    num_samples = data.shape[0]
    filtered_data = np.zeros_like(data)
    nyquist = 0.5 * framerate
    num_blocks = int(np.ceil(num_samples / block_size))

    for i in range(num_blocks):
        start = i * block_size
        end = min((i + 1) * block_size, num_samples)
        block = data[start:end]
        
        # Handle color_mod_data for the block
        if color_mod_data is not None and len(color_mod_data) > 0:
            avg_color_mod = np.mean(color_mod_data[start:end])
        else:
            avg_color_mod = 0.0 # No color modulation if data is missing

        cutoff_freq = base_cutoff_freq * (1 + avg_color_mod * color_filter_scale)
        cutoff_freq = min(max(cutoff_freq, 20.0), nyquist - 1)
        normal_cutoff = cutoff_freq / nyquist
        
        if normal_cutoff <= 0 or normal_cutoff >= 1: # Handle invalid cutoff
            filtered_data[start:end] = block
            continue

        b, a = butter(5, normal_cutoff, btype='low', analog=False)
        filtered_block = lfilter(b, a, block, axis=0)
        filtered_data[start:end] = filtered_block

    return filtered_data

def load_impulse_response(path, framerate):
    if not path or not os.path.exists(path):
        return None
    try:
        with wave.open(path, 'rb') as wav_file:
            n_channels = wav_file.getnchannels()
            sampwidth = wav_file.getsampwidth()
            n_frames = wav_file.getnframes()
            framerate_ir = wav_file.getframerate()
            frames = wav_file.readframes(n_frames)
            dtype = np.int16 if sampwidth == 2 else np.int8
            impulse_response = np.frombuffer(frames, dtype=dtype)
            impulse_response = impulse_response.astype(np.float32) / np.max(np.abs(impulse_response))
            
            if framerate_ir != framerate:
                from scipy.signal import resample
                num_samples = int(len(impulse_response) * framerate / framerate_ir)
                impulse_response = resample(impulse_response, num_samples)
            
            if n_channels > 1: # If stereo IR, convert to mono by averaging
                impulse_response = np.mean(impulse_response.reshape(-1, n_channels), axis=1)

            return impulse_response
    except Exception as e:
        print(f"Error loading impulse response: {e}")
        return None

def generate_parameter_envelope(num_samples, start_value, end_value, curve='linear'):
    if curve == 'linear':
        return np.linspace(start_value, end_value, num_samples)
    elif curve == 'exponential':
        # Add a small epsilon to avoid log(0)
        return np.logspace(np.log10(start_value + 1e-6), np.log10(end_value + 1e-6), num_samples)
    else:
        return np.linspace(start_value, end_value, num_samples)

def normalize_wave(wave_data):
    max_amplitude = np.max(np.abs(wave_data))
    return wave_data / max_amplitude if max_amplitude > 0 else wave_data

def limiter(wave_data, threshold=0.8):
    return np.clip(wave_data, -threshold, threshold)

def generate_spectrogram_data(data, framerate):
    f, t, Sxx = spectrogram(data, framerate)
    return 10 * np.log10(Sxx + 1e-6)

def stutter(audio_data, sample_rate, stutter_length_ms, repetitions):
    if repetitions <= 1:
        return audio_data
    stutter_length_samples = int(sample_rate * (stutter_length_ms / 1000.0))
    if stutter_length_samples == 0:
        return audio_data
    num_stutters = len(audio_data) // stutter_length_samples
    stuttered_audio = []
    for i in range(num_stutters):
        start = i * stutter_length_samples
        end = start + stutter_length_samples
        chunk = audio_data[start:end]
        for _ in range(repetitions):
            stuttered_audio.append(chunk)
    return np.concatenate(stuttered_audio)

def bitcrush(audio_data, bit_depth):
    if bit_depth >= 16:
        return audio_data
    max_val = 2**(bit_depth - 1)
    crushed_audio = np.round(audio_data * max_val) / max_val
    return crushed_audio

def granular_synthesis(audio_data, sample_rate, grain_size_ms, density):
    if density <= 0:
        return audio_data
    grain_size_samples = int(sample_rate * (grain_size_ms / 1000.0))
    if grain_size_samples == 0:
        return audio_data
    num_grains = int(len(audio_data) / grain_size_samples * density)
    output_audio = np.zeros(len(audio_data))
    for i in range(num_grains):
        start_sample = np.random.randint(0, len(audio_data) - grain_size_samples)
        grain = audio_data[start_sample : start_sample + grain_size_samples]
        output_start = i * grain_size_samples
        if output_start + grain_size_samples > len(output_audio):
            break
        output_end = output_start + grain_size_samples
        output_audio[output_start:output_end] = grain
    return output_audio

def reverse_audio(audio_data):
    return audio_data[::-1]

def add_echo(audio_data, sample_rate, delay_seconds, decay):
    if delay_seconds == 0:
        return audio_data
    delay_samples = int(delay_seconds * sample_rate)
    echo_samples = np.zeros_like(audio_data, dtype=float)
    if delay_samples < len(audio_data):
        echo_samples[delay_samples:] = audio_data[:-delay_samples] * decay
    return audio_data + echo_samples

