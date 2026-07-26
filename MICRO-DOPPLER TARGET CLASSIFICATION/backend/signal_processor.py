import numpy as np
import scipy.signal as signal
import io
import base64
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class SignalProcessor:
    """
    Micro-Doppler Radar Signal Processor.
    Handles FMCW radar simulation, STFT computation, spectrogram extraction,
    and physical feature estimation (Doppler shift, modulation rate, bandwidth).
    """

    def __init__(self, fs=8000, duration=1.0):
        self.fs = fs             # Sampling frequency in Hz
        self.duration = duration # Signal duration in seconds

    def generate_synthetic_signal(self, target_type="drone", fs=8000, duration=1.0, snr_db=20, seed=None):
        """
        Generates realistic FMCW radar micro-Doppler time-series signals for:
        - 'drone': Multi-rotor UAV with fast rotating propeller blades
        - 'bird': Flapping wing biological target with sinusoidal frequency modulation
        - 'noise': Ambient white Gaussian noise and ground clutter
        """
        if seed is not None:
            np.random.seed(seed)
        self.fs = fs
        self.duration = duration
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)

        if target_type.lower() == "drone":
            # Dynamic Bulk body Doppler shift (flight velocity ~12-28 m/s, e.g. 140-280 Hz offset)
            f_bulk = float(np.random.uniform(140.0, 270.0))
            bulk_signal = 0.8 * np.exp(1j * 2 * np.pi * f_bulk * t)

            # Micro-Doppler from propeller blades:
            # Propeller rotation frequency (35 to 75 Hz RPM)
            f_rot = float(np.random.uniform(35.0, 75.0))
            num_blades = int(np.random.choice([2, 3, 4, 6]))
            micro_doppler = np.zeros_like(t, dtype=np.complex128)

            for b in range(num_blades):
                phase_offset = b * (2 * np.pi / num_blades)
                mod_index = float(np.random.uniform(6.0, 11.0))
                blade_phase = mod_index * np.sin(2 * np.pi * f_rot * t + phase_offset)
                micro_doppler += 0.5 * np.exp(1j * (2 * np.pi * f_bulk * t + blade_phase))
            
            # Harmonic flashes (blade flashes when perpendicular to radar beam)
            flash_freq = f_rot * num_blades
            flash_envelope = (0.5 + 0.5 * np.cos(2 * np.pi * flash_freq * t)) ** 4
            
            raw_signal = (bulk_signal + micro_doppler) * flash_envelope

        elif target_type.lower() == "bird":
            # Dynamic Bulk body Doppler shift (slow flight, e.g. 35-85 Hz offset)
            f_bulk = float(np.random.uniform(35.0, 85.0))
            
            # Wing flap frequency (3 to 9 Hz periodic sweeping)
            f_flap = float(np.random.uniform(3.5, 9.0))
            # Broad frequency excursion due to large wing surface motion
            mod_index = float(np.random.uniform(18.0, 32.0))
            
            wing_phase = mod_index * np.sin(2 * np.pi * f_flap * t)
            
            # Asymmetrical wing beat (downstroke vs upstroke)
            body_signal = 0.9 * np.exp(1j * 2 * np.pi * f_bulk * t)
            wing_signal = 0.75 * np.exp(1j * (2 * np.pi * f_bulk * t + wing_phase))
            
            raw_signal = body_signal + wing_signal

        else:  # 'noise' / background clutter
            f_clutter = float(np.random.uniform(5.0, 35.0)) # Low frequency ground clutter
            clutter = 0.35 * np.exp(1j * 2 * np.pi * f_clutter * t)
            raw_signal = clutter

        # Convert complex I/Q signal to real beat signal
        real_signal = np.real(raw_signal)

        # Add additive white Gaussian noise (AWGN) based on specified SNR
        sig_power = np.mean(real_signal ** 2)
        if sig_power > 0:
            noise_power = sig_power / (10 ** (snr_db / 10.0))
            noise = np.random.normal(0, np.sqrt(noise_power), len(real_signal))
            real_signal = real_signal + noise

        # Normalize signal to [-1, 1]
        max_val = np.max(np.abs(real_signal))
        if max_val > 0:
            real_signal = real_signal / max_val

        return t.tolist(), real_signal.tolist()

    def compute_stft(self, signal_data, fs=8000, nperseg=256, noverlap=192):
        """
        Computes Short-Time Fourier Transform (STFT) for joint time-frequency analysis.
        Returns:
            f: Frequency array (Hz)
            t: Time array (s)
            Sxx_db: Power spectral density in dB
        """
        signal_array = np.array(signal_data, dtype=float)
        # Remove DC offset for accurate Doppler spectrum analysis
        if len(signal_array) > 0:
            signal_array = signal_array - np.mean(signal_array)
        
        # Ensure signal has minimum length by zero-padding if too short
        min_len = 512
        if len(signal_array) < min_len:
            pad_width = min_len - len(signal_array)
            signal_array = np.pad(signal_array, (0, pad_width), mode='constant')

        # Ensure nperseg does not exceed signal length and noverlap < nperseg
        sig_len = len(signal_array)
        if nperseg > sig_len:
            nperseg = max(16, sig_len)
        if noverlap >= nperseg:
            noverlap = max(0, nperseg - 32)

        # Scipy STFT computation
        f, t, Zxx = signal.stft(signal_array, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
        
        # Magnitude Spectrogram
        Sxx = np.abs(Zxx)
        
        # Convert to dB scale with offset floor
        Sxx_db = 20 * np.log10(Sxx + 1e-6)
        
        # Dynamic threshold floor (normalize top dynamic range to e.g. 60 dB)
        max_db = np.max(Sxx_db)
        Sxx_db = np.maximum(Sxx_db, max_db - 60.0)

        return f, t, Sxx_db

    def extract_features(self, f, t, Sxx_db):
        """
        Extracts physical micro-Doppler features from the STFT matrix:
        - Estimated Bulk Doppler Shift (Hz)
        - Peak Modulation Rate (Hz)
        - Micro-Doppler Modulation Bandwidth (Hz)
        - Harmonic Ratio
        - Spectral Centroid (Hz)
        """
        # Linear power matrix
        Sxx_lin = 10 ** (Sxx_db / 20.0)
        
        # Average spectrum across time (ignore index 0 for DC)
        mean_spectrum = np.mean(Sxx_lin, axis=1)
        if len(mean_spectrum) > 1:
            peak_idx = 1 + np.argmax(mean_spectrum[1:])
        else:
            peak_idx = np.argmax(mean_spectrum)
        doppler_shift = float(f[peak_idx])

        # Spectral Centroid
        total_power = np.sum(mean_spectrum) + 1e-9
        spectral_centroid = float(np.sum(f * mean_spectrum) / total_power)

        # Micro-Doppler modulation bandwidth (range of frequencies carrying 90% energy)
        cum_energy = np.cumsum(mean_spectrum) / total_power
        idx_low = np.searchsorted(cum_energy, 0.05)
        idx_high = np.searchsorted(cum_energy, 0.95)
        mod_bandwidth = float(max(10.0, f[idx_high] - f[idx_low]))

        # Modulation frequency rate estimation via autocorrelation of energy temporal envelope
        temporal_envelope = np.sum(Sxx_lin, axis=0)
        temporal_envelope = temporal_envelope - np.mean(temporal_envelope)
        
        if np.std(temporal_envelope) > 1e-5:
            autocorr = np.correlate(temporal_envelope, temporal_envelope, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find first dominant peak after lag 0
            dt = t[1] - t[0] if len(t) > 1 else 0.01
            min_lag = int(0.01 / dt) # > 100 Hz
            max_lag = int(0.5 / dt)   # < 2 Hz
            
            if len(autocorr) > max_lag and min_lag < max_lag:
                valid_autocorr = autocorr[min_lag:max_lag]
                if len(valid_autocorr) > 0 and np.max(valid_autocorr) > 0:
                    peak_lag = min_lag + np.argmax(valid_autocorr)
                    mod_rate = float(1.0 / (peak_lag * dt))
                else:
                    mod_rate = 5.0
            else:
                mod_rate = 5.0
        else:
            mod_rate = 0.0

        # Harmonic Ratio (ratio of high-frequency micro-Doppler sideband energy to low-frequency energy)
        mid_freq_idx = len(f) // 2
        high_freq_energy = np.sum(mean_spectrum[mid_freq_idx:])
        low_freq_energy = np.sum(mean_spectrum[:mid_freq_idx]) + 1e-9
        harmonic_ratio = float(high_freq_energy / low_freq_energy)

        return {
            "doppler_shift_hz": round(doppler_shift, 2),
            "doppler_velocity_mps": round(doppler_shift * 0.03, 2), # Assuming X-band radar wavelength lambda ~ 3cm
            "spectral_centroid_hz": round(spectral_centroid, 2),
            "mod_bandwidth_hz": round(mod_bandwidth, 2),
            "mod_rate_hz": round(mod_rate, 2),
            "harmonic_ratio": round(harmonic_ratio, 4)
        }

    def prepare_spectrogram_tensor(self, Sxx_db, target_size=(128, 128)):
        """
        Normalizes STFT matrix and resizes to (128, 128) tensor matrix ready for CNN inference.
        """
        # Min-Max Normalization to [0, 1]
        s_min, s_max = np.min(Sxx_db), np.max(Sxx_db)
        if s_max > s_min:
            norm_matrix = (Sxx_db - s_min) / (s_max - s_min)
        else:
            norm_matrix = np.zeros_like(Sxx_db)

        # Interpolate / Resize to target_size (128, 128)
        from scipy.ndimage import zoom
        zoom_f = target_size[0] / norm_matrix.shape[0]
        zoom_t = target_size[1] / norm_matrix.shape[1]
        resized_matrix = zoom(norm_matrix, (zoom_f, zoom_t), order=1)

        # Crop or pad if shape slightly off by rounding
        resized_matrix = resized_matrix[:target_size[0], :target_size[1]]
        if resized_matrix.shape != target_size:
            padded = np.zeros(target_size, dtype=np.float32)
            padded[:resized_matrix.shape[0], :resized_matrix.shape[1]] = resized_matrix
            resized_matrix = padded

        return resized_matrix.astype(np.float32)

    def render_spectrogram_png_base64(self, f, t, Sxx_db, title="Joint Time-Frequency Micro-Doppler Spectrogram"):
        """
        Generates a visually striking Matplotlib spectrogram image encoded in base64 PNG.
        """
        fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=110)
        fig.patch.set_facecolor('#050505')
        ax.set_facecolor('#050505')

        # Plot Spectrogram with colormap 'turbo'
        mesh = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='turbo')
        
        ax.set_title(title, color='#00ff33', fontsize=10, fontweight='bold', pad=10, family='monospace')
        ax.set_xlabel('Time (s)', color='#00ff33', fontsize=8.5, family='monospace')
        ax.set_ylabel('Doppler Frequency (Hz)', color='#00ff33', fontsize=8.5, family='monospace')
        
        ax.tick_params(colors='#666666', labelsize=7.5)
        for spine in ax.spines.values():
            spine.set_color('#222222')

        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label('Power (dB)', color='#00ff33', fontsize=7.5, family='monospace')
        cbar.ax.tick_params(colors='#666666', labelsize=7)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')

        # Explicitly clear figure and close handles to prevent state leakage across requests
        fig.clf()
        plt.close(fig)
        plt.close('all')

        return f"data:image/png;base64,{encoded}"

    def parse_file(self, file_bytes, filename):
        """
        Parses uploaded file (CSV, JSON, WAV) into time array and signal array.
        """
        filename_lower = filename.lower()
        if filename_lower.endswith('.json'):
            data = json.loads(file_bytes.decode('utf-8'))
            if isinstance(data, dict):
                signal_data = data.get('signal', data.get('data', []))
                fs = data.get('fs', 8000)
            else:
                signal_data = data
                fs = 8000
            signal_arr = np.array(signal_data, dtype=float)
            t = np.linspace(0, len(signal_arr)/fs, len(signal_arr)).tolist()
            return t, signal_arr.tolist(), fs

        elif filename_lower.endswith('.csv') or filename_lower.endswith('.txt'):
            import pandas as pd
            buf = io.BytesIO(file_bytes)
            try:
                df = pd.read_csv(buf)
                print("CSV Columns found:", list(df.columns), flush=True)
                if 'Amplitude_dB' in df.columns:
                    signal_arr = df['Amplitude_dB'].to_numpy(dtype=float)
                elif 'amplitude' in df.columns:
                    signal_arr = df['amplitude'].to_numpy(dtype=float)
                elif 'signal' in df.columns:
                    signal_arr = df['signal'].to_numpy(dtype=float)
                else:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        signal_arr = df[numeric_cols[0]].to_numpy(dtype=float)
                    else:
                        signal_arr = df.iloc[:, 0].to_numpy(dtype=float)
            except Exception as pe:
                print(f"[CSV Parsing Warning]: {pe}, trying raw text fallback", flush=True)
                text = file_bytes.decode('utf-8', errors='ignore')
                lines = [line.strip() for line in text.split('\n') if line.strip() and not line.startswith('#')]
                vals = []
                for line in lines:
                    parts = line.replace(',', ' ').split()
                    if len(parts) >= 2:
                        try:
                            vals.append(float(parts[1]))
                        except ValueError:
                            pass
                    elif len(parts) == 1:
                        try:
                            vals.append(float(parts[0]))
                        except ValueError:
                            pass
                signal_arr = np.array(vals if vals else [0.0]*1000, dtype=float)

            print("Signal array shape:", signal_arr.shape, flush=True)
            fs = 8000
            t = np.linspace(0, len(signal_arr)/fs, len(signal_arr)).tolist()
            return t, signal_arr.tolist(), fs

        elif filename_lower.endswith('.wav'):
            import scipy.io.wavfile as wavfile
            buf = io.BytesIO(file_bytes)
            fs, data = wavfile.read(buf)
            if data.ndim > 1:
                data = data[:, 0] # mono
            data = data.astype(float)
            if np.max(np.abs(data)) > 0:
                data = data / np.max(np.abs(data))
            t = np.linspace(0, len(data)/fs, len(data)).tolist()
            return t, data.tolist(), fs

        else:
            raise ValueError(f"Unsupported file format: {filename}. Please upload WAV, CSV, or JSON.")
