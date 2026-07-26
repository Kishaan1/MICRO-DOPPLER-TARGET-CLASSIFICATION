from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import sys

# Ensure backend path is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.signal_processor import SignalProcessor
from backend.model import MicroDopplerClassifier
from backend.database import init_db, log_classification, get_history, get_stats, clear_history

app = Flask(__name__, template_folder="../templates", static_folder="../static")

# Initialize modules
signal_processor = SignalProcessor()
classifier = MicroDopplerClassifier()
init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload-signal', methods=['POST'])
@app.route('/api/analyze-target', methods=['POST'])
def upload_signal():
    """
    Accepts uploaded radar signal file (CSV, JSON, WAV), executes STFT,
    performs CNN inference, logs entry into DB, and returns results.
    """
    print("Received request files:", request.files, flush=True)

    if 'file' not in request.files:
        print("[Upload] Error: 'file' key not found in request.files", flush=True)
        return jsonify({
            "status": "error",
            "message": "CRITICAL: No file stream detected. Real-time telemetry processing requires valid input bytes."
        }), 400
    
    file = request.files['file']
    if file.filename == '':
        print("[Upload] Error: Selected file has empty filename", flush=True)
        return jsonify({
            "status": "error",
            "message": "CRITICAL: No file stream detected. Real-time telemetry processing requires valid input bytes."
        }), 400

    try:
        file_bytes = file.read()
        filename = file.filename
        print(f"[Upload] Reading file stream: '{filename}' ({len(file_bytes)} bytes)", flush=True)
        
        # 1. Parse physical time-series signal data
        t, raw_signal, fs = signal_processor.parse_file(file_bytes, filename)
        print(f"[Pipeline 1/5] Parsed signal array of length {len(raw_signal)} at fs={fs}Hz", flush=True)
        
        # 2. Process Short-Time Fourier Transform (STFT) Spectrogram
        f, t_stft, Sxx_db = signal_processor.compute_stft(raw_signal, fs=fs)
        print(f"[Pipeline 2/5] Computed STFT Spectrogram matrix Sxx_db shape: {Sxx_db.shape}", flush=True)
        
        # 3. Extract physical micro-Doppler features
        features = signal_processor.extract_features(f, t_stft, Sxx_db)
        print(f"[Pipeline 3/5] Extracted features: Doppler={features['doppler_shift_hz']}Hz, Bandwidth={features['mod_bandwidth_hz']}Hz", flush=True)
        
        # 4. Prepare (128, 128) 2D Spectrogram tensor for PyTorch CNN model
        tensor_mat = signal_processor.prepare_spectrogram_tensor(Sxx_db, target_size=(128, 128))
        
        # 5. PyTorch CNN Model Inference (model.eval() & with torch.no_grad())
        target_class, confidence, probabilities = classifier.predict(tensor_mat, extracted_features=features)
        print(f"[Pipeline 4/5] Model Inference Verdict: '{target_class}' with {confidence}% confidence", flush=True)
        
        # Render Spectrogram Image (base64 PNG)
        spectrogram_img = signal_processor.render_spectrogram_png_base64(f, t_stft, Sxx_db, title=f"STFT Spectrogram - {filename}")
        print(f"[Pipeline 5/5] Rendered base64 spectrogram image plot", flush=True)
        
        # Log result into SQLite Database
        log_id = log_classification(
            filename=filename,
            target_class=target_class,
            confidence=confidence,
            doppler_shift_hz=features["doppler_shift_hz"],
            mod_bandwidth_hz=features["mod_bandwidth_hz"],
            mod_rate_hz=features["mod_rate_hz"],
            harmonic_ratio=features["harmonic_ratio"]
        )

        # Downsample waveform for crisp high-resolution JS chart rendering
        downsample_factor = max(1, len(raw_signal) // 2000)
        time_downsampled = t[::downsample_factor]
        signal_downsampled = raw_signal[::downsample_factor]

        return jsonify({
            "success": True,
            "filename": filename,
            "verdict": target_class.upper(),
            "target_class": target_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "features": features,
            "metrics": {
                "doppler_shift_hz": features["doppler_shift_hz"],
                "mod_bandwidth_hz": features["mod_bandwidth_hz"],
                "harmonic_ratio": features["harmonic_ratio"]
            },
            "spectrogram_img": spectrogram_img,
            "spectrogram_base64": spectrogram_img,
            "waveform": {
                "time": time_downsampled,
                "amplitude": signal_downsampled
            },
            "spectrogram_matrix": {
                "freq": f.tolist(),
                "time": t_stft.tolist(),
                "values": Sxx_db.tolist()
            },
            "log_id": log_id,
            "stats": get_stats()
        })

    except Exception as e:
        return jsonify({"error": f"Failed to process radar signal: {str(e)}"}), 500


@app.route('/api/classify', methods=['POST'])
def classify_signal():
    """
    Classifies raw array radar signal sent via JSON payload.
    """
    data = request.get_json() or {}
    raw_signal = data.get("signal", [])
    filename = data.get("filename", "raw_signal.json")
    fs = data.get("fs", 8000)

    if not raw_signal or len(raw_signal) < 10:
        return jsonify({"error": "Invalid or empty signal array"}), 400

    try:
        t = np.linspace(0, len(raw_signal)/fs, len(raw_signal)).tolist()
        f, t_stft, Sxx_db = signal_processor.compute_stft(raw_signal, fs=fs)
        features = signal_processor.extract_features(f, t_stft, Sxx_db)
        tensor_mat = signal_processor.prepare_spectrogram_tensor(Sxx_db, target_size=(128, 128))
        
        target_class, confidence, probabilities = classifier.predict(tensor_mat, extracted_features=features)
        spectrogram_img = signal_processor.render_spectrogram_png_base64(f, t_stft, Sxx_db, title=f"STFT Spectrogram - {filename}")
        
        log_id = log_classification(
            filename=filename,
            target_class=target_class,
            confidence=confidence,
            doppler_shift_hz=features["doppler_shift_hz"],
            mod_bandwidth_hz=features["mod_bandwidth_hz"],
            mod_rate_hz=features["mod_rate_hz"],
            harmonic_ratio=features["harmonic_ratio"]
        )

        downsample_factor = max(1, len(raw_signal) // 2000)
        return jsonify({
            "success": True,
            "filename": filename,
            "target_class": target_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "features": features,
            "spectrogram_img": spectrogram_img,
            "waveform": {
                "time": t[::downsample_factor],
                "amplitude": raw_signal[::downsample_factor]
            },
            "log_id": log_id,
            "stats": get_stats()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/generate-sample', methods=['POST'])
def generate_sample():
    """
    Generates synthetic benchmark FMCW radar signal samples (Drone, Bird, Noise) on demand.
    """
    data = request.get_json() or {}
    target_type = data.get("target_type", "drone").lower()
    snr_db = float(data.get("snr_db", 20.0))
    fs = 8000
    duration = 1.0

    target_labels = {
        "drone": "Synthetic_Quadcopter_Drone.wav",
        "bird": "Synthetic_Flapping_Bird.wav",
        "noise": "Synthetic_Ambient_Clutter.wav"
    }
    filename = target_labels.get(target_type, "Synthetic_Radar_Sample.wav")

    try:
        t, raw_signal = signal_processor.generate_synthetic_signal(target_type=target_type, fs=fs, duration=duration, snr_db=snr_db)
        f, t_stft, Sxx_db = signal_processor.compute_stft(raw_signal, fs=fs)
        features = signal_processor.extract_features(f, t_stft, Sxx_db)
        tensor_mat = signal_processor.prepare_spectrogram_tensor(Sxx_db, target_size=(128, 128))
        
        target_class, confidence, probabilities = classifier.predict(tensor_mat, extracted_features=features)
        spectrogram_img = signal_processor.render_spectrogram_png_base64(f, t_stft, Sxx_db, title=f"STFT Micro-Doppler - {target_type.upper()}")
        
        log_id = log_classification(
            filename=filename,
            target_class=target_class,
            confidence=confidence,
            doppler_shift_hz=features["doppler_shift_hz"],
            mod_bandwidth_hz=features["mod_bandwidth_hz"],
            mod_rate_hz=features["mod_rate_hz"],
            harmonic_ratio=features["harmonic_ratio"]
        )

        downsample_factor = max(1, len(raw_signal) // 2000)
        return jsonify({
            "success": True,
            "filename": filename,
            "target_type": target_type,
            "target_class": target_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "features": features,
            "spectrogram_img": spectrogram_img,
            "waveform": {
                "time": t[::downsample_factor],
                "amplitude": raw_signal[::downsample_factor]
            },
            "spectrogram_matrix": {
                "freq": f.tolist(),
                "time": t_stft.tolist(),
                "values": Sxx_db.tolist()
            },
            "log_id": log_id,
            "stats": get_stats()
        })

    except Exception as e:
        return jsonify({"error": f"Failed to generate synthetic target sample: {str(e)}"}), 500


@app.route('/api/history', methods=['GET'])
def history():
    """
    Returns historical classification records and overall radar stats.
    """
    try:
        logs = get_history(limit=50)
        stats = get_stats()
        return jsonify({
            "success": True,
            "logs": logs,
            "stats": stats
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/history', methods=['DELETE'])
def clear_logs():
    """
    Clears all classification database records.
    """
    try:
        clear_history()
        return jsonify({"success": True, "stats": get_stats(), "message": "History cleared successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("[Server] Starting Micro-Doppler Radar Classification Server on http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
