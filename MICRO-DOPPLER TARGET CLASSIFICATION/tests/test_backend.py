import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.signal_processor import SignalProcessor
from backend.model import MicroDopplerClassifier
from backend.database import init_db, log_classification, get_history, get_stats, clear_history
from backend.app import app

class TestMicroDopplerBackend(unittest.TestCase):
    def setUp(self):
        self.processor = SignalProcessor(fs=8000, duration=1.0)
        self.app_client = app.test_client()
        init_db()

    def test_synthetic_drone_signal_generation(self):
        t, sig = self.processor.generate_synthetic_signal("drone", fs=8000, duration=1.0)
        self.assertEqual(len(t), 8000)
        self.assertEqual(len(sig), 8000)
        self.assertTrue(np.max(np.abs(sig)) <= 1.0)

    def test_stft_computation(self):
        t, sig = self.processor.generate_synthetic_signal("bird", fs=8000, duration=1.0)
        f, t_stft, Sxx_db = self.processor.compute_stft(sig, fs=8000)
        self.assertTrue(len(f) > 0)
        self.assertTrue(len(t_stft) > 0)
        self.assertEqual(Sxx_db.shape, (len(f), len(t_stft)))

    def test_feature_extraction(self):
        t, sig = self.processor.generate_synthetic_signal("drone", fs=8000, duration=1.0)
        f, t_stft, Sxx_db = self.processor.compute_stft(sig, fs=8000)
        features = self.processor.extract_features(f, t_stft, Sxx_db)
        self.assertIn("doppler_shift_hz", features)
        self.assertIn("mod_bandwidth_hz", features)
        self.assertIn("mod_rate_hz", features)
        self.assertIn("harmonic_ratio", features)

    def test_cnn_model_prediction(self):
        classifier = MicroDopplerClassifier()
        t, sig = self.processor.generate_synthetic_signal("drone", fs=8000, duration=1.0)
        f, t_stft, Sxx_db = self.processor.compute_stft(sig, fs=8000)
        tensor_mat = self.processor.prepare_spectrogram_tensor(Sxx_db, target_size=(128, 128))
        features = self.processor.extract_features(f, t_stft, Sxx_db)
        
        target_class, confidence, probs = classifier.predict(tensor_mat, extracted_features=features)
        self.assertIn(target_class, ["Drone", "Bird", "Noise"])
        self.assertTrue(0.0 <= confidence <= 100.0)
        self.assertIn("Drone", probs)

    def test_api_generate_sample_endpoint(self):
        response = self.app_client.post('/api/generate-sample', json={"target_type": "drone", "snr_db": 20})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn(data["target_class"], ["Drone", "Bird", "Noise"])

    def test_api_history_endpoint(self):
        response = self.app_client.get('/api/history')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn("stats", data)

if __name__ == '__main__':
    unittest.main()
