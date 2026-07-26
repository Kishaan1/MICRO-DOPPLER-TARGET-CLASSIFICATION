import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from backend.signal_processor import SignalProcessor

class MicroDopplerCNN(nn.Module):
    """
    2D Convolutional Neural Network for Micro-Doppler Spectrogram Image Classification.
    Classifies targets into: [0: Drone, 1: Bird, 2: Noise/Clutter]
    """
    def __init__(self, num_classes=3):
        super(MicroDopplerCNN, self).__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # Output: 16 x 64 x 64
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # Output: 32 x 32 x 32
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # Output: 64 x 16 x 16
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)) # Output: 128 x 4 x 4
        )

        self.fc_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        logits = self.fc_classifier(x)
        return logits


class MicroDopplerClassifier:
    """
    Inference and Training Wrapper for Micro-Doppler CNN Classifier.
    """
    CLASSES = ["Drone", "Bird", "Noise"]

    def __init__(self, model_path="models/microdoppler_cnn.pth"):
        # Enforce deterministic random seeds across PyTorch and NumPy
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MicroDopplerCNN(num_classes=3).to(self.device)
        self.model_path = model_path
        self.is_trained = False
        
        # Load weights or warm-up on synthetic dataset
        self._load_or_train_weights()

    def _load_or_train_weights(self):
        os.makedirs(os.path.dirname(self.model_path) if os.path.dirname(self.model_path) else ".", exist_ok=True)
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.eval()
                self.is_trained = True
                print(f"[Model] Successfully loaded pre-trained model weights from {self.model_path}")
                return
            except Exception as e:
                print(f"[Model] Warning loading state dict: {e}, re-training synthetic weights...")

        print("[Model] Training initial synthetic Micro-Doppler CNN weights...")
        self.train_synthetic(epochs=12)

    def train_synthetic(self, num_samples_per_class=30, epochs=12):
        """
        Trains the CNN model on runtime-generated synthetic FMCW radar signals.
        Ensures the network learns micro-Doppler time-frequency features.
        """
        sp = SignalProcessor()
        X_list = []
        y_list = []

        targets = ["drone", "bird", "noise"]
        for class_idx, target in enumerate(targets):
            for i in range(num_samples_per_class):
                snr = np.random.uniform(10, 30)
                dur = np.random.uniform(0.8, 1.2)
                t, sig = sp.generate_synthetic_signal(target_type=target, fs=8000, duration=dur, snr_db=snr)
                f, t_stft, Sxx_db = sp.compute_stft(sig, fs=8000)
                tensor_mat = sp.prepare_spectrogram_tensor(Sxx_db, target_size=(128, 128))
                
                X_list.append(tensor_mat)
                y_list.append(class_idx)

        X = torch.tensor(np.array(X_list), dtype=torch.float32).unsqueeze(1).to(self.device) # (N, 1, 128, 128)
        y = torch.tensor(np.array(y_list), dtype=torch.long).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        self.model.eval()
        self.is_trained = True
        
        try:
            torch.save(self.model.state_dict(), self.model_path)
            print(f"[Model] Saved trained weights to {self.model_path}")
        except Exception as e:
            print(f"[Model] Could not save state dict: {e}")

    def predict(self, spectrogram_tensor, extracted_features=None):
        """
        Runs CNN inference on a 2D spectrogram tensor (128, 128) and combines with physical feature ensemble.
        Returns:
            predicted_class (str): 'Drone', 'Bird', or 'Noise'
            confidence (float): 0.0 to 100.0 percentage
            probabilities (dict): { 'Drone': %, 'Bird': %, 'Noise': % }
        """
        self.model.eval()
        with torch.no_grad():
            tensor_input = torch.tensor(spectrogram_tensor, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            logits = self.model(tensor_input)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        prob_dict = {
            "Drone": float(probs[0]),
            "Bird": float(probs[1]),
            "Noise": float(probs[2])
        }

        # Secondary Ensemble / Physics Check for robustness
        if extracted_features is not None:
            harmonic_ratio = extracted_features.get("harmonic_ratio", 0.0)
            mod_bandwidth = extracted_features.get("mod_bandwidth_hz", 0.0)
            doppler_shift = extracted_features.get("doppler_shift_hz", 0.0)
            mod_rate = extracted_features.get("mod_rate_hz", 0.0)

            # Heuristic physics score boost:
            # Drone: High harmonic ratio (>0.12), higher bulk Doppler (>80Hz), fast propeller mod rate
            # Bird: Lower harmonic ratio (<0.10), high modulation bandwidth (>350Hz), low flap rate (3-12Hz)
            # Noise: Low bulk Doppler (<40Hz) and low harmonic ratio (<0.08)
            if harmonic_ratio > 0.12 and doppler_shift > 80:
                prob_dict["Drone"] += 0.65
            elif mod_bandwidth > 350 and mod_rate < 20 and harmonic_ratio < 0.12:
                prob_dict["Bird"] += 0.65
            elif doppler_shift < 40 and harmonic_ratio < 0.08:
                prob_dict["Noise"] += 0.65

            # Re-normalize probabilities
            total = sum(prob_dict.values())
            for k in prob_dict:
                prob_dict[k] /= total

        top_class = max(prob_dict, key=prob_dict.get)
        confidence = round(prob_dict[top_class] * 100.0, 2)

        # Format probabilities to percentage
        formatted_probs = {k: round(v * 100.0, 2) for k, v in prob_dict.items()}

        return top_class, confidence, formatted_probs
