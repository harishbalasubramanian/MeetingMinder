# audio_sentiment.py
import torchaudio
from transformers import AutoFeatureExtractor, HubertForSequenceClassification
import torch
from torchaudio.transforms import Resample
import numpy as np

class AudioSentimentAnalyzer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model with correct label count
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "superb/hubert-large-superb-er"
        )
        
        # Match the checkpoint's 4 classes
        self.model = HubertForSequenceClassification.from_pretrained(
            "superb/hubert-large-superb-er",
            num_labels=4,  # Must match pretrained model's classes
            ignore_mismatched_sizes=True  # Critical fix
        ).to(self.device)
        
        # Update LSTM dimensions to match actual output
        self.lstm = torch.nn.LSTM(
            input_size=self.model.config.hidden_size,  # Dynamic size
            hidden_size=256,
            num_layers=2,
            bidirectional=True
        ).to(self.device)
        self.target_sample_rate = 16000
        self.resamplers = {}
        self.id2label = {
            0: 'angry',
            1: 'happy',
            2: 'sad',
            3: 'neutral'
        }
    def get_resampler(self, original_rate: int):
        if original_rate not in self.resamplers:
            self.resamplers[original_rate] = Resample(orig_freq = original_rate, new_freq = self.target_sample_rate)
        return self.resamplers[original_rate]

    def analyze(self, audio_path: str, windows: list, original_sample_rate):
        """Main analysis method to be called from the endpoint"""
        # Load audio file
        waveform, _ = torchaudio.load(audio_path)
        if original_sample_rate != self.target_sample_rate:
            resampler = self.get_resampler(original_sample_rate)
            waveform = resampler(waveform)
        # Process each window
        results = []
        for start, end in windows:
            # Extract audio segment
            start_sample = int(start * self.target_sample_rate)
            end_sample = int(end * self.target_sample_rate)
            segment = waveform[:, start_sample:end_sample]
            
            # Extract features
            inputs = self.feature_extractor(
                segment.squeeze().numpy(),
                sampling_rate=self.target_sample_rate,
                return_tensors="pt"
            ).to(self.device)
            
            # Get model outputs
            # In the analyze method:
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                pred_idx = torch.argmax(probabilities).item()
                confidence = torch.max(probabilities).item()
                
                # Ensure features are 1D tensors
                if hasattr(self, 'lstm'):
                    hidden_states = outputs.hidden_states[-1]
                    lstm_out, _ = self.lstm(hidden_states)
                    features = torch.mean(lstm_out, dim=1).squeeze()
                else:
                    features = outputs.hidden_states[-1].mean(dim=1).squeeze()
                
                # Add dimension if scalar
                if isinstance(features, np.ndarray):
                    features = torch.from_numpy(features)
                
                if features.dim() == 0:  # Scalar tensor
                    features = features.unsqueeze(0)  # Convert to 1D tensor
            # Process outputs
            results.append({
                'features': features.numpy(),
                'emotion': self.id2label[pred_idx],
                'confidence': confidence
            })

        return results