# visual_sentiment.py
import cv2
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification
import torch

class VisualSentimentAnalyzer:
    def __init__(self):
        self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.model = ViTForImageClassification.from_pretrained(
            "trpakov/vit-face-expression",
            num_labels=7  # angry, disgust, fear, happy, sad, surprise, neutral
        )
        self.id2label = self.model.config.id2label
        
        # Optical flow setup
        self.farneback_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

    def _get_optical_flow(self, prev_frame, current_frame):
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, **self.farneback_params
        )
        return np.linalg.norm(flow)

    def analyze_frame(self, frame, prev_frame=None):
        # Static expression analysis
        inputs = self.processor(images=frame, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probabilities).item()
        
        # Dynamic motion analysis
        motion_intensity = 0
        if prev_frame is not None:
            motion_intensity = self._get_optical_flow(prev_frame, frame)
        
        # Create feature tensor
        static_tensor = torch.tensor(probabilities.tolist()[0])
        if static_tensor.numel() != 7:
            static_tensor = torch.zeros(7)
        motion_tensor = torch.tensor([motion_intensity])
        
        return {
            "features": torch.cat([static_tensor, motion_tensor]),
            "emotion": self.id2label[pred_idx],
            "confidence": torch.max(probabilities).item()
        }