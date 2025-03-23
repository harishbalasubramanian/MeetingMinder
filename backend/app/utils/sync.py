# utils/sync.py
import cv2
import numpy as np
import torchaudio
def synchronize_modalities(file_path: str, window_size: float = 0.5) -> dict:
    """Handle both audio and video files with proper audio duration"""
    # Common output structure
    result = {
        "fps": None,
        "video_frames": [],
        "audio_windows": [],
        "duration": 0,
        "sample_rate": 16000
    }
    
    # Get audio duration and sample rate
    try:
        audio_info = torchaudio.info(file_path)
        result["duration"] = audio_info.num_frames / audio_info.sample_rate
        result["sample_rate"] = audio_info.sample_rate
    except Exception as e:
        pass
    
    # Video-specific processing
    cap = cv2.VideoCapture(file_path)
    if cap.isOpened():
        result["fps"] = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ret, frame = cap.read()
            if not ret: break
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            result["video_frames"].append((timestamp, frame))
        cap.release()
        
        # Use video duration if audio duration not available
        if result["duration"] == 0 and result["video_frames"]:
            result["duration"] = result["video_frames"][-1][0]

    # Generate audio windows based on actual duration
    start = 0.0
    while start < result["duration"]:
        end = min(start + window_size, result["duration"])
        result["audio_windows"].append((start, end))
        start = end
    
    return result