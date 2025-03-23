from moviepy import VideoFileClip
import cv2
import tempfile
import os
from pydub import AudioSegment

def extract_audio_from_video(video_path: str) -> str:
    """Extract audio from video and save as temporary WAV file"""
    # with VideoFileClip(video_path) as video:
    #     audio = video.audio
    #     temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    #     audio.write_audiofile(temp_audio.name)
    #     return temp_audio.name
    try:
        # Load video and extract audio
        video = AudioSegment.from_file(video_path)
        
        # Convert to required format
        audio = video.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            audio.export(temp_audio.name, format="wav")
            return temp_audio.name
            
    except Exception as e:
        raise RuntimeError(f"Audio extraction failed: {str(e)}")

def get_video_metadata(video_path: str) -> dict:
    """Get basic video metadata using OpenCV"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return {}
    
    metadata = {
        "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return metadata