import os
os.environ["TOKENIZERS_PARALLELISM"] = 'false'
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
import uuid
import aiofiles
from transcriber import transcribe_audio
from summarizer import summarize_text
from action_items import extract_action_items
from video_processor import extract_audio_from_video, get_video_metadata
from utils.sync import synchronize_modalities
from audio_sentiment import AudioSentimentAnalyzer
from visual_sentiment import VisualSentimentAnalyzer
from fusion import MultimodalFusion
import torch

app = FastAPI()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Explicitly specify React's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]  # Add this line
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
class SentimentWindow(BaseModel):
    start_time: float
    end_time: float
    audio_emotion: Optional[str] = None
    visual_emotion: Optional[str] = None
    combined_sentiment: Optional[str] = None
    confidence: Optional[float] = None
class AnalysisResponse(BaseModel):
    transcript: str
    summary: str
    action_items: List[str]
    video_metadata: Optional[dict] = None
    sentiment_windows: List[SentimentWindow] = []

async def save_upload_file(upload_file: UploadFile) -> str:
    """Save uploaded file to disk and return path"""
    file_id = str(uuid.uuid4())
    file_path = f"{UPLOAD_DIR}/{file_id}_{upload_file.filename}"
    
    async with aiofiles.open(file_path, "wb") as f:
        content = await upload_file.read()
        await f.write(content)
    
    return file_path


@app.on_event("startup")
async def load_models():
    app.state.audio_analyzer = AudioSentimentAnalyzer()
    app.state.visual_analyzer = VisualSentimentAnalyzer()
    app.state.fusion_model = MultimodalFusion().eval()




@app.post("/analyze/")
async def analyze_file(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        logger.info('Starting File Save')
        file_path = await save_upload_file(file)
        temp_files = [file_path]
        metadata = {}
        # Process based on file type
        if file.content_type.startswith('video/'):
            # Extract audio and get metadata
            audio_path = extract_audio_from_video(file_path)
            temp_files.append(audio_path)
            metadata = get_video_metadata(file_path)
        elif file.content_type.startswith('audio/'):
            audio_path = file_path
            metadata = None
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        logger.info('Finished File Save')
        # Process audio pipeline
        logger.info('Starting Transcription')
        transcript = transcribe_audio(audio_path)
        logger.info('Finished Transcription')
        logger.info('Starting Summarization')
        summary = summarize_text(transcript)
        logger.info('Finished Summarization')
        logger.info('Starting Action Item Extraction')
        action_items = extract_action_items(transcript)
        logger.info('Finished Action Item Extraction')
        logger.info('Starting Synchronize Modalities')
        sync_data = synchronize_modalities(file_path)
        logger.info('Finished Synchronize Modalities')
        # Initialize sentiment results
        sentiment_results = []

        # Process each audio window
        logger.info('Starting Sentiment Analysis')
        for i, (window_start, window_end) in enumerate(sync_data["audio_windows"]):
            # Analyze audio segment
            try:
                logger.info(f'Window {i+1} Sentiment Analysis beginning out of {len(sync_data["audio_windows"])}')
                audio_results = app.state.audio_analyzer.analyze(
                    audio_path, 
                    windows=[(window_start, window_end)],
                    original_sample_rate=sync_data.get('sample_rate', 16000)
                )
                print('here1')
                audio_result = audio_results[0]  # Only one result per window
                audio_emotion = audio_result["emotion"]
                audio_confidence = audio_result["confidence"]
                print('here1.5')
                audio_features = torch.tensor(audio_result["features"])  # Convert to tensor
                # Find corresponding video frames
                print('here2')
                if audio_features.dim() == 0:  # Scalar case
                    audio_features = audio_features.unsqueeze(0).unsqueeze(0)  # [1, 1]
                elif audio_features.dim() == 1:  # 1D vector
                    audio_features = audio_features.unsqueeze(0)  # [1, 256]
                print('here3')
                video_frames_in_window = [
                    frame for ts, frame in sync_data["video_frames"]
                    if window_start <= ts <= window_end
                ]
                print('here4')
                # Analyze video frames
                visual_emotions = []
                visual_confidences = []
                visual_features = []
                prev_frame = None
                print('here5')
                if not file.content_type.startswith('audio/'):
                    for frame in video_frames_in_window:
                        visual_result = app.state.visual_analyzer.analyze_frame(frame, prev_frame)
                        visual_emotions.append(visual_result["emotion"])
                        visual_confidences.append(visual_result["confidence"])
                        visual_features.append(visual_result["features"])
                        prev_frame = frame

                    if visual_features:
                        visual_tensor = torch.stack(visual_features)
                        if visual_tensor.dim() == 1:
                            visual_tensor = visual_tensor.unsqueeze(0)
                        visual_tensor = visual_tensor.mean(dim=0)
                        
                        # Calculate dominant visual emotion
                        from collections import Counter
                        dominant_visual = Counter(visual_emotions).most_common(1)[0][0]
                        visual_confidence = sum(visual_confidences)/len(visual_confidences)
                    else:
                        visual_tensor = torch.zeros(8)  # Match feature size
                        dominant_visual = "neutral"
                        visual_confidence = 0.0  # Initialize visual_confidence

                # 6. Combine modalities
                audio_features = audio_features.float()
                if not file.content_type.startswith('audio/'):
                    visual_tensor = visual_tensor.float()
                    print('tick 1')
                    combined = app.state.fusion_model(
                        audio_features.unsqueeze(0),  # Add batch dimension
                        visual_tensor.unsqueeze(0)
                    )
                    print('tick 2')
                    # 7. Get final sentiment
                    sentiment_labels = {0: "negative", 1: "neutral", 2: "positive"}
                    combined_confidence = (audio_confidence + visual_confidence) / 2  # Now using defined visual_confidence
                
                sw = SentimentWindow(
                    start_time=window_start,
                    end_time=window_end,
                    audio_emotion=audio_emotion,
                    visual_emotion=dominant_visual if not file.content_type.startswith('audio/') else None,  # Now defined in both if/else cases
                    combined_sentiment=sentiment_labels[torch.argmax(combined).item()] if not file.content_type.startswith('audio/') else None,
                    confidence=combined_confidence if not file.content_type.startswith('audio/') else None
                )
            except Exception as e:
                print(e)
                sw = SentimentWindow(
                    start_time=window_start,
                    end_time=window_end,
                    audio_emotion=None,
                    visual_emotion=None,
                    combined_sentiment=None,
                    confidence=None
                )

            finally:
                sentiment_results.append(sw)
                logger.info(f'Window {i+1} Sentiment Analysis Ending')



        # Cleanup temporary files
        for f in temp_files:
            try:
                os.remove(f)
            except Exception as e:
                logger.warning(f"Error deleting temp file {f}: {str(e)}")
        
        return AnalysisResponse(
            transcript=transcript,
            summary=summary,
            action_items=action_items,
            video_metadata= metadata if metadata else None,
            sentiment_windows=sentiment_results
        )
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/process_audio/")
# async def process_audio(file: UploadFile = File(...)):
#     try:
#         # Save uploaded file
#         file_id = str(uuid.uuid4())
#         file_path = f"{UPLOAD_DIR}/{file_id}_{file.filename}"
        
#         async with aiofiles.open(file_path, "wb") as f:
#             content = await file.read()
#             await f.write(content)
        
#         # Process pipeline
#         transcript = transcribe_audio(file_path)
#         summary = summarize_text(transcript)
#         action_items = extract_action_items(transcript)
        
#         return SummaryResponse(
#             transcript=transcript,
#             summary=summary,
#             action_items=action_items
#         )
        
#     except Exception as e:
#         logger.error(f"Processing failed: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Meeting Minder Backend Running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)