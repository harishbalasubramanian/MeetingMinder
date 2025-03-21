from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from transcriber import transcribe_audio
from summarizer import summarize_text
from action_items import extract_action_items
import uuid
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)

UPLOAD_DIR = 'uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

class SummaryResponse(BaseModel):
    transcript: str
    summary: str
    action_items: List[str]




@app.get('/')
def read_root():
    return {'message': 'Meeting Minder Backend Running'}

@app.post('/process_audio/')
async def process_audio(file: UploadFile = File(...)):
    try:
        print('Started rn')
        file_id = str(uuid.uuid4())
        file_path = f"{UPLOAD_DIR}/{file_id}_{file.filename}"
        with open(file_path, 'wb') as f:
            f.write(await file.read())
        print('Start Transcription')
        transcript = transcribe_audio(file_path)
        print('Finish Transcription')
        print('Start Summarization')
        summary = summarize_text(transcript)
        print('Finish Summarization')
        print('Start Extraction')
        action_items = extract_action_items(transcript)
        print('End Transcription')
        return SummaryResponse(
            transcript = transcript,
            summary = summary,
            action_items = action_items
        )
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port = 8000)