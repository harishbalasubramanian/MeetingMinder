import whisper

def transcribe_audio(audio_path: str) -> str:
    print('Got here')
    model = whisper.load_model('base')

    result = model.transcribe(audio_path)

    return result['text']

if __name__ == '__main__':
    test_audio = 'audio_files/sample_audio_1.mp3'

    transcript = transcribe_audio(test_audio)
    print(f'Transcription: {transcript}')