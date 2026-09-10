# Voice Service

Speech-to-Text (STT using Whisper) and Text-to-Speech (TTS using F5-TTS) REST microservice built with FastAPI.

## Endpoints

### 1. Transcribe Audio (Speech-to-Text)
- **POST** `/transcribe`
- **Content-Type**: `multipart/form-data`
- **Body**: `file` (audio file e.g. wav, mp3, m4a)
- **Response**:
```json
{
  "text": "transcribed speech"
}
```

### 2. Synthesize Speech (Text-to-Speech)
- **POST** `/synthesize`
- **Content-Type**: `application/json`
- **Body**:
```json
{
  "text": "Hello world, this is synthesized speech.",
  "voice_id": "family_member_clip"
}
```
- **Response**: Audio stream (`audio/wav`)

---

## Installation & Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run the FastAPI server:
```bash
uvicorn main:app --reload --port 8000
```

3. Access interactive Swagger API documentation at `http://localhost:8000/docs`.
