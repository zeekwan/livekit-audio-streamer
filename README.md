# LiveKit Audio File Streaming

A simple web application that allows you to upload and stream audio files to LiveKit rooms in real-time. Perfect for sharing audio with other participants in a meeting, classroom, or virtual event.

## Quick Setup Guide

### Prerequisites

- Python 3.8+
- LiveKit account and server (cloud or self-hosted)

### Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/yourusername/livekit-audio-streaming.git
   cd livekit-audio-streaming
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**:
   ```bash
   pip install fastapi uvicorn python-dotenv python-multipart livekit-server-sdk livekit-rtc soundfile numpy pydantic
   ```

4. **Create a `.env` file** in the project root with your LiveKit credentials:
   ```
   LIVEKIT_WS_URL=wss://your-livekit-server-url
   LIVEKIT_API_KEY=your-api-key
   LIVEKIT_API_SECRET=your-api-secret
   PORT=3001
   OPENAI_API_KEY=api-key-here

   ```

### Running the Application

1. **Start the server**:
   ```bash
   python server.py
   ```

2. **Access the application** at http://localhost:3001

### Using the Application

1. **Connect to a Room**:
   - Enter a room name (or use the default)
   - Click "Connect to Room"
   - The status will show "Connected" when successful

2. **Upload an Audio File**:
   - Click "Choose File" and select an audio file
   - Click "Upload File"
   - The file will be processed and ready for playback

3. **Play Audio to the Room**:
   - Click "Play to Room"
   - The audio will be streamed to all participants in the room
   - You can stop playback with the "Stop Playback" button

4. **Local Preview**:
   - The uploaded audio file will be available for local preview
   - Use the audio player controls to listen to it locally

## Troubleshooting

- **Connection Issues**: Verify your LiveKit credentials in the `.env` file
- **Audio Quality Problems**: If you experience crackles or pops, the application uses advanced buffer management to minimize these issues
- **Large Files**: If you have issues with large audio files, try smaller files first

## Project Structure

```
livekit-audio-streaming/
├── server.py               # FastAPI server with audio streaming logic
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not committed)
├── public/                 # Static web files
│   ├── index.html          # Web interface
│   ├── css/
│   │   └── styles.css      # Styling
│   └── js/
│       └── app.js          # Frontend logic
```

## Technical Details

This implementation uses:
- **FastAPI**: For the web server and API endpoints
- **LiveKit SDK**: For real-time audio streaming
- **SoundFile**: For audio file processing
- **Advanced Buffer Management**: To ensure smooth audio playback

The audio streaming process:
1. Audio files are loaded and processed on the server
2. Audio data is chunked into small segments
3. These chunks are streamed to LiveKit with proper buffer management
4. All participants in the room hear the audio in real-time
