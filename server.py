import os
import asyncio
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import logging
from typing import Dict, Optional
import time
from openai import OpenAI, AsyncOpenAI
import io
import tempfile
from openai.helpers import LocalAudioPlayer

# Import LiveKit Python SDK components
from livekit import rtc, api
import soundfile as sf
import numpy as np
from livekit.api import AccessToken

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# FastAPI app setup
app = FastAPI(title="LiveKit Audio Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LiveKit configuration & open AI config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LIVEKIT_URL = os.getenv("LIVEKIT_WS_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
DEFAULT_ROOM_NAME = "audio-room"

# Debug logging for environment variables
logger.info(f"LiveKit URL: {LIVEKIT_URL}")
logger.info(f"LiveKit API Key: {LIVEKIT_API_KEY}")
logger.info(f"LiveKit API Secret: {LIVEKIT_API_SECRET[:4]}...")  # Only log first 4 chars of secret for security

# Storage for active rooms and audio files
active_rooms = {}
audio_files = {}

# Model for token request
class TokenRequest(BaseModel):
    room_name: str = DEFAULT_ROOM_NAME
    participant_name: str = None

# Model for audio playback request
class PlaybackRequest(BaseModel):
    room_name: str
    file_id: str

class TTSRequest(BaseModel):
    room_name: str
    text: str
    voice: str = "alloy"
    model: str = "gpt-4o-mini-tts"
    instructions: str = "Speak in a somber and depressed tone"

class AudioBufferManager:
    def __init__(self, room_name, audio_source, sample_rate, num_channels=1):
        self.room_name = room_name
        self.audio_source = audio_source
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        
        # Buffer management settings
        self.target_buffer = 0.2  # 100ms target buffer
        self.min_buffer = 0.1    # 50ms minimum buffer
        self.max_buffer = 0.4     # 300ms maximum buffer
        
        # Statistics
        self.frames_sent = 0
        self.buffer_stats = []
        
    async def send_chunk(self, chunk, chunk_duration):
        """Send an audio chunk with proper buffer management"""
        # Check current buffer state
        current_buffer = self.audio_source.queued_duration
        self.buffer_stats.append(current_buffer)
        
        # Keep only the last 50 buffer measurements
        if len(self.buffer_stats) > 50:
            self.buffer_stats.pop(0)
            
        # Calculate average buffer level
        avg_buffer = sum(self.buffer_stats) / len(self.buffer_stats)
        
        # If buffer is too full, wait for it to drain
        if current_buffer > self.max_buffer:
            wait_time = current_buffer - self.target_buffer
            logger.debug(f"Buffer full ({current_buffer:.3f}s), waiting {wait_time:.3f}s")
            await asyncio.sleep(wait_time)
            
        # Create an AudioFrame
        int16_data = (chunk * 32767).astype(np.int16).tobytes()
        frame = rtc.AudioFrame(
            data=int16_data,
            sample_rate=self.sample_rate,
            num_channels=self.num_channels,
            samples_per_channel=len(chunk)
        )
        
        # Send the frame
        await self.audio_source.capture_frame(frame)
        self.frames_sent += 1
        
        # Adaptively adjust sleep time based on buffer state
        if avg_buffer < self.min_buffer:
            # Buffer is too low, send next frame quickly
            sleep_time = chunk_duration * 0.7
        elif avg_buffer > self.target_buffer:
            # Buffer is high, slow down slightly
            sleep_time = chunk_duration * 1.1
        else:
            # Buffer is in good range, use normal timing
            sleep_time = chunk_duration
            
        # Log buffer stats occasionally
        if self.frames_sent % 100 == 0:
            logger.info(f"Room {self.room_name}: Sent {self.frames_sent} frames, buffer: {current_buffer:.3f}s, avg: {avg_buffer:.3f}s")
            
        return sleep_time

# Function to create a LiveKit token
def create_token(room_name: str, participant_name: str = None):
    """Generate a LiveKit access token for a participant"""
    if not participant_name:
        participant_name = f"audio-server-{uuid.uuid4()}"
    
    token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET) \
        .with_identity(participant_name) \
        .with_name(participant_name) \
        .with_grants(api.VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True
        ))
    
    return token.to_jwt()

# Create or get room service client
async def get_room_service():
    """Get or create a LiveKit room service client"""
    return api.LiveKitAPI(
        url=LIVEKIT_URL,
        api_key=LIVEKIT_API_KEY,
        api_secret=LIVEKIT_API_SECRET
    )

# Ensure room exists
async def ensure_room_exists(room_name: str):
    """Make sure a room exists in LiveKit"""
    # In LiveKit, rooms are created automatically when someone joins
    # We don't need to pre-create them
    pass

# Connect to a LiveKit room
async def connect_to_room(room_name: str, participant_name: str = None):
    """Connect to a LiveKit room and return the room object"""
    # Ensure the room exists
    await ensure_room_exists(room_name)
    
    # Generate token
    token = create_token(room_name, participant_name)
    
    # Create and connect to room
    room = rtc.Room()
    await room.connect(LIVEKIT_URL, token)
    logger.info(f"Connected to room: {room_name} as {room.local_participant.identity}")
    
    # Store the room
    active_rooms[room_name] = room
    
    return room

# Load audio file
async def load_audio_file(file_path: str, file_id: str):
    """Load an audio file and store its data"""
    try:
        # Load audio file using soundfile
        data, sample_rate = sf.read(file_path)
        
        # Convert to mono if stereo
        if len(data.shape) > 1 and data.shape[1] > 1:
            data = np.mean(data, axis=1)
        
        # Store the audio data
        audio_files[file_id] = {
            "data": data,
            "sample_rate": sample_rate,
            "file_path": file_path
        }
        
        logger.info(f"Loaded audio file: {file_path} (ID: {file_id})")
        return True
    except Exception as e:
        logger.error(f"Error loading audio file: {e}")
        return False
async def play_audio_to_room(room_name: str, file_id: str):
    """Stream an audio file to a LiveKit room using native features with advanced buffer management"""
    if room_name not in active_rooms:
        logger.error(f"Room not found: {room_name}")
        return False
    
    if file_id not in audio_files:
        logger.error(f"Audio file not found: {file_id}")
        return False
    
    try:
        room = active_rooms[room_name]
        audio_data = audio_files[file_id]
        
        # Target sample rate (standard for WebRTC)
        target_sample_rate = 48000
        
        # Create an audio source with the target sample rate
        audio_source = rtc.AudioSource(
            sample_rate=target_sample_rate,
            num_channels=1,
            queue_size_ms=1000  # 1 second buffer
        )
        
        # Create and publish a local audio track
        track = rtc.LocalAudioTrack.create_audio_track("audio-file", audio_source)
        
        try:
            # Publish the track and wait for it to be established
            publication = await room.local_participant.publish_track(track)
            logger.info(f"Track published to room {room_name}")
            
            # Get the audio data and original sample rate
            data = audio_data["data"]
            source_sample_rate = audio_data["sample_rate"]
            
            # Create a resampler to convert from source sample rate to target sample rate
            resampler = rtc.AudioResampler(
                input_rate=source_sample_rate,
                output_rate=target_sample_rate,
                num_channels=1,
                quality=rtc.AudioResamplerQuality.HIGH
            )
            
            # Calculate the chunk size for 20ms segments at the source sample rate
            chunk_duration_ms = 20
            chunk_size = int(source_sample_rate * chunk_duration_ms / 1000)
            chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
            
            logger.info(f"Starting audio playback from file ID {file_id} in room {room_name}")
            logger.info(f"File details: {len(chunks)} chunks at {source_sample_rate}Hz → {target_sample_rate}Hz, {chunk_duration_ms}ms per chunk")
            
            # Stream chunks with managed buffering
            for i, chunk in enumerate(chunks):
                # Pad the last chunk if needed
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                
                # Convert numpy array to AudioFrame
                int16_data = (chunk * 32767).astype(np.int16).tobytes()
                input_frame = rtc.AudioFrame(
                    data=int16_data,
                    sample_rate=source_sample_rate,
                    num_channels=1,
                    samples_per_channel=len(chunk)
                )
                
                # Resample to target sample rate
                resampled_frames = resampler.push(input_frame)
                
                # Process each resampled frame
                for frame in resampled_frames:
                    # Send the frame directly to the audio source without numpy conversion
                    await audio_source.capture_frame(frame)
                    
                    # To maintain timing, sleep a small amount
                    # This is based on frame duration (samples / rate)
                    frame_duration = frame.samples_per_channel / target_sample_rate
                    await asyncio.sleep(frame_duration * 0.5)  # Sleep for half the frame duration
                
                # Log progress occasionally
                if i % 500 == 0 and i > 0:
                    progress_percent = (i / len(chunks)) * 100
                    logger.info(f"Playback progress: {i}/{len(chunks)} chunks ({progress_percent:.1f}%)")
            
            # Flush the resampler for any remaining data
            flush_frames = resampler.flush()
            for frame in flush_frames:
                await audio_source.capture_frame(frame)
            
            logger.info(f"All chunks sent for file ID {file_id}, waiting for buffer to empty")
            
            # Wait for the audio source to finish playing everything
            while audio_source.queued_duration > 0:
                logger.debug(f"Waiting for final audio to play, {audio_source.queued_duration:.3f}s remaining")
                await asyncio.sleep(0.1)
                
            logger.info(f"Finished audio playback from file ID {file_id}")
            
        finally:
            # Always unpublish the track and clean up resources
            try:
                if track:
                    await room.local_participant.unpublish_track(track)
                    track.stop()
                    logger.info(f"Track unpublished from room {room_name}")
            except Exception as e:
                logger.error(f"Error unpublishing track: {e}")
            
            try:
                if audio_source:
                    await audio_source.aclose()
                    logger.info(f"Audio source closed for room {room_name}")
            except Exception as e:
                logger.error(f"Error closing audio source: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error playing audio to room: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
async def stream_tts_to_room(room_name: str, text: str, voice: str = "alloy", model: str = "gpt-4o-mini-tts", instructions: str = ""):
    """Stream text-to-speech to a LiveKit room using LiveKit's resampler"""
    if room_name not in active_rooms:
        logger.error(f"Room not found: {room_name}")
        return False
    
    try:
        room = active_rooms[room_name]
        
        # Initialize OpenAI client
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # Create an audio source at the target sample rate (48kHz is standard for WebRTC)
        target_sample_rate = 48000
        audio_source = rtc.AudioSource(
            sample_rate=target_sample_rate,
            num_channels=1,
            queue_size_ms=1000
        )
        
        # Create and publish a local audio track
        track = rtc.LocalAudioTrack.create_audio_track("tts-stream", audio_source)
        publication = None
        
        try:
            # Publish the track
            publication = await room.local_participant.publish_track(track)
            logger.info(f"TTS track published to room {room_name}")
            
            # Create a buffer manager
            buffer_manager = AudioBufferManager(
                room_name=room_name,
                audio_source=audio_source,
                sample_rate=target_sample_rate
            )
            
            # Create a resampler to convert from OpenAI's 24kHz to our target 48kHz
            # Use HIGH quality for better audio
            resampler = rtc.AudioResampler(
                input_rate=24000,  # OpenAI PCM is 24kHz
                output_rate=target_sample_rate,
                num_channels=1,
                quality=rtc.AudioResamplerQuality.HIGH
            )
            
            # Create a streaming response from OpenAI
            async with client.audio.speech.with_streaming_response.create(
                model=model,
                voice=voice,
                input=text,
                instructions=instructions,
                response_format="pcm"  # Raw PCM for better streaming
            ) as response:
                # Process streaming chunks
                buffer = bytearray()
                samples_per_frame = 480  # 20ms at 24kHz
                bytes_per_sample = 2     # int16 = 2 bytes
                frame_size = samples_per_frame * bytes_per_sample
                
                async for chunk in response.iter_bytes():
                    if chunk:
                        # Add to buffer
                        buffer.extend(chunk)
                        
                        # Process complete frames
                        while len(buffer) >= frame_size:
                            # Extract a frame
                            frame_data = buffer[:frame_size]
                            del buffer[:frame_size]
                            
                            # Create an AudioFrame with OpenAI's sample rate
                            input_frame = rtc.AudioFrame(
                                data=bytes(frame_data),
                                sample_rate=24000,
                                num_channels=1,
                                samples_per_channel=samples_per_frame
                            )
                            
                            # Resample to target sample rate using LiveKit's resampler
                            resampled_frames = resampler.push(input_frame)
                            
                            # Process each resampled frame
                            for frame in resampled_frames:
                                # Convert to numpy for buffer manager
                                frame_data = frame.data
                                samples = np.frombuffer(frame_data, dtype=np.int16).astype(np.float32) / 32767.0
                                
                                # Calculate chunk duration
                                chunk_duration = len(samples) / target_sample_rate
                                
                                # Send chunk through buffer manager
                                sleep_time = await buffer_manager.send_chunk(samples, chunk_duration)
                                
                                # Sleep for the recommended duration
                                await asyncio.sleep(sleep_time)
                
                # Process any remaining data in buffer
                if buffer:
                    # Pad to a complete frame if needed
                    if len(buffer) % bytes_per_sample != 0:
                        buffer.extend(b'\x00' * (bytes_per_sample - (len(buffer) % bytes_per_sample)))
                    
                    # Create final frame
                    final_samples = len(buffer) // bytes_per_sample
                    input_frame = rtc.AudioFrame(
                        data=bytes(buffer),
                        sample_rate=24000,
                        num_channels=1,
                        samples_per_channel=final_samples
                    )
                    
                    # Resample and send
                    final_frames = resampler.push(input_frame)
                    for frame in final_frames:
                        samples = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32767.0
                        chunk_duration = len(samples) / target_sample_rate
                        await buffer_manager.send_chunk(samples, chunk_duration)
                
                # Flush the resampler
                flush_frames = resampler.flush()
                for frame in flush_frames:
                    samples = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32767.0
                    chunk_duration = len(samples) / target_sample_rate
                    await buffer_manager.send_chunk(samples, chunk_duration)
            
            logger.info(f"Finished streaming TTS to room {room_name}")
            
            # Wait for the audio source to finish playing everything
            while audio_source.queued_duration > 0:
                await asyncio.sleep(0.1)
            
            logger.info(f"TTS playback completed for room {room_name}")
            
        finally:
            # Clean up resources
            try:
                if publication:
                    await publication.unpublish()
                    logger.info(f"TTS track unpublished from room {room_name}")
                if track:
                    track.stop()
            except Exception as e:
                logger.error(f"Error cleaning up TTS track: {e}")
            
            try:
                if audio_source:
                    await audio_source.aclose()
                    logger.info(f"TTS audio source closed for room {room_name}")
            except Exception as e:
                logger.error(f"Error closing TTS audio source: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error streaming TTS to room: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# API Endpoints
@app.post("/get-token")
async def get_token(request: TokenRequest):
    """Get a LiveKit token for the frontend"""
    try:
        # Generate a token for the participant
        token = create_token(request.room_name, request.participant_name)
        
        return {
            "token": token,
            "serverUrl": LIVEKIT_URL
        }
    except Exception as e:
        logger.error(f"Error generating token: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    """Upload an audio file to the server"""
    try:
        # Generate a unique ID for the file
        file_id = str(uuid.uuid4())
        
        # Create a temporary file path
        file_path = f"temp_{file_id}_{file.filename}"
        
        # Write the file to disk
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Load the audio file
        success = await load_audio_file(file_path, file_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process audio file")
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error uploading audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/play-audio")
async def play_audio(request: PlaybackRequest):
    """Play an audio file to a LiveKit room"""
    try:
        # Check if we're already connected to the room
        if request.room_name not in active_rooms:
            # Connect to the room
            await connect_to_room(request.room_name, "audio-server")
        
        # Start playback in a background task
        asyncio.create_task(play_audio_to_room(request.room_name, request.file_id))
        
        return {
            "status": "started",
            "room_name": request.room_name,
            "file_id": request.file_id
        }
    except Exception as e:
        logger.error(f"Error playing audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/leave-room/{room_name}")
async def leave_room(room_name: str):
    """Leave a LiveKit room"""
    if room_name not in active_rooms:
        return {"status": "not_found"}
    
    try:
        # Disconnect from the room
        await active_rooms[room_name].disconnect()
        del active_rooms[room_name]
        
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error leaving room: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts-stream")
async def tts_stream(request: TTSRequest):
    """Stream text-to-speech to a LiveKit room"""
    try:
        # Check if we're already connected to the room
        if request.room_name not in active_rooms:
            # Connect to the room
            await connect_to_room(request.room_name, "tts-server")
        
        # Start streaming in a background task
        asyncio.create_task(
            stream_tts_to_room(
                request.room_name,
                request.text,
                request.voice,
                request.model,
                request.instructions
            )
        )
        
        return {
            "status": "started",
            "room_name": request.room_name,
            "message": "TTS streaming started"
        }
    except Exception as e:
        logger.error(f"Error starting TTS stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files from the public directory
app.mount("/", StaticFiles(directory="public", html=True), name="public")

# Run the server
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=3001, reload=True)