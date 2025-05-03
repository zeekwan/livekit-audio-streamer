document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const roomNameInput = document.getElementById('room-name');
  const participantNameInput = document.getElementById('participant-name');
  const connectButton = document.getElementById('connect-button');
  const connectionStatus = document.getElementById('connection-status');
  
  const audioFileInput = document.getElementById('audio-file');
  const uploadButton = document.getElementById('upload-button');
  const fileStatus = document.getElementById('file-status');
  
  const playButton = document.getElementById('play-button');
  const stopButton = document.getElementById('stop-button');
  const playbackStatus = document.getElementById('playback-status');
  
  const conversationDiv = document.getElementById('conversation');
  const localPreview = document.getElementById('local-preview');
  const localPlaybackToggle = document.getElementById('local-playback-toggle');

  const ttsText = document.getElementById('tts-text');
  const ttsVoice = document.getElementById('tts-voice');
  const speakButton = document.getElementById('speak-button');
  const stopSpeechButton = document.getElementById('stop-speech-button');
  const ttsStatus = document.getElementById('tts-status');
  
  // LiveKit Room
  let room = null;
  let isConnected = false;

  // TTS state
  let isSpeaking = false;
  
  // File tracking
  let currentFileId = null;
  let isPlaying = false;

  // API endpoints (update with your server URL if different)
  const API_BASE = '';  // Empty string if serving from same origin
  const TOKEN_ENDPOINT = `${API_BASE}/get-token`;  // Use our local server endpoint
  const UPLOAD_ENDPOINT = `${API_BASE}/upload-audio`;
  const PLAY_ENDPOINT = `${API_BASE}/play-audio`;
  const LEAVE_ENDPOINT = `${API_BASE}/leave-room`;
  const TTS_ENDPOINT = `${API_BASE}/tts-stream`;

  // Add logging function
  function log(message, type = 'info') {
    // Console log
    console.log(message);
    
    // Add to conversation div
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${type}`;
    
    const timestamp = new Date().toLocaleTimeString();
    logEntry.textContent = `[${timestamp}] ${message}`;
    
    conversationDiv.appendChild(logEntry);
    conversationDiv.scrollTop = conversationDiv.scrollHeight;
  }

  // Update status message
  function updateStatus(element, message, className = '') {
    element.textContent = message;
    element.className = 'status ' + className;
  }

  // Connect to the LiveKit room through our Python backend
  async function connectToRoom() {
    try {
      const roomName = roomNameInput.value.trim() || 'audio-room';
      const participantName = participantNameInput.value.trim() || `user-${Date.now()}`;
      
      updateStatus(connectionStatus, 'Connecting...', 'processing');
      log(`Connecting to room: ${roomName} as ${participantName}...`);
      
      // Get token from our Python backend
      const response = await fetch(TOKEN_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          room_name: roomName,
          participant_name: participantName
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to get token from server');
      }
      
      const data = await response.json();
      
      // Create and connect to room
      room = new LivekitClient.Room({
        adaptiveStream: true,
        dynacast: true,
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      
      // Add event listeners
      setupRoomListeners();
      
      // Connect to the room
      await room.connect(data.serverUrl, data.token);
      log(`Connected to LiveKit room: ${room.name}`);
      
      isConnected = true;
      updateStatus(connectionStatus, `Connected to ${roomName}`, 'success');
      
      // Enable buttons
      uploadButton.disabled = false;
      speakButton.disabled = false;
      
      return true;
    } catch (error) {
      log(`Connection error: ${error.message}`, 'error');
      updateStatus(connectionStatus, 'Connection failed', 'error');
      return false;
    }
  }

  // Setup room event listeners
  function setupRoomListeners() {
    if (!room) return;
    
    room
      .on(LivekitClient.RoomEvent.Connected, () => {
        log('Room connection established', 'info');
      })
      .on(LivekitClient.RoomEvent.Disconnected, () => {
        log('Room disconnected', 'warn');
        handleDisconnect();
      })
      .on(LivekitClient.RoomEvent.Reconnecting, () => {
        log('Room connection lost, attempting to reconnect...', 'warn');
      })
      .on(LivekitClient.RoomEvent.Reconnected, () => {
        log('Room connection reestablished', 'info');
      })
      .on(LivekitClient.RoomEvent.TrackSubscribed, (track, publication, participant) => {
        log(`Subscribed to ${participant.identity}'s ${track.kind} track`, 'info');
        
        // Play audio tracks when subscribed
        if (track.kind === 'audio') {
          track.attach();  // This automatically creates and attaches an audio element
          log(`Playing audio from ${participant.identity}`, 'info');
        }
      })
      .on(LivekitClient.RoomEvent.TrackUnsubscribed, (track, publication, participant) => {
        log(`Unsubscribed from ${participant.identity}'s ${track.kind} track`, 'info');
        
        // Clean up audio tracks when unsubscribed
        if (track.kind === 'audio') {
          track.detach();
        }
      })
      .on(LivekitClient.RoomEvent.ParticipantConnected, (participant) => {
        log(`Participant joined: ${participant.identity}`, 'info');
      })
      .on(LivekitClient.RoomEvent.ParticipantDisconnected, (participant) => {
        log(`Participant left: ${participant.identity}`, 'warn');
      })
      .on(LivekitClient.RoomEvent.AudioPlaybackStatusChanged, () => {
        const canPlayback = room.canPlaybackAudio;
        log(`Audio playback status changed: ${canPlayback ? 'enabled' : 'disabled'}`, 
          canPlayback ? 'info' : 'warn');
      });
  }

  // Handle room disconnect
  function handleDisconnect() {
    isConnected = false;
    room = null;
    
    // Disable buttons
    uploadButton.disabled = true;
    playButton.disabled = true;
    stopButton.disabled = true;
    speakButton.disabled = true;
    stopSpeechButton.disabled = true;
    
    // Update status
    updateStatus(connectionStatus, 'Disconnected', 'error');
  }

  // Upload audio file to server
  async function uploadAudioFile() {
    try {
      if (!audioFileInput.files || audioFileInput.files.length === 0) {
        log('No file selected', 'error');
        return false;
      }
      
      const file = audioFileInput.files[0];
      log(`Uploading file: ${file.name}`);
      updateStatus(fileStatus, 'Uploading...', 'processing');
      
      // Create form data
      const formData = new FormData();
      formData.append('file', file);
      
      // Upload to server
      const response = await fetch(UPLOAD_ENDPOINT, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error('Failed to upload file');
      }
      
      const data = await response.json();
      currentFileId = data.file_id;
      
      log(`File uploaded successfully. File ID: ${currentFileId}`);
      updateStatus(fileStatus, `File ready: ${file.name}`, 'success');
      
      // Create local preview
      localPreview.src = URL.createObjectURL(file);
      
      // Enable play button
      playButton.disabled = false;
      
      return true;
    } catch (error) {
      log(`Upload error: ${error.message}`, 'error');
      updateStatus(fileStatus, 'Upload failed', 'error');
      return false;
    }
  }

  // Play audio file to the room
  async function playAudioToRoom() {
    try {
      if (!isConnected) {
        log('Not connected to room', 'error');
        return false;
      }
      
      if (!currentFileId) {
        log('No file uploaded', 'error');
        return false;
      }
      
      log('Starting audio playback...');
      updateStatus(playbackStatus, 'Playing...', 'listening');
      
      // Request playback from server
      const response = await fetch(PLAY_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          room_name: roomNameInput.value.trim() || 'audio-room',
          file_id: currentFileId
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to start playback');
      }
      
      isPlaying = true;
      stopButton.disabled = false;
      playButton.disabled = true;
      
      // Play locally only if toggle is enabled
      if (localPlaybackToggle.checked) {
        localPreview.play();
      }
      
      return true;
    } catch (error) {
      log(`Playback error: ${error.message}`, 'error');
      updateStatus(playbackStatus, 'Playback failed', 'error');
      return false;
    }
  }

  // Stop audio playback
  function stopAudioPlayback() {
    if (localPreview) {
      localPreview.pause();
      localPreview.currentTime = 0;
    }
    
    isPlaying = false;
    stopButton.disabled = true;
    playButton.disabled = false;
    
    updateStatus(playbackStatus, 'Stopped', '');
    log('Audio playback stopped');
  }

  // TTS Functions
  async function startTTS() {
    try {
      if (!isConnected) {
        log('Not connected to room', 'error');
        return;
      }
      
      const text = ttsText.value.trim();
      if (!text) {
        log('No text to speak', 'error');
        return;
      }
      
      const voice = ttsVoice.value;
      
      updateStatus(ttsStatus, 'Starting TTS...', 'processing');
      log(`Starting TTS with voice: ${voice}`);
      
      // Call the backend to start TTS
      const response = await fetch(TTS_ENDPOINT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          room_name: roomNameInput.value.trim() || 'audio-room',
          text: text,
          voice: voice,
          model: "gpt-4o-mini-tts",
          instructions: ""
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to start TTS');
      }
      
      isSpeaking = true;
      speakButton.disabled = true;
      stopSpeechButton.disabled = false;
      updateStatus(ttsStatus, 'Speaking...', 'listening');
      
    } catch (error) {
      log(`TTS error: ${error.message}`, 'error');
      updateStatus(ttsStatus, 'TTS failed', 'error');
      isSpeaking = false;
      speakButton.disabled = false;
      stopSpeechButton.disabled = true;
    }
  }
  
  async function stopTTS() {
    try {
      // For now, we don't have a stop endpoint, but we can add one later
      // For now, just update the UI state
      isSpeaking = false;
      speakButton.disabled = false;
      stopSpeechButton.disabled = true;
      updateStatus(ttsStatus, 'Stopped', '');
      log('TTS stopped');
      
    } catch (error) {
      log(`Stop TTS error: ${error.message}`, 'error');
    }
  }

  // Disconnect from the room
  async function disconnectFromRoom() {
    if (!room) return;
    
    try {
      log('Disconnecting from room...');
      
      // First stop any playing audio
      if (isPlaying) {
        stopAudioPlayback();
      }
      
      // Stop TTS if running
      if (isSpeaking) {
        await stopTTS();
      }
      
      // Notify server to clean up
      const roomName = roomNameInput.value.trim() || 'audio-room';
      await fetch(`${LEAVE_ENDPOINT}/${roomName}`, {
        method: 'POST'
      });
      
      // Disconnect client
      await room.disconnect();
      
      handleDisconnect();
      log('Disconnected from room');
    } catch (error) {
      log(`Disconnect error: ${error.message}`, 'error');
    }
  }

  // Event listeners
  connectButton.addEventListener('click', async () => {
    if (!isConnected) {
      await connectToRoom();
      connectButton.textContent = 'Disconnect';
    } else {
      await disconnectFromRoom();
      connectButton.textContent = 'Connect to Room';
    }
  });

  audioFileInput.addEventListener('change', () => {
    if (audioFileInput.files && audioFileInput.files.length > 0) {
      const file = audioFileInput.files[0];
      updateStatus(fileStatus, `Selected: ${file.name}`, '');
      
      // Enable upload button if connected
      uploadButton.disabled = !isConnected;
    } else {
      updateStatus(fileStatus, 'No file selected', '');
      uploadButton.disabled = true;
    }
  });

  uploadButton.addEventListener('click', uploadAudioFile);
  playButton.addEventListener('click', playAudioToRoom);
  stopButton.addEventListener('click', stopAudioPlayback);

  speakButton.addEventListener('click', startTTS);
  stopSpeechButton.addEventListener('click', stopTTS);

  // Handle page unload
  window.addEventListener('beforeunload', () => {
    disconnectFromRoom();
  });

  // Initialize
  updateStatus(connectionStatus, 'Not connected', '');
  updateStatus(fileStatus, 'No file selected', '');
  updateStatus(playbackStatus, 'Ready', '');
  updateStatus(ttsStatus, 'Ready', '');
});