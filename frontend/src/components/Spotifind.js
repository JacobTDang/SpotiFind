import React, { useState, useRef, useEffect } from 'react';
import { Music } from 'lucide-react';

// Import all the new components
import TabNavigation from './TabNavigation';
import MessageDisplay from './MessageDisplay';
import UploadTab from './UploadTab';
import YouTubeTab from './YouTubeTab';
import RecordingTab from './RecordingTab';
import ResultsSection from './ResultsSection';
import PlaylistTab from './PlaylistTab';

const Spotifind = () => {
  const [activeTab, setActiveTab] = useState('upload');
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState('');

  // Upload form state
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadTitle, setUploadTitle] = useState('');
  const [uploadArtist, setUploadArtist] = useState('');

  // YouTube form state
  const [youtubeUrl, setYoutubeUrl] = useState('');

  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [audioBlob, setAudioBlob] = useState(null);

  // Refs for audio recording
  const recordingInterval = useRef(null);
  const audioChunks = useRef([]);
  // test flask end point to check if cors connection works
  const testConnection = async () => {
      try {
        const response = await fetch('http://localhost:5000/test');
        const data = await response.json();
        console.log('Connection test:', data);
      } catch (error) {
        console.error('Connection failed:', error);
      }
    };
  // Cleanup recording interval
  useEffect(() => {
    return () => {
      if (recordingInterval.current) {
        clearInterval(recordingInterval.current);
      }
    };
  }, []);
// Test connection when component mounts
  useEffect(() => {
    testConnection();
  }, []);
  // Base flask URL
  const FLASK_URL = 'http://localhost:5000';

  const showMessage = (text, type) => {
    setMessage(text);
    setMessageType(type);
    setTimeout(() => setMessage(''), 5000);
  };

  const handleFileUpload = async () => {
    if (!uploadFile || !uploadTitle || !uploadArtist) {
      showMessage('Please fill in all fields and select a file', 'error');
      return;
    }

    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', uploadFile);
    formData.append('title', uploadTitle);
    formData.append('artist', uploadArtist);
    // Add this to test connection

    try {
      const response = await fetch(`${FLASK_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        showMessage(`Successfully uploaded: ${data.song}`, 'success');
        setUploadFile(null);
        setUploadTitle('');
        setUploadArtist('');
        document.getElementById('file-input').value = '';
      } else {
        showMessage(data.message || 'Upload failed', 'error');
      }
    } catch (error) {
      showMessage('Network error during upload', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleYouTubeAdd = async () => {
    if (!youtubeUrl) {
      showMessage('Please enter a YouTube URL', 'error');
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch(`${FLASK_URL}/youtube`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: youtubeUrl }),
      });

      const data = await response.json();

      if (data.success) {
        showMessage(`Successfully added: ${data.song}`, 'success');
        setYoutubeUrl('');
      } else {
        showMessage(data.message || 'Failed to add YouTube video', 'error');
      }
    } catch (error) {
      showMessage('Network error during YouTube processing', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 44100,
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false
        }
      });

      const recorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      audioChunks.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        const blob = new Blob(audioChunks.current, { type: 'audio/webm' });
        setAudioBlob(blob);
        stream.getTracks().forEach(track => track.stop());
      };

      setMediaRecorder(recorder);
      recorder.start(1000);
      setIsRecording(true);
      setRecordingTime(0);

      recordingInterval.current = setInterval(() => {
        setRecordingTime(prev => {
          if (prev >= 10) {
            stopRecording();
            return prev;
          }
          return prev + 1;
        });
      }, 1000);

    } catch (error) {
      showMessage('Microphone access denied or not available', 'error');
    }
  };

  const stopRecording = () => {
    if (mediaRecorder && isRecording) {
      mediaRecorder.stop();
      setIsRecording(false);

      if (recordingInterval.current) {
        clearInterval(recordingInterval.current);
      }
    }
  };

  const clearRecording = () => {
    setAudioBlob(null);
    setRecordingTime(0);
  };

  const handleAudioSearch = async () => {
    if (!audioBlob) {
      showMessage('Please record audio first', 'error');
      return;
    }

    setIsLoading(true);
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');

    try {
      const response = await fetch(`${FLASK_URL}/search-audio`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok && data.similar_songs && data.similar_songs.length > 0) {
        setResults(data.similar_songs);
        showMessage(`Found ${data.similar_songs.length} similar songs`, 'success');
      } else {
        setResults([]);
        showMessage(data.message || 'No similar songs found', 'error');
      }
    } catch (error) {
      showMessage('Network error during audio search', 'error');
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const validTypes = ['audio/mpeg', 'audio/wav', 'audio/mp3', 'audio/m4a', 'audio/ogg'];
      if (!validTypes.includes(file.type) && !file.name.match(/\.(mp3|wav|m4a|ogg)$/i)) {
        showMessage('Please select a valid audio file (MP3, WAV, M4A, OGG)', 'error');
        return;
      }

      if (file.size > 50 * 1024 * 1024) {
        showMessage('File size must be less than 50MB', 'error');
        return;
      }

      setUploadFile(file);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2 flex items-center justify-center gap-3">
            <Music className="w-10 h-10" />
            Music Similarity Search
          </h1>
          <p className="text-gray-300">Upload audio, add YouTube videos, and find similar songs using AI</p>
        </div>

        {/* Message Display */}
        <MessageDisplay message={message} messageType={messageType} />

        {/* Tab Navigation */}
        <TabNavigation activeTab={activeTab} setActiveTab={setActiveTab} />

        {/* Tab Content */}
        <div className="max-w-2xl mx-auto">
          {activeTab === 'upload' && (
            <UploadTab
              uploadFile={uploadFile}
              uploadTitle={uploadTitle}
              uploadArtist={uploadArtist}
              isLoading={isLoading}
              setUploadTitle={setUploadTitle}
              setUploadArtist={setUploadArtist}
              handleFileChange={handleFileChange}
              handleFileUpload={handleFileUpload}
            />
          )}

          {activeTab === 'youtube' && (
            <YouTubeTab
              youtubeUrl={youtubeUrl}
              isLoading={isLoading}
              setYoutubeUrl={setYoutubeUrl}
              handleYouTubeAdd={handleYouTubeAdd}
            />
          )}

          {activeTab === 'search' && (
            <RecordingTab
              audioBlob={audioBlob}
              isRecording={isRecording}
              recordingTime={recordingTime}
              isLoading={isLoading}
              startRecording={startRecording}
              stopRecording={stopRecording}
              clearRecording={clearRecording}
              handleAudioSearch={handleAudioSearch}
            />
          )}

          {activate === 'playlist' && (
            <PlaylistTab
            isloading={isLoading}
            setIsLoading={setIsLoading}
            showMessages={showMessage}
            />
          )}
        </div>

        {/* Results Section */}
        <ResultsSection results={results} />
      </div>
    </div>
  );
};

export default Spotifind;
