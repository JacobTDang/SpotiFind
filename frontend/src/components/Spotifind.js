import React, { useState, useRef, useEffect } from 'react';
import { Music } from 'lucide-react';

// Import components - Remove PlaylistTab since it's now inside YouTubeTab
import TabNavigation from './TabNavigation';
import MessageDisplay from './MessageDisplay';
import UploadTab from './UploadTab';
import YouTubeTab from './YouTubeTab';
import RecordingTab from './RecordingTab';
import ResultsSection from './ResultsSection';

const Spotifind = () => {
  // MAIN STATE - Only 3 tabs now: upload, youtube, search
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

  // Song identification state
  const [matchedSong, setMatchedSong] = useState(null);
  const [confidence, setConfidence] = useState(0);

  // Refs for audio recording
  const recordingInterval = useRef(null);
  const audioChunks = useRef([]);

  // Base flask URL - hopefully dont matter because im only hosting it locally
  const FLASK_URL = 'http://localhost:5000';

  // CONNECTION TEST FUNCTION
  const testConnection = async () => {
    try {
      const response = await fetch('http://localhost:5000/test');
      const data = await response.json();
      console.log('Connection test:', data);
    } catch (error) {
      console.error('Connection failed:', error);
    }
  };

  //  CLEANUP EFFECT - Cleanup recording interval on unmount
  useEffect(() => {
    return () => {
      if (recordingInterval.current) {
        clearInterval(recordingInterval.current);
      }
    };
  }, []);

  // INITIALIZATION EFFECT - Test connection when component mounts
  useEffect(() => {
    testConnection();
  }, []);

  // MESSAGE UTILITY FUNCTION
  const showMessage = (text, type) => {
    setMessage(text);
    setMessageType(type);
    setTimeout(() => setMessage(''), 5000);
  };

  // FILE UPLOAD HANDLER
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

    try {
      const response = await fetch(`${FLASK_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        showMessage(`Successfully uploaded: ${data.song}`, 'success');
        // FORM RESET PATTERN - Clear form after successful upload
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

  // YOUTUBE SINGLE VIDEO HANDLER
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
        setYoutubeUrl(''); // Clear URL on success
      } else {
        showMessage(data.message || 'Failed to add YouTube video', 'error');
      }
    } catch (error) {
      showMessage('Network error during YouTube processing', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  // RECORDING FUNCTIONS
  const startRecording = async () => {
    try {
      // MEDIA CONSTRAINTS PATTERN - Optimized audio settings
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 44100,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          latency: 0.01,
          volume: 1.0
        }
      });

      // MEDIA RECORDER PATTERN - Configure for quality
      const recorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
        audioBitsPerSecond: 128000
      });

      audioChunks.current = [];

      // EVENT HANDLER PATTERN - Handle recorder events
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        const blob = new Blob(audioChunks.current, { type: 'audio/webm' });
        setAudioBlob(blob);
        stream.getTracks().forEach(track => track.stop());
        console.log(`Recording completed: ${recordingTime}s, size: ${blob.size} bytes`);
      };

      setMediaRecorder(recorder);
      recorder.start(1000); // Collect data every second
      setIsRecording(true);
      setRecordingTime(0);

      // INTERVAL PATTERN - Auto-stop after 20 seconds
      recordingInterval.current = setInterval(() => {
        setRecordingTime(prev => {
          if (prev >= 20) {
            stopRecording();
            return prev;
          }
          return prev + 1;
        });
      }, 1000);

    } catch (error) {
      console.error('Recording error:', error);
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
    setMatchedSong(null);
    setConfidence(0);
  };

  // AUDIO SEARCH HANDLER with improved error handling
  const handleAudioSearch = async () => {
    // Early validation with state cleanup
    if (!audioBlob) {
      setResults([]);
      showMessage('Please record audio first', 'error');
      return;
    }

    // Set loading state and clear previous results
    setIsLoading(true);
    setResults([]); // Clear immediately when starting new search

    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');

    try {
      const response = await fetch(`${FLASK_URL}/search-audio`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

// Conditional state updates based on response
 if (response.ok && data.similar_songs && data.similar_songs.length > 0) {
   setResults(data.similar_songs);
  // Set the best match for RecordingTab
  const bestMatch = data.similar_songs[0];
  setMatchedSong(bestMatch);
  setConfidence(bestMatch.confidence || 0);
   showMessage(`Found ${data.similar_songs.length} similar songs`, 'success');
 } else {
   // Explicitly set empty results for failed searches
   setResults([]);
  setMatchedSong(false); // Indicates no match found
  setConfidence(0);
   showMessage(data.message || 'No similar songs found', 'error');
 }
    } catch (error) {
      // Error handling with state cleanup
      setResults([]);
      showMessage('Network error during audio search', 'error');
    } finally {
      // Always cleanup loading state
      setIsLoading(false);
    }
  };

  //  FILE VALIDATION HANDLER
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
          <p className="text-gray-300">Upload audio, add YouTube videos, and identify songs using AI</p>
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

          {/*  FIXED: Now passing all required props to YouTubeTab */}
          {activeTab === 'youtube' && (
            <YouTubeTab
              youtubeUrl={youtubeUrl}
              isLoading={isLoading}
              setYoutubeUrl={setYoutubeUrl}
              handleYouTubeAdd={handleYouTubeAdd}
              setIsLoading={setIsLoading}    // ← This was missing!
              showMessage={showMessage}      // ← This was missing!
            />
          )}

          {activeTab === 'search' && (
            <RecordingTab
              audioBlob={audioBlob}
              isRecording={isRecording}
              recordingTime={recordingTime}
              isLoading={isLoading}
              matchedSong={matchedSong}
              confidence={confidence}
              startRecording={startRecording}
              stopRecording={stopRecording}
              clearRecording={clearRecording}
              handleAudioSearch={handleAudioSearch}
              setMatchedSong={setMatchedSong}
            />
          )}
        </div>

        {/* Results Section - Show detailed results when not in identification mode */}
        {results.length > 0 && activeTab !== 'search' && (
          <ResultsSection results={results} />
        )}

        {/* Show results for search tab only when there are multiple results or no match found */}
        {activeTab === 'search' && results.length > 1 && (
          <div className="max-w-4xl mx-auto mt-8">
            <div className="bg-gray-800/50 backdrop-blur-md rounded-xl p-8 border border-gray-700">
              <h3 className="text-xl font-bold text-white mb-4">Other Possible Matches</h3>
              <ResultsSection results={results.slice(1)} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Spotifind;
