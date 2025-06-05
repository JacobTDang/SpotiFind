// frontend/src/components/Spotifind.js

import React, { useState, useRef, useEffect } from 'react';
import { Music } from 'lucide-react';

// Import components
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

  // Add state for recording configuration
  const [maxRecordingTime, setMaxRecordingTime] = useState(60);
  const [recordingQuality, setRecordingQuality] = useState('high');

  // Song identification state
  const [matchedSong, setMatchedSong] = useState(null);
  const [confidence, setConfidence] = useState(0);

  // Refs for audio recording
  const recordingInterval = useRef(null);
  const audioChunks = useRef([]);

  // Base Flask URL
  const FLASK_URL = 'http://localhost:5000';

  // CONNECTION TEST FUNCTION
  const testConnection = async () => {
    try {
      const response = await fetch(`${FLASK_URL}/test`);
      const data = await response.json();
      console.log('Connection test:', data);
    } catch (error) {
      console.error('Connection failed:', error);
    }
  };

  // CLEANUP EFFECT - Cleanup recording interval on unmount
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

  // ENHANCED RECORDING FUNCTION WITH CONFIGURABLE SETTINGS
  const startRecording = async (duration = 60, quality = 'high') => {
    try {
      // Quality-based audio constraints
      const getAudioConstraints = (quality) => {
        const baseConstraints = {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: false, // Disable AGC for music recording
        };

        switch (quality) {
          case 'high':
            return {
              ...baseConstraints,
              sampleRate: 48000,
              channelCount: 2, // Stereo for better quality
              latency: 0.01,
              volume: 1.0,
            };
          case 'medium':
            return {
              ...baseConstraints,
              sampleRate: 44100,
              channelCount: 1,
              latency: 0.02,
              volume: 1.0,
            };
          case 'low':
            return {
              ...baseConstraints,
              sampleRate: 22050,
              channelCount: 1,
              latency: 0.05,
              volume: 1.0,
            };
          default:
            return baseConstraints;
        }
      };

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: getAudioConstraints(quality),
      });

      // Enhanced MediaRecorder configuration for longer recordings
      const getRecorderOptions = (quality) => {
        switch (quality) {
          case 'high':
            return {
              mimeType: 'audio/webm;codecs=opus',
              audioBitsPerSecond: 256000, // Higher quality for longer clips
            };
          case 'medium':
            return {
              mimeType: 'audio/webm;codecs=opus',
              audioBitsPerSecond: 128000,
            };
          case 'low':
            return {
              mimeType: 'audio/webm;codecs=opus',
              audioBitsPerSecond: 64000,
            };
          default:
            return {
              mimeType: 'audio/webm;codecs=opus',
              audioBitsPerSecond: 128000,
            };
        }
      };

      const recorder = new MediaRecorder(stream, getRecorderOptions(quality));
      audioChunks.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        const blob = new Blob(audioChunks.current, { type: 'audio/webm' });
        setAudioBlob(blob);
        stream.getTracks().forEach((track) => track.stop());

        console.log(`Recording completed: ${recordingTime}s, size: ${blob.size} bytes, quality: ${quality}`);

        // Show recording completed message
        if (recordingTime >= 45) {
          showMessage(`Excellent ${recordingTime}s recording captured! This should give great results.`, 'success');
        } else if (recordingTime >= 30) {
          showMessage(`Good ${recordingTime}s recording captured!`, 'success');
        } else if (recordingTime >= 15) {
          showMessage(`${recordingTime}s recording captured. Consider longer recordings for better accuracy.`, 'success');
        }
      };

      setMediaRecorder(recorder);
      recorder.start(1000); // Collect data every second
      setIsRecording(true);
      setRecordingTime(0);
      setMaxRecordingTime(duration);
      setRecordingQuality(quality);

      // Auto-stop timer with configurable duration
      recordingInterval.current = setInterval(() => {
        setRecordingTime((prev) => {
          if (prev >= duration) {
            stopRecording();
            return prev;
          }
          return prev + 1;
        });
      }, 1000);

      showMessage(`Started ${duration}s recording in ${quality} quality`, 'success');
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

  // ENHANCED CLEAR FUNCTION
  const clearRecording = () => {
    setAudioBlob(null);
    setRecordingTime(0);
    setMatchedSong(null);
    setConfidence(0);

    if (recordingInterval.current) {
      clearInterval(recordingInterval.current);
    }

    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
    }
  };

  // ENHANCED AUDIO SEARCH WITH PROGRESS INDICATION
  const handleAudioSearch = async () => {
    if (!audioBlob) {
      setResults([]);
      showMessage('Please record audio first', 'error');
      return;
    }

    setIsLoading(true);
    setResults([]);

    // Show different messages based on recording length
    if (recordingTime >= 45) {
      showMessage('Analyzing your excellent long recording... This may take a moment but should give great results!', 'success');
    } else if (recordingTime >= 30) {
      showMessage('Analyzing your recording... Longer clips provide better accuracy!', 'success');
    } else {
      showMessage('Analyzing your recording...', 'success');
    }

    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');

    // Add metadata about the recording
    formData.append('recording_duration', recordingTime.toString());
    formData.append('recording_quality', recordingQuality);

    try {
      const response = await fetch(`${FLASK_URL}/search-audio`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok && data.similar_songs && data.similar_songs.length > 0) {
        setResults(data.similar_songs);
        const bestMatch = data.similar_songs[0];
        setMatchedSong(bestMatch);
        setConfidence(bestMatch.confidence || 0);

        // Enhanced success message for longer recordings
        let successMessage = `Found ${data.similar_songs.length} matches!`;
        if (recordingTime >= 45) {
          successMessage += ` Your ${recordingTime}s recording provided excellent analysis.`;
        } else if (recordingTime >= 30) {
          successMessage += ` Good ${recordingTime}s recording length helped accuracy.`;
        }

        if (data.debug_info) {
          successMessage += ` Debug: ${data.debug_info.session_id}`;
        }

        showMessage(successMessage, 'success');
      } else {
        setResults([]);
        setMatchedSong(false);
        setConfidence(0);

        let errorMessage = data.message || 'No similar songs found';
        if (recordingTime < 30) {
          errorMessage += '. Try recording longer (30-60s) for better results.';
        }
        showMessage(errorMessage, 'error');
      }
    } catch (error) {
      setResults([]);
      setMatchedSong(false);
      setConfidence(0);
      showMessage('Network error during audio search', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  // FILE VALIDATION HANDLER
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

          {activeTab === 'youtube' && (
            <YouTubeTab
              youtubeUrl={youtubeUrl}
              isLoading={isLoading}
              setYoutubeUrl={setYoutubeUrl}
              handleYouTubeAdd={handleYouTubeAdd}
              setIsLoading={setIsLoading}
              showMessage={showMessage}
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
              startRecording={startRecording} // Now accepts duration and quality
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
