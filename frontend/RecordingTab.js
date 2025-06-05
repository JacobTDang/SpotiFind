// Update your frontend/src/components/RecordingTab.js with extended recording

import React, { useState } from 'react';
import { Search, Mic, Square, Play, Trash2, Loader2, Music, Youtube, Volume2, Clock, Settings } from 'lucide-react';
import AudioPlayback from './AudioPlayback';

const RecordingTab = ({
  audioBlob,
  isRecording,
  recordingTime,
  isLoading,
  matchedSong,
  confidence,
  startRecording,
  stopRecording,
  clearRecording,
  handleAudioSearch,
  setMatchedSong
}) => {
  // NEW STATE: Recording duration settings
  const [maxRecordingTime, setMaxRecordingTime] = useState(30); // Default 30 seconds
  const [showSettings, setShowSettings] = useState(false);
  const [recordingPlayback, setRecordingPlayback] = useState(null);
  const [searchHistory, setSearchHistory] = useState([]);

  // Recording duration options
  const durationOptions = [
    { value: 15, label: '15 seconds', description: 'Quick identification' },
    { value: 30, label: '30 seconds', description: 'Standard (recommended)' },
    { value: 45, label: '45 seconds', description: 'Extended analysis' },
    { value: 60, label: '1 minute', description: 'Long song samples' },
    { value: 90, label: '1.5 minutes', description: 'Full song sections' },
    { value: 120, label: '2 minutes', description: 'Multiple song parts' },
    { value: 180, label: '3 minutes', description: 'Very long recordings' }
  ];

  const clearAll = () => {
    clearRecording();
    setMatchedSong(null);
    setRecordingPlayback(null);
  };

  // Enhanced audio search with playback info
  const handleAudioSearchWithPlayback = async () => {
    if (!audioBlob) {
      return;
    }

    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.webm');

    try {
      const response = await fetch('http://localhost:5000/search-audio', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        if (data.similar_songs && data.similar_songs.length > 0) {
          const bestMatch = data.similar_songs[0];
          setMatchedSong(bestMatch);

          if (data.recording_playback) {
            setRecordingPlayback(data.recording_playback);

            const newHistoryItem = {
              ...data.recording_playback,
              searchResults: data.similar_songs,
              bestMatch: bestMatch,
              confidence: bestMatch.similarity ? (bestMatch.similarity * 100) : 0
            };

            setSearchHistory(prev => [newHistoryItem, ...prev.slice(0, 4)]);
          }
        } else {
          setMatchedSong(false);
          if (data.recording_playback) {
            setRecordingPlayback(data.recording_playback);
          }
        }
      }
    } catch (error) {
      console.error('Search error:', error);
    }
  };

  // Enhanced start recording with custom duration
  const startRecordingWithDuration = () => {
    startRecording(maxRecordingTime); // Pass duration to parent component
  };

  // Progress calculation
  const progressPercentage = maxRecordingTime > 0 ? (recordingTime / maxRecordingTime) * 100 : 0;
  const remainingTime = Math.max(0, maxRecordingTime - recordingTime);

  // Get time color based on remaining time
  const getTimeColor = () => {
    if (remainingTime <= 5) return 'text-red-400';
    if (remainingTime <= 15) return 'text-yellow-400';
    return 'text-white';
  };

  return (
    <div className="bg-gray-800/50 backdrop-blur-md rounded-xl p-8 border border-gray-700">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-white">Find Song by Recording</h2>

        {/* Settings Button */}
        <button
          onClick={() => setShowSettings(!showSettings)}
          className="flex items-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
        >
          <Settings className="w-4 h-4" />
          Settings
        </button>
      </div>

      <div className="space-y-6">
        {/* Recording Settings Panel */}
        {showSettings && (
          <div className="bg-gray-700/50 rounded-xl p-6 border border-gray-600">
            <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
              <Clock className="w-5 h-5" />
              Recording Duration
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {durationOptions.map((option) => (
                <button
                  key={option.value}
                  onClick={() => setMaxRecordingTime(option.value)}
                  className={`p-4 rounded-lg border-2 transition-all text-left ${
                    maxRecordingTime === option.value
                      ? 'border-white bg-white/10 text-white'
                      : 'border-gray-600 bg-gray-800/30 text-gray-300 hover:border-gray-500'
                  }`}
                >
                  <div className="font-medium">{option.label}</div>
                  <div className="text-sm opacity-75">{option.description}</div>
                </button>
              ))}
            </div>

            <div className="mt-4 p-4 bg-blue-900/30 border border-blue-700 rounded-lg">
              <p className="text-blue-200 text-sm">
                <strong>Tip:</strong> Longer recordings capture more song sections but may include more noise.
                30-60 seconds is usually optimal for most songs.
              </p>
            </div>
          </div>
        )}

        {/* Recording Tips */}
        <div className="bg-blue-900/30 p-4 rounded-lg border border-blue-700">
          <h3 className="text-blue-300 font-medium mb-2">For Best Results:</h3>
          <ul className="text-blue-200 text-sm space-y-1">
            <li>• Record close to speakers or headphones</li>
            <li>• Minimize background noise</li>
            <li>• Include vocals or prominent instruments</li>
            <li>• Record the chorus or most recognizable part</li>
            <li>• {maxRecordingTime} seconds allows capturing multiple song sections</li>
            <li>• Try different parts if first attempt fails</li>
          </ul>
        </div>

        <div className="text-center">
          <p className="text-gray-300 mb-6">
            Record up to {maxRecordingTime} seconds of music to identify the song
          </p>

          {/* Enhanced Recording Controls */}
          <div className="bg-gray-700/50 rounded-xl p-8 border border-gray-600">
            {!audioBlob ? (
              <div>
                {!isRecording ? (
                  <div className="space-y-4">
                    <button
                      onClick={startRecordingWithDuration}
                      disabled={isLoading}
                      className="bg-red-600 hover:bg-red-700 text-white font-bold py-6 px-12 rounded-full transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-4 mx-auto text-lg"
                    >
                      <Mic className="w-8 h-8" />
                      Start Recording ({maxRecordingTime}s)
                    </button>

                    <p className="text-gray-400 text-sm">
                      Click settings above to change recording duration
                    </p>
                  </div>
                ) : (
                  <div className="space-y-6">
                    {/* Enhanced Recording Display */}
                    <div className="space-y-4">
                      <div className="flex items-center justify-center gap-6">
                        <div className="animate-pulse">
                          <Mic className="w-12 h-12 text-red-400" />
                        </div>
                        <div className="text-center">
                          <div className={`text-4xl font-bold ${getTimeColor()}`}>
                            {recordingTime}s
                          </div>
                          <div className="text-gray-400 text-sm">
                            of {maxRecordingTime}s
                          </div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-gray-300">
                            -{remainingTime}s
                          </div>
                          <div className="text-gray-400 text-sm">
                            remaining
                          </div>
                        </div>
                      </div>

                      {/* Progress Bar */}
                      <div className="w-full max-w-md mx-auto">
                        <div className="w-full h-3 bg-gray-600 rounded-full overflow-hidden">
                          <div
                            className={`h-full transition-all duration-100 ${
                              remainingTime <= 5 ? 'bg-red-500' :
                              remainingTime <= 15 ? 'bg-yellow-500' :
                              'bg-green-500'
                            }`}
                            style={{ width: `${progressPercentage}%` }}
                          />
                        </div>
                        <div className="flex justify-between text-xs text-gray-400 mt-1">
                          <span>0s</span>
                          <span>{maxRecordingTime}s</span>
                        </div>
                      </div>
                    </div>

                    <button
                      onClick={stopRecording}
                      className="bg-gray-600 hover:bg-gray-700 text-white font-bold py-4 px-8 rounded-lg transition-all duration-200 flex items-center gap-3 mx-auto text-lg"
                    >
                      <Square className="w-6 h-6" />
                      Stop Recording
                    </button>

                    <p className="text-gray-400">
                      Recording will auto-stop at {maxRecordingTime} seconds
                    </p>
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-6">
                <div className="flex items-center justify-center gap-6">
                  <Play className="w-8 h-8 text-green-400" />
                  <span className="text-white font-medium text-xl">
                    Recording ready ({recordingTime}s / {maxRecordingTime}s)
                  </span>
                </div>

                {/* Recording Quality Indicator */}
                <div className="flex justify-center">
                  <div className={`px-4 py-2 rounded-lg ${
                    recordingTime >= 30 ? 'bg-green-900/30 border border-green-700' :
                    recordingTime >= 15 ? 'bg-yellow-900/30 border border-yellow-700' :
                    'bg-red-900/30 border border-red-700'
                  }`}>
                    <p className={`text-sm ${
                      recordingTime >= 30 ? 'text-green-300' :
                      recordingTime >= 15 ? 'text-yellow-300' :
                      'text-red-300'
                    }`}>
                      {recordingTime >= 30 ? '✅ Good length for analysis' :
                       recordingTime >= 15 ? '⚠️ Adequate length' :
                       '❌ Short recording - may affect accuracy'}
                    </p>
                  </div>
                </div>

                <div className="flex gap-4 justify-center">
                  <button
                    onClick={handleAudioSearchWithPlayback}
                    disabled={isLoading}
                    className="bg-white text-black font-bold py-4 px-8 rounded-lg hover:bg-gray-100 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-3 text-lg"
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="w-6 h-6 animate-spin" />
                        Identifying...
                      </>
                    ) : (
                      <>
                        <Search className="w-6 h-6" />
                        Identify Song
                      </>
                    )}
                  </button>
                  <button
                    onClick={clearAll}
                    disabled={isLoading}
                    className="bg-gray-600 hover:bg-gray-700 text-white font-bold py-4 px-8 rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-3 text-lg"
                  >
                    <Trash2 className="w-6 h-6" />
                    Clear
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Recording Playback Section */}
          {recordingPlayback && (
            <div className="mt-6">
              <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-600">
                <div className="flex items-center gap-3 mb-4">
                  <Volume2 className="w-6 h-6 text-blue-400" />
                  <h3 className="text-xl font-bold text-white">Your Recording</h3>
                  <span className="text-gray-400 text-sm">
                    ({recordingPlayback.duration?.toFixed(1)}s)
                  </span>
                </div>
                <AudioPlayback
                  playbackUrl={`http://localhost:5000${recordingPlayback.playback_url}`}
                  title="Your Recorded Audio"
                  searchId={recordingPlayback.search_id}
                  timestamp={recordingPlayback.timestamp}
                  duration={recordingPlayback.duration}
                />
                <div className="mt-3 text-center">
                  <p className="text-gray-400 text-sm">
                    Listen to understand recording quality and why similarity might be low/high
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Rest of your existing result display code... */}
          {matchedSong && (
            <div className="mt-8">
              {/* Your existing match display code */}
            </div>
          )}

          {matchedSong === false && (
            <div className="mt-8">
              {/* Your existing no match display code */}
            </div>
          )}

          {/* Search History */}
          {searchHistory.length > 0 && (
            <div className="mt-8">
              <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-600">
                <h3 className="text-xl font-bold text-white mb-4">Recent Searches</h3>
                <div className="space-y-4">
                  {searchHistory.map((search, index) => (
                    <div key={search.search_id} className="border border-gray-600 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-gray-300 text-sm">
                          {new Date(search.timestamp).toLocaleString()}
                        </span>
                        <span className="text-gray-400 text-sm">
                          {search.duration?.toFixed(1)}s recording
                        </span>
                      </div>

                      <AudioPlayback
                        playbackUrl={`http://localhost:5000${search.playback_url}`}
                        title={`Recording ${index + 1}`}
                        searchId={search.search_id}
                        timestamp={search.timestamp}
                        duration={search.duration}
                      />

                      {search.bestMatch && (
                        <div className="mt-3 p-3 bg-gray-700/50 rounded">
                          <p className="text-white text-sm">
                            <strong>Best match:</strong> {search.bestMatch.title} by {search.bestMatch.artist}
                          </p>
                          <p className="text-gray-400 text-xs">
                            Confidence: {search.confidence?.toFixed(1)}%
                          </p>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RecordingTab;
