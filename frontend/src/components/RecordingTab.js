import React from 'react';
import { Search, Mic, Square, Play, Trash2, Loader2, Music, Youtube } from 'lucide-react';

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
  const clearAll = () => {
    clearRecording();
    setMatchedSong(null);
  };

  return (
    <div className="bg-gray-800/50 backdrop-blur-md rounded-xl p-8 border border-gray-700">
      <h2 className="text-2xl font-bold text-white mb-6">Find Song by Recording</h2>

      <div className="space-y-6">
        <div className="text-center">
          <p className="text-gray-300 mb-6">
            Record a snippet of music to identify the song (like Shazam!)
          </p>

          {/* Recording Controls */}
          <div className="bg-gray-700/50 rounded-xl p-8 border border-gray-600">
            {!audioBlob ? (
              <div>
                {!isRecording ? (
                  <button
                    onClick={startRecording}
                    disabled={isLoading}
                    className="bg-red-600 hover:bg-red-700 text-white font-bold py-6 px-12 rounded-full transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-4 mx-auto text-lg"
                  >
                    <Mic className="w-8 h-8" />
                    Start Recording
                  </button>
                ) : (
                  <div className="space-y-6">
                    <div className="flex items-center justify-center gap-6">
                      <div className="animate-pulse">
                        <Mic className="w-12 h-12 text-red-400" />
                      </div>
                      <span className="text-3xl font-bold text-white">
                        {recordingTime}s
                      </span>
                    </div>
                    <button
                      onClick={stopRecording}
                      className="bg-gray-600 hover:bg-gray-700 text-white font-bold py-4 px-8 rounded-lg transition-all duration-200 flex items-center gap-3 mx-auto text-lg"
                    >
                      <Square className="w-6 h-6" />
                      Stop Recording
                    </button>
                    <p className="text-gray-400">
                      Recording will auto-stop at 10 seconds
                    </p>
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-6">
                <div className="flex items-center justify-center gap-6">
                  <Play className="w-8 h-8 text-green-400" />
                  <span className="text-white font-medium text-xl">
                    Recording ready ({recordingTime}s)
                  </span>
                </div>
                <div className="flex gap-4 justify-center">
                  <button
                    onClick={handleAudioSearch}
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

          {/* Result Display */}
          {matchedSong && (
            <div className="mt-8">
              <div className={`bg-gray-700/50 rounded-xl p-6 border ${
                confidence > 70 ? 'border-green-600' :
                confidence > 50 ? 'border-yellow-600' :
                'border-red-600'
              }`}>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-bold text-white">Match Found!</h3>
                  <span className={`text-lg font-semibold ${
                    confidence > 70 ? 'text-green-400' :
                    confidence > 50 ? 'text-yellow-400' :
                    'text-red-400'
                  }`}>
                    {confidence}% Confidence
                  </span>
                </div>

                <div className="bg-gray-800/50 rounded-lg p-4">
                  <div className="flex items-center gap-4">
                    <Music className="w-12 h-12 text-gray-400" />
                    <div className="flex-1 text-left">
                      <h4 className="text-xl font-semibold text-white">{matchedSong.title}</h4>
                      <p className="text-gray-300">{matchedSong.artist}</p>
                      <p className="text-sm text-gray-400 mt-1">
                        Source: {matchedSong.source}
                      </p>
                    </div>
                    {matchedSong.youtube_id && (
                      <a
                        href={`https://www.youtube.com/watch?v=${matchedSong.youtube_id}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg transition-colors flex items-center gap-2"
                      >
                        <Youtube className="w-5 h-5" />
                        Play
                      </a>
                    )}
                  </div>
                </div>

                {confidence < 50 && (
                  <p className="text-yellow-400 text-sm mt-4">
                    Low confidence match. Try recording a clearer or different part of the song.
                  </p>
                )}
              </div>
            </div>
          )}

          {/* No Match Message */}
          {matchedSong === false && (
            <div className="mt-8 bg-gray-700/50 rounded-xl p-6 border border-gray-600">
              <p className="text-gray-300">
                No matching song found. Try recording a different or clearer part of the song.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RecordingTab;
