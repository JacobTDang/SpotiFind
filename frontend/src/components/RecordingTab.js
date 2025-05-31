import React from 'react';
import { Search, Mic, Square, Play, Trash2, Loader2 } from 'lucide-react';

const RecordingTab = ({
  audioBlob,
  isRecording,
  recordingTime,
  isLoading,
  startRecording,
  stopRecording,
  clearRecording,
  handleAudioSearch
}) => {
  return (
    <div className="bg-gray-800/50 backdrop-blur-md rounded-xl p-8 border border-gray-700">
      <h2 className="text-2xl font-bold text-white mb-6">Find Similar Songs</h2>

      <div className="space-y-6">
        <div className="text-center">
          <p className="text-gray-300 mb-6">
            Record a snippet of music to find similar songs (like Shazam!)
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
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Search className="w-6 h-6" />
                        Find Similar
                      </>
                    )}
                  </button>
                  <button
                    onClick={clearRecording}
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
        </div>
      </div>
    </div>
  );
};

export default RecordingTab;
