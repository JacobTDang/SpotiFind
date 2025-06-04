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
        {/* Recording Tips */}
        <div className="bg-blue-900/30 p-4 rounded-lg border border-blue-700">
          <h3 className="text-blue-300 font-medium mb-2">For Best Results:</h3>
          <ul className="text-blue-200 text-sm space-y-1">
            <li>• Record close to speakers or headphones</li>
            <li>• Minimize background noise</li>
            <li>• Include vocals or prominent instruments</li>
            <li>• Record the chorus or most recognizable part</li>
            <li>• 15-20 seconds gives best results</li>
            <li>• Try different parts if first attempt fails</li>
          </ul>
        </div>

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
                      Recording will auto-stop at 20 seconds
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
                      {matchedSong.distance && (
                        <p className="text-xs text-gray-500 mt-1">
                          Distance: {matchedSong.distance.toFixed(3)}
                          {matchedSong.match_count && ` • Segments: ${matchedSong.embeddings_matched}/${matchedSong.match_count}`}
                        </p>
                      )}
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

                {/* Confidence level messages */}
                {confidence > 70 && (
                  <div className="mt-4 p-3 bg-green-900/30 border border-green-700 rounded-lg">
                    <p className="text-green-300 text-sm">
                      <strong>High confidence match!</strong> This is very likely the correct song.
                      {matchedSong.coverage && matchedSong.coverage > 0.7 && (
                        <span className="block mt-1">Multiple segments matched consistently.</span>
                      )}
                    </p>
                  </div>
                )}

                {confidence > 50 && confidence <= 70 && (
                  <div className="mt-4 p-3 bg-yellow-900/30 border border-yellow-700 rounded-lg">
                    <p className="text-yellow-300 text-sm">
                      <strong>Moderate confidence match.</strong> This might be the song you're looking for.
                      {matchedSong.coverage && matchedSong.coverage < 0.5 && (
                        <span className="block mt-1">Try recording a different part for better results.</span>
                      )}
                    </p>
                  </div>
                )}

                {confidence <= 50 && (
                  <div className="mt-4 p-3 bg-red-900/30 border border-red-700 rounded-lg">
                    <p className="text-red-300 text-sm">
                      <strong>Low confidence match.</strong> This might not be the correct song.
                    </p>
                    <div className="mt-2 text-red-200 text-xs">
                      <p>Suggestions:</p>
                      <ul className="list-disc list-inside mt-1 space-y-1">
                        <li>Record a clearer or different part (chorus works best)</li>
                        <li>Move closer to the audio source</li>
                        <li>Record for the full 20 seconds</li>
                        <li>Reduce background noise</li>
                      </ul>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* No Match Message */}
          {matchedSong === false && (
            <div className="mt-8 bg-gray-700/50 rounded-xl p-6 border border-gray-600">
              <div className="flex items-center justify-center gap-3 mb-3">
                <Search className="w-8 h-8 text-gray-400" />
                <h3 className="text-xl font-semibold text-gray-300">No Match Found</h3>
              </div>
              <p className="text-gray-300 mb-4">
                No matching song found in our database.
              </p>
              <div className="bg-blue-900/30 p-4 rounded-lg border border-blue-700">
                <p className="text-blue-200 text-sm mb-2"><strong>Try these tips:</strong></p>
                <ul className="text-blue-200 text-sm space-y-1">
                  <li>• Record a different part of the song (chorus works best)</li>
                  <li>• Move closer to the audio source</li>
                  <li>• Record for the full 20 seconds to capture more segments</li>
                  <li>• Reduce background noise and echoes</li>
                  <li>• Make sure the song is loud enough</li>
                  <li>• Verify the song exists in our database</li>
                </ul>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RecordingTab;
