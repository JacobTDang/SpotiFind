import React, { useState } from 'react';
import { Music, Loader2, CheckCircle, XCircle } from 'lucide-react';

//  PROPER PROP DESTRUCTURING - Accept setIsLoading as a prop
const PlaylistTab = ({ isLoading, setIsLoading, showMessage }) => {
  //  LOCAL STATE for playlist-specific data
  const [playlistUrl, setPlaylistUrl] = useState('');
  const [maxVideos, setMaxVideos] = useState(20);
  const [playlistResults, setPlaylistResults] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  // ✅ ASYNC/AWAIT PATTERN with proper error handling
  const handlePlaylistAdd = async () => {
    // Input validation
    if (!playlistUrl) {
      showMessage('Please enter a YouTube playlist URL', 'error');
      return;
    }

    //  STATE MANAGEMENT PATTERN - Set loading states
    setIsLoading(true);        // ← Parent component loading state
    setIsProcessing(true);     // ← Local component processing state
    setPlaylistResults(null);  // ← Clear previous results

    try {
      // FETCH API PATTERN with proper configuration
      const response = await fetch('http://localhost:5000/youtube-playlist', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          url: playlistUrl,
          max_videos: maxVideos
        }),
      });

      //  RESPONSE HANDLING PATTERN
      const data = await response.json();

      if (data.success) {
        //  SUCCESS STATE UPDATES
        setPlaylistResults(data);
        showMessage(
          `Playlist processed! ${data.successful_count}/${data.total_videos} songs added successfully`,
          'success'
        );
        setPlaylistUrl(''); // Clear form on success
      } else {
        //  ERROR HANDLING - Show server error message
        showMessage(data.message || 'Failed to process playlist', 'error');
      }
    } catch (error) {
      //  NETWORK ERROR HANDLING
      console.error('Playlist processing error:', error);
      showMessage('Network error during playlist processing', 'error');
    } finally {
      // CLEANUP PATTERN - Always runs regardless of success/failure
      setIsLoading(false);
      setIsProcessing(false);
    }
  };

  return (
    <div className="bg-gray-800/50 backdrop-blur-md rounded-xl p-8 border border-gray-700">
      <h2 className="text-2xl font-bold text-white mb-6">Add YouTube Playlist</h2>

      <div className="space-y-6">
        {/*  CONTROLLED COMPONENT PATTERN */}
        <div>
          <label className="block text-gray-300 font-medium mb-2">Playlist URL</label>
          <input
            type="url"
            value={playlistUrl}
            onChange={(e) => setPlaylistUrl(e.target.value)}
            className="w-full p-3 rounded-lg bg-gray-700/50 border border-gray-600 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-white focus:border-transparent"
            placeholder="https://www.youtube.com/playlist?list=..."
            required
          />
          <p className="text-sm text-gray-400 mt-1">Paste a YouTube playlist URL to process multiple videos at once</p>
        </div>

        {/*  NUMBER INPUT WITH VALIDATION */}
        <div>
          <label className="block text-gray-300 font-medium mb-2">Max Videos to Process</label>
          <input
            type="number"
            value={maxVideos}
            onChange={(e) => {
              //  INPUT VALIDATION PATTERN - Clamp values
              const value = Math.max(1, Math.min(100, parseInt(e.target.value) || 20));
              setMaxVideos(value);
            }}
            className="w-full p-3 rounded-lg bg-gray-700/50 border border-gray-600 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-white focus:border-transparent"
            min="1"
            max="100"
          />
          <p className="text-sm text-gray-400 mt-1">Limit: 1-100 videos (recommended: 20-50)</p>
        </div>

        {/*  BUTTON STATE MANAGEMENT - Disabled during loading */}
        <button
          onClick={handlePlaylistAdd}
          disabled={isLoading}
          className="w-full bg-white text-black font-bold py-4 px-6 rounded-lg hover:bg-gray-100 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Processing Playlist...
            </>
          ) : (
            <>
              <Music className="w-5 h-5" />
              Process Playlist
            </>
          )}
        </button>

        {/*  CONDITIONAL RENDERING PATTERN - Processing indicator */}
        {isProcessing && (
          <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
            <div className="flex items-center gap-3">
              <Loader2 className="w-5 h-5 animate-spin text-blue-400" />
              <span className="text-white">Processing playlist... This may take several minutes.</span>
            </div>
          </div>
        )}

        {/*  RESULTS DISPLAY with NULL CHECKING */}
        {playlistResults && (
          <div className="bg-gray-700/50 rounded-lg p-6 border border-gray-600">
            <h3 className="text-xl font-bold text-white mb-4">Playlist Results</h3>

            {/* Summary Grid */}
            <div className="grid grid-cols-2 gap-4 mb-6">
              <div className="bg-green-900/30 p-4 rounded-lg border border-green-700">
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-green-400" />
                  <span className="text-green-300 font-medium">Successful</span>
                </div>
                <p className="text-2xl font-bold text-white">{playlistResults.successful_count}</p>
              </div>

              <div className="bg-red-900/30 p-4 rounded-lg border border-red-700">
                <div className="flex items-center gap-2">
                  <XCircle className="w-5 h-5 text-red-400" />
                  <span className="text-red-300 font-medium">Failed</span>
                </div>
                <p className="text-2xl font-bold text-white">{playlistResults.failed_count}</p>
              </div>
            </div>

            {/*  ARRAY CHECKING PATTERN - Prevent errors on undefined arrays */}
            {playlistResults.successful_songs?.length > 0 && (
              <div className="mb-4">
                <h4 className="text-lg font-medium text-green-300 mb-2">Successfully Added:</h4>
                <div className="max-h-40 overflow-y-auto space-y-2">
                  {playlistResults.successful_songs.map((song, index) => (
                    <div key={index} className="text-sm text-gray-300 bg-gray-800/50 p-2 rounded">
                      {song.song}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {playlistResults.failed_songs?.length > 0 && (
              <div>
                <h4 className="text-lg font-medium text-red-300 mb-2">Failed to Process:</h4>
                <div className="max-h-40 overflow-y-auto space-y-2">
                  {playlistResults.failed_songs.map((song, index) => (
                    <div key={index} className="text-sm text-gray-300 bg-gray-800/50 p-2 rounded">
                      <div className="font-medium">{song.song}</div>
                      <div className="text-gray-400">{song.message}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default PlaylistTab;
