import React, { useState } from 'react';
import { Youtube, Loader2, Music, CheckCircle, XCircle } from 'lucide-react';

const YouTubeTab = ({
  isLoading,
  setIsLoading,
  showMessage
}) => {
  // Local state for this component
  const [inputType, setInputType] = useState('video'); // 'video' or 'playlist'
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [maxVideos, setMaxVideos] = useState(20);
  const [playlistResults, setPlaylistResults] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  // Helper function to clean YouTube URLs
  const cleanYouTubeUrl = (url) => {
    // If it's a single video request, remove playlist parameters
    if (inputType === 'video') {
      try {
        const urlObj = new URL(url);
        // Remove list parameter for single video URLs
        urlObj.searchParams.delete('list');
        urlObj.searchParams.delete('index');
        return urlObj.toString();
      } catch (e) {
        return url; // Return original if URL parsing fails
      }
    }
    return url;
  };

  const handleYouTubeAdd = async () => {
    if (!youtubeUrl) {
      showMessage('Please enter a YouTube URL', 'error');
      return;
    }

    setIsLoading(true);

    // Clean the URL for single videos
    const cleanedUrl = cleanYouTubeUrl(youtubeUrl);

    try {
      const response = await fetch('http://localhost:5000/youtube', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: cleanedUrl }),
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

  const handlePlaylistAdd = async () => {
    if (!youtubeUrl) {
      showMessage('Please enter a YouTube playlist URL', 'error');
      return;
    }

    setIsLoading(true);
    setIsProcessing(true);
    setPlaylistResults(null);

    try {
      const response = await fetch('http://localhost:5000/youtube-playlist', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          url: youtubeUrl,
          max_videos: maxVideos
        }),
      });

      const data = await response.json();

      if (data.success) {
        setPlaylistResults(data);
        showMessage(
          `Playlist processed! ${data.successful_count}/${data.total_videos} songs added successfully`,
          'success'
        );
        setYoutubeUrl('');
      } else {
        showMessage(data.message || 'Failed to process playlist', 'error');
      }
    } catch (error) {
      showMessage('Network error during playlist processing', 'error');
    } finally {
      setIsLoading(false);
      setIsProcessing(false);
    }
  };

  const handleSubmit = () => {
    // Auto-detect URL type and suggest switching if needed
    const isPlaylistUrl = youtubeUrl.includes('/playlist') || youtubeUrl.includes('list=');
    const isVideoOnlyUrl = (youtubeUrl.includes('/watch') || youtubeUrl.includes('youtu.be/')) && !youtubeUrl.includes('list=');

    // Only show warnings for clear mismatches
    if (inputType === 'video' && youtubeUrl.includes('/playlist')) {
      showMessage('This is a playlist URL. Switch to "Playlist" mode to process multiple videos.', 'error');
      return;
    }

    if (inputType === 'playlist' && isVideoOnlyUrl) {
      showMessage('This looks like a single video URL without a playlist. Switch to "Single Video" mode.', 'error');
      return;
    }

    if (inputType === 'video') {
      handleYouTubeAdd();
    } else {
      handlePlaylistAdd();
    }
  };

  return (
    <div className="bg-gray-800/50 backdrop-blur-md rounded-xl p-8 border border-gray-700">
      <h2 className="text-2xl font-bold text-white mb-6">Add from YouTube</h2>

      {/* Toggle between video and playlist */}
      <div className="flex gap-4 mb-6">
        <button
          onClick={() => {
            setInputType('video');
            setPlaylistResults(null);
          }}
          className={`flex-1 py-2 px-4 rounded-lg font-medium transition-all duration-200 ${
            inputType === 'video'
              ? 'bg-white text-black'
              : 'bg-gray-700/50 text-gray-300 hover:bg-gray-600/50'
          }`}
        >
          Single Video
        </button>
        <button
          onClick={() => {
            setInputType('playlist');
            setPlaylistResults(null);
          }}
          className={`flex-1 py-2 px-4 rounded-lg font-medium transition-all duration-200 ${
            inputType === 'playlist'
              ? 'bg-white text-black'
              : 'bg-gray-700/50 text-gray-300 hover:bg-gray-600/50'
          }`}
        >
          Playlist
        </button>
      </div>

      <div className="space-y-6">
        <div>
          <label className="block text-gray-300 font-medium mb-2">
            {inputType === 'video' ? 'YouTube Video URL' : 'YouTube Playlist URL'}
          </label>
          <input
            type="url"
            value={youtubeUrl}
            onChange={(e) => setYoutubeUrl(e.target.value)}
            className="w-full p-3 rounded-lg bg-gray-700/50 border border-gray-600 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-white focus:border-transparent"
            placeholder={inputType === 'video'
              ? "https://www.youtube.com/watch?v=..."
              : "https://www.youtube.com/playlist?list=..."}
            required
          />
          <p className="text-sm text-gray-400 mt-1">
            {inputType === 'video'
              ? 'Paste any YouTube video URL - audio will be extracted and analyzed'
              : 'Paste a YouTube playlist URL to process multiple videos at once'}
          </p>
          {inputType === 'video' && youtubeUrl.includes('&list=') && (
            <p className="text-sm text-yellow-400 mt-1">
              Note: This URL contains playlist info. Only the single video will be processed.
            </p>
          )}
        </div>

        {/* Max videos input - only show for playlist */}
        {inputType === 'playlist' && (
          <div>
            <label className="block text-gray-300 font-medium mb-2">Max Videos to Process</label>
            <input
              type="number"
              value={maxVideos}
              onChange={(e) => setMaxVideos(Math.max(1, Math.min(100, parseInt(e.target.value) || 20)))}
              className="w-full p-3 rounded-lg bg-gray-700/50 border border-gray-600 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-white focus:border-transparent"
              min="1"
              max="100"
            />
            <p className="text-sm text-gray-400 mt-1">Limit: 1-100 videos (recommended: 20-50)</p>
          </div>
        )}

        <button
          onClick={handleSubmit}
          disabled={isLoading}
          className="w-full bg-white text-black font-bold py-4 px-6 rounded-lg hover:bg-gray-100 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              {inputType === 'video' ? 'Processing...' : 'Processing Playlist...'}
            </>
          ) : (
            <>
              {inputType === 'video' ? (
                <>
                  <Youtube className="w-5 h-5" />
                  Add Video
                </>
              ) : (
                <>
                  <Music className="w-5 h-5" />
                  Process Playlist
                </>
              )}
            </>
          )}
        </button>

        {/* Processing Progress - only for playlists */}
        {isProcessing && inputType === 'playlist' && (
          <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
            <div className="flex items-center gap-3">
              <Loader2 className="w-5 h-5 animate-spin text-blue-400" />
              <span className="text-white">Processing playlist... This may take several minutes.</span>
            </div>
          </div>
        )}

        {/* Playlist Results Display */}
        {playlistResults && inputType === 'playlist' && (
          <div className="bg-gray-700/50 rounded-lg p-6 border border-gray-600">
            <h3 className="text-xl font-bold text-white mb-4">Playlist Results</h3>

            {/* Summary */}
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

            {/* Successful Songs */}
            {playlistResults.successful_songs && playlistResults.successful_songs.length > 0 && (
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

            {/* Failed Songs */}
            {playlistResults.failed_songs && playlistResults.failed_songs.length > 0 && (
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

export default YouTubeTab;
