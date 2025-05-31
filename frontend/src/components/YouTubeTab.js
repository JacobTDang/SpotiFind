import React from 'react';
import { Youtube, Loader2 } from 'lucide-react';

const YouTubeTab = ({
  youtubeUrl,
  isLoading,
  setYoutubeUrl,
  handleYouTubeAdd
}) => {
  return (
    <div className="bg-gray-800/50 backdrop-blur-md rounded-xl p-8 border border-gray-700">
      <h2 className="text-2xl font-bold text-white mb-6">Add YouTube Video</h2>
      <div className="space-y-6">
        <div>
          <label className="block text-gray-300 font-medium mb-2">YouTube URL</label>
          <input
            type="url"
            value={youtubeUrl}
            onChange={(e) => setYoutubeUrl(e.target.value)}
            className="w-full p-3 rounded-lg bg-gray-700/50 border border-gray-600 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-white focus:border-transparent"
            placeholder="https://www.youtube.com/watch?v=..."
            required
          />
          <p className="text-sm text-gray-400 mt-1">Paste any YouTube video URL - audio will be extracted and analyzed</p>
        </div>

        <button
          onClick={handleYouTubeAdd}
          disabled={isLoading}
          className="w-full bg-white text-black font-bold py-4 px-6 rounded-lg hover:bg-gray-100 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Processing...
            </>
          ) : (
            <>
              <Youtube className="w-5 h-5" />
              Add from YouTube
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default YouTubeTab;
