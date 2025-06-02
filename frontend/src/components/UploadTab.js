import React from 'react';
import { Upload, Loader2 } from 'lucide-react';

const UploadTab = ({
  uploadFile,
  uploadTitle,
  uploadArtist,
  isLoading,
  setUploadTitle,
  setUploadArtist,
  handleFileChange,
  handleFileUpload
}) => {
  return (
    <div className="bg-gray-800/50 backdrop-blur-md rounded-xl p-8 border border-gray-700">
      <h2 className="text-2xl font-bold text-white mb-6">Upload Audio File</h2>
      <div className="space-y-6">
        <div>
          <label className="block text-gray-300 font-medium mb-2">Audio File</label>
          <input
            id="file-input"
            type="file"
            accept="audio/*"
            onChange={handleFileChange}
            className="w-full p-3 rounded-lg bg-gray-700/50 border border-gray-600 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-white focus:border-transparent"
            required
          />
          <p className="text-sm text-gray-400 mt-1">Supported: MP3, WAV, M4A, OGG (max 50MB)</p>
        </div>

        <div>
          <label className="block text-gray-300 font-medium mb-2">Title</label>
          <input
            type="text"
            value={uploadTitle}
            onChange={(e) => setUploadTitle(e.target.value)}
            className="w-full p-3 rounded-lg bg-gray-700/50 border border-gray-600 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-white focus:border-transparent"
            placeholder="Song title"
            required
          />
        </div>

        <div>
          <label className="block text-gray-300 font-medium mb-2">Artist</label>
          <input
            type="text"
            value={uploadArtist}
            onChange={(e) => setUploadArtist(e.target.value)}
            className="w-full p-3 rounded-lg bg-gray-700/50 border border-gray-600 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-white focus:border-transparent"
            placeholder="Artist name"
            required
          />
        </div>

        <button
          onClick={handleFileUpload}
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
              <Upload className="w-5 h-5" />
              Upload & Analyze
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default UploadTab;
