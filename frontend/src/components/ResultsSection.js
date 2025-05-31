import React from 'react';
import { ExternalLink } from 'lucide-react';

const ResultsSection = ({ results }) => {
  if (results.length === 0) return null;

  return (
    <div className="max-w-4xl mx-auto mt-8">
      <div className="bg-gray-800/50 backdrop-blur-md rounded-xl p-8 border border-gray-700">
        <h2 className="text-2xl font-bold text-white mb-6">Similar Songs</h2>
        <div className="grid gap-4">
          {results.map((song, index) => (
            <div
              key={song.songID}
              className="bg-gray-700/50 rounded-lg p-4 border border-gray-600 hover:bg-gray-600/50 transition-all duration-200"
            >
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-white">{song.title}</h3>
                  <p className="text-gray-300">{song.artist}</p>
                  <div className="flex items-center gap-4 mt-2">
                    <span className="text-sm text-gray-400">
                      Similarity: {(1 - song.distance).toFixed(3)}
                    </span>
                    <span className="text-sm text-gray-400 capitalize">
                      Source: {song.source}
                    </span>
                  </div>
                </div>
                {song.youtube_id && (
                  <a
                    href={`https://www.youtube.com/watch?v=${song.youtube_id}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-gray-400 hover:text-white transition-colors"
                  >
                    <ExternalLink className="w-5 h-5" />
                  </a>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ResultsSection;
