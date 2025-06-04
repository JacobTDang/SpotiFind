import React from 'react';
import { Upload, Youtube, Search } from 'lucide-react';

const TabNavigation = ({ activeTab, setActiveTab }) => {
  return (
    <div className="flex justify-center mb-8">
      <div className="bg-gray-800/80 backdrop-blur-sm rounded-lg p-1 border border-gray-700 flex w-auto">
        <button
          onClick={() => setActiveTab('upload')}
          className={`px-8 py-3 rounded-md font-medium transition-all duration-200 flex items-center gap-2 flex-1 justify-center ${
            activeTab === 'upload'
              ? 'bg-white text-black shadow-lg'
              : 'text-gray-300 hover:bg-gray-700/50'
          }`}
        >
          <Upload className="w-4 h-4" />
          Upload Audio
        </button>

        <button
          onClick={() => setActiveTab('youtube')}
          className={`px-8 py-3 rounded-md font-medium transition-all duration-200 flex items-center gap-2 flex-1 justify-center ${
            activeTab === 'youtube'
              ? 'bg-white text-black shadow-lg'
              : 'text-gray-300 hover:bg-gray-700/50'
          }`}
        >
          <Youtube className="w-4 h-4" />
          Add YouTube
        </button>

        <button
          onClick={() => setActiveTab('search')}
          className={`px-8 py-3 rounded-md font-medium transition-all duration-200 flex items-center gap-2 flex-1 justify-center ${
            activeTab === 'search'
              ? 'bg-white text-black shadow-lg'
              : 'text-gray-300 hover:bg-gray-700/50'
          }`}
        >
          <Search className="w-4 h-4" />
          Find Similar
        </button>
      </div>
    </div>
  );
};

export default TabNavigation;
