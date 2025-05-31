import React from 'react';

const MessageDisplay = ({ message, messageType }) => {
  if (!message) return null;

  return (
    <div className={`mb-6 p-4 rounded-lg text-center font-medium ${
      messageType === 'success'
        ? 'bg-green-900/50 text-green-300 border border-green-700'
        : 'bg-red-900/50 text-red-300 border border-red-700'
    }`}>
      {message}
    </div>
  );
};

export default MessageDisplay;
