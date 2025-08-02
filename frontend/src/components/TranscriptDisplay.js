import React from 'react';

function TranscriptDisplay({ segments }) {
  return (
    <div className="bg-white p-4 rounded-lg shadow-lg max-h-96 overflow-y-auto">
      <h4 className="mb-2 text-xl font-semibold text-indigo-600">Transcript</h4>
      {segments?.map((seg, i) => (
        <div key={i} className="mb-4 p-2 border border-gray-200 rounded hover:bg-gray-50 cursor-pointer transition hover:scale-105">
          <p><b>Speaker {seg.speaker}:</b> {seg.text}</p>
          <p className="text-sm text-gray-600">Emotion: {seg.final_emotion}</p>
        </div>
      ))}
    </div>
  );
}

export default TranscriptDisplay;
