import React from 'react';

function SpeakerTimeline({ segments }) {
  return (
    <div className="bg-white p-4 rounded-lg shadow-lg overflow-x-auto whitespace-nowrap">
      <h4 className="mb-2 text-xl font-semibold text-indigo-600">Speaker Timeline</h4>
      <div className="flex space-x-4">
        {segments?.map((seg, i) => (
          <div key={i} className="bg-indigo-100 hover:bg-indigo-200 p-2 rounded-lg cursor-pointer transition hover:scale-105">
            <p className="text-sm font-semibold">[{seg.start_time.toFixed(1)}s - {seg.end_time.toFixed(1)}s]</p>
            <p className="font-medium">Speaker {seg.speaker}</p>
            <p className="text-xs text-gray-600">Emotion: {seg.final_emotion}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default SpeakerTimeline;
