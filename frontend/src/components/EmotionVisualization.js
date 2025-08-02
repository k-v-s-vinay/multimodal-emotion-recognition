import React from 'react';

function EmotionVisualization({ summary }) {
  return (
    <div className="bg-gray-100 p-4 rounded-lg shadow-lg">
      <h4 className="text-xl font-semibold mb-2 text-indigo-600">Emotion Distribution</h4>
      <div className="flex space-x-4 justify-center my-4">
        {Object.entries(summary?.overall_emotion_distribution || {}).map(([emotion, value]) => (
          <div key={emotion} className="text-center">
            <div
              className={`w-12 h-12 mx-auto rounded-full bg-gradient-to-r from-indigo-400 to-indigo-600`}
              style={{ clipPath: 'circle(50% at 50% 50%)' }}
            >
              <div
                className="flex items-center justify-center h-full text-white font-bold text-lg"
                style={{ transform: `rotate(-${(Object.keys(summary.overall_emotion_distribution).indexOf(emotion) * (360 / Object.keys(summary.overall_emotion_distribution).length))}deg)` }}
              >
                {emotion.charAt(0).toUpperCase()}
              </div>
            </div>
            <p className="mt-2 text-sm">{emotion}</p>
            <p className="font-semibold">{(value * 100).toFixed(1)}%</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default EmotionVisualization;
