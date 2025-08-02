// import React, { useState } from 'react';
// import EmotionVisualization from './EmotionVisualization';
// import SpeakerTimeline from './SpeakerTimeline';
// import TranscriptDisplay from './TranscriptDisplay';

// const Dashboard = ({ analysisResult, onNewAnalysis }) => {
//   const [activeTab, setActiveTab] = useState('overview');

//   if (!analysisResult || !analysisResult.segments) {
//     return (
//       <div className="text-center py-8">
//         <p className="text-gray-600">No analysis results to display</p>
//         <button
//           onClick={onNewAnalysis}
//           className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
//         >
//           Upload New File
//         </button>
//       </div>
//     );
//   }

//   const { summary, segments } = analysisResult;

//   const tabs = [
//     { id: 'overview', label: 'Overview', icon: '📊' },
//     { id: 'timeline', label: 'Timeline', icon: '⏰' },
//     { id: 'transcript', label: 'Transcript', icon: '📝' },
//     { id: 'emotions', label: 'Emotions', icon: '😊' }
//   ];

//   return (
//     <div className="max-w-7xl mx-auto">
//       {/* Header */}
//       <div className="bg-white rounded-lg shadow-md p-6 mb-6">
//         <div className="flex justify-between items-center">
//           <div>
//             <h2 className="text-2xl font-bold text-gray-900 mb-2">
//               Analysis Results
//             </h2>
//             <p className="text-gray-600">
//               File: {analysisResult.filename} | 
//               Speakers: {summary.total_speakers || 0} | 
//               Duration: {Math.round(summary.conversation_duration || 0)}s
//             </p>
//           </div>
//           <button
//             onClick={onNewAnalysis}
//             className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700 transition-colors"
//           >
//             New Analysis
//           </button>
//         </div>
//       </div>

//       {/* Tabs */}
//       <div className="bg-white rounded-lg shadow-md">
//         <div className="border-b border-gray-200">
//           <nav className="flex space-x-8 px-6">
//             {tabs.map((tab) => (
//               <button
//                 key={tab.id}
//                 onClick={() => setActiveTab(tab.id)}
//                 className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
//                   activeTab === tab.id
//                     ? 'border-blue-500 text-blue-600'
//                     : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
//                 }`}
//               >
//                 <span className="mr-2">{tab.icon}</span>
//                 {tab.label}
//               </button>
//             ))}
//           </nav>
//         </div>

//         <div className="p-6">
//           {activeTab === 'overview' && (
//             <div className="space-y-6">
//               {/* Summary Cards */}
//               <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
//                 <div className="bg-blue-50 rounded-lg p-6">
//                   <h3 className="text-lg font-semibold text-blue-900 mb-2">
//                     Overall Emotion
//                   </h3>
//                   <p className="text-3xl font-bold text-blue-600">
//                     {summary.dominant_conversation_emotion || 'Unknown'}
//                   </p>
//                 </div>
                
//                 <div className="bg-green-50 rounded-lg p-6">
//                   <h3 className="text-lg font-semibold text-green-900 mb-2">
//                     Total Speakers
//                   </h3>
//                   <p className="text-3xl font-bold text-green-600">
//                     {summary.total_speakers || 0}
//                   </p>
//                 </div>
                
//                 <div className="bg-purple-50 rounded-lg p-6">
//                   <h3 className="text-lg font-semibold text-purple-900 mb-2">
//                     Segments Analyzed
//                   </h3>
//                   <p className="text-3xl font-bold text-purple-600">
//                     {segments.length}
//                   </p>
//                 </div>
//               </div>

//               {/* Speaker Emotions */}
//               {summary.speaker_summaries && (
//                 <div>
//                   <h3 className="text-xl font-semibold mb-4">Speaker Emotions</h3>
//                   <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
//                     {Object.entries(summary.speaker_summaries).map(([speaker, data]) => (
//                       <div key={speaker} className="border rounded-lg p-4">
//                         <h4 className="font-semibold text-lg mb-2">
//                           Speaker {speaker}
//                         </h4>
//                         <p className="text-gray-600 mb-2">
//                           Dominant Emotion: <span className="font-medium text-gray-900">
//                             {data.dominant_emotion}
//                           </span>
//                         </p>
//                         <p className="text-sm text-gray-500">
//                           {data.total_segments} segments
//                         </p>
//                       </div>
//                     ))}
//                   </div>
//                 </div>
//               )}
//             </div>
//           )}

//           {activeTab === 'timeline' && (
//             <SpeakerTimeline segments={segments} />
//           )}

//           {activeTab === 'transcript' && (
//             <TranscriptDisplay segments={segments} />
//           )}

//           {activeTab === 'emotions' && (
//             <EmotionVisualization 
//               summary={summary} 
//               segments={segments} 
//             />
//           )}
//         </div>
//       </div>
//     </div>
//   );
// };

// export default Dashboard;

// ======================================================================================


// import React from 'react';

// function Dashboard({ data, onNew }) {
//   return (
//     <div className="bg-white rounded-xl shadow-lg p-6 space-y-4 max-h-screen overflow-y-auto">
//       <div className="flex justify-between items-center border-b pb-2 mb-4">
//         <div>
//           <h3 className="text-2xl font-bold text-indigo-600">Analysis Summary</h3>
//           <p className="text-sm text-gray-500">File: {data.filename || 'testfile'}</p>
//         </div>
//         <button
//           className="bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 transition"
//           onClick={onNew}
//         >
//           Upload New File
//         </button>
//       </div>
//       <div className="grid md:grid-cols-3 gap-4 mb-4">
//         <div className="bg-indigo-50 p-4 rounded-lg shadow hover:scale-105 transition">
//           <h4 className="text-lg font-semibold text-indigo-700 mb-2">Overall Emotion</h4>
//           <p className="text-3xl text-center font-bold text-indigo-800">{(data.summary?.dominant_emotion || 'Neutral').toUpperCase()}</p>
//         </div>
//         <div className="bg-green-50 p-4 rounded-lg shadow hover:scale-105 transition">
//           <h4 className="text-lg font-semibold text-green-700 mb-2">Total Speakers</h4>
//           <p className="text-3xl text-center font-bold text-green-800">{data.summary?.total_speakers || 0}</p>
//         </div>
//         <div className="bg-purple-50 p-4 rounded-lg shadow hover:scale-105 transition">
//           <h4 className="text-lg font-semibold text-purple-700 mb-2">Segments Analyzed</h4>
//           <p className="text-3xl text-center font-bold text-purple-800">{data.segments?.length || 0}</p>
//         </div>
//       </div>
//       {/* Further sections: Timelines, Emotions, Transcripts */}
//       {/* Placeholder for detailed visualizations */}
//       <div className="space-y-4">
//         <h4 className="text-xl font-semibold text-indigo-600 mb-2">Speaker Emotions</h4>
//         {Object.entries(data.summary?.speaker_summaries || {}).map(([speaker, sc]) => (
//           <div key={speaker} className="bg-gray-50 p-4 rounded-lg hover:scale-105 transition">
//             <h5 className="font-semibold">Speaker {speaker}</h5>
//             <p>Dominant Emotion: {sc.dominant_emotion}</p>
//             <p>Total Segments: {sc.total_segments}</p>
//           </div>
//         ))}
//       </div>
//       {/* Add further detailed charts or timeline components as needed */}
//     </div>
//   );
// }

// export default Dashboard;
// ======================================================================================


// src/components/Dashboard.js
// import React, { useState } from 'react';

// const Dashboard = ({ data, onNew }) => {
//   const [activeTab, setActiveTab] = useState('overview');

//   // Emotion to emoji mapping
//   const emotionEmojis = {
//     happy: '😊',
//     sad: '😢',
//     angry: '😠',
//     neutral: '😐',
//     fear: '😨',
//     surprise: '😲',
//     disgust: '🤢'
//   };

//   // Emotion to color mapping for timeline
//   const emotionColors = {
//     happy: 'bg-green-500',
//     sad: 'bg-blue-500',
//     angry: 'bg-red-500',
//     neutral: 'bg-gray-500',
//     fear: 'bg-purple-500',
//     surprise: 'bg-yellow-500',
//     disgust: 'bg-orange-500'
//   };

//   const formatTime = (seconds) => {
//     const mins = Math.floor(seconds / 60);
//     const secs = Math.floor(seconds % 60);
//     return `${mins}:${secs.toString().padStart(2, '0')}`;
//   };

//   const getEmoji = (emotion) => emotionEmojis[emotion] || '😐';
//   const getColor = (emotion) => emotionColors[emotion] || 'bg-gray-500';

//   const renderOverview = () => (
//     <div className="space-y-6">
//       {/* Summary Cards */}
//       <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
//         <div className="card text-center">
//           <h3 className="text-lg font-semibold text-gray-700 mb-2">Overall Emotion</h3>
//           <div className="text-4xl mb-2">
//             {getEmoji(data.summary?.dominant_conversation_emotion)}
//           </div>
//           <p className="text-2xl font-bold text-indigo-600 uppercase">
//             {data.summary?.dominant_conversation_emotion || 'NEUTRAL'}
//           </p>
//         </div>

//         <div className="card text-center">
//           <h3 className="text-lg font-semibold text-green-700 mb-2">Total Speakers</h3>
//           <p className="text-4xl font-bold text-green-600">
//             {data.summary?.total_speakers || 0}
//           </p>
//         </div>

//         <div className="card text-center">
//           <h3 className="text-lg font-semibold text-purple-700 mb-2">Segments Analyzed</h3>
//           <p className="text-4xl font-bold text-purple-600">
//             {data.segments?.length || 0}
//           </p>
//         </div>
//       </div>

//       {/* Emotion Distribution Chart */}
//       <div className="card">
//         <h3 className="text-xl font-semibold mb-4">Conversation Emotion Breakdown</h3>
//         <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
//           {data.summary?.overall_emotion_distribution && 
//             Object.entries(data.summary.overall_emotion_distribution).map(([emotion, percentage]) => (
//               <div key={emotion} className="text-center p-4 rounded-lg bg-gray-50">
//                 <div className="text-3xl mb-2">{getEmoji(emotion)}</div>
//                 <div className="font-semibold capitalize text-gray-700">{emotion}</div>
//                 <div className="text-2xl font-bold text-indigo-600">
//                   {Math.round(percentage * 100)}%
//                 </div>
//               </div>
//             ))
//           }
//         </div>
//       </div>

//       {/* Speaker Emotions */}
//       <div className="card">
//         <h3 className="text-xl font-semibold mb-4">Speaker Emotions</h3>
//         <div className="space-y-4">
//           {data.summary?.speaker_summaries && 
//             Object.entries(data.summary.speaker_summaries).map(([speaker, info]) => (
//               <div key={speaker} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
//                 <div className="flex items-center space-x-4">
//                   <div className="text-3xl">{getEmoji(info.dominant_emotion)}</div>
//                   <div>
//                     <h4 className="font-semibold text-lg">{speaker}</h4>
//                     <p className="text-gray-600 capitalize">
//                       Dominant Emotion: <span className="font-medium">{info.dominant_emotion}</span>
//                     </p>
//                   </div>
//                 </div>
//                 <div className="text-right">
//                   <p className="text-sm text-gray-500">Total Segments</p>
//                   <p className="text-2xl font-bold text-indigo-600">{info.total_segments}</p>
//                 </div>
//               </div>
//             ))
//           }
//         </div>
//       </div>
//     </div>
//   );

//   const renderTimeline = () => (
//     <div className="space-y-6">
//       <div className="card">
//         <h3 className="text-xl font-semibold mb-6">Conversation Timeline</h3>
        
//         {/* Timeline Header */}
//         <div className="mb-6">
//           <h4 className="text-lg font-medium mb-3">Emotion Legend:</h4>
//           <div className="flex flex-wrap gap-3">
//             {Object.entries(emotionEmojis).map(([emotion, emoji]) => (
//               <div key={emotion} className="flex items-center space-x-2 bg-gray-100 px-3 py-1 rounded-full">
//                 <span className="text-lg">{emoji}</span>
//                 <span className="capitalize text-sm font-medium">{emotion}</span>
//               </div>
//             ))}
//           </div>
//         </div>

//         {/* Visual Timeline */}
//         <div className="space-y-4">
//           {data.segments && data.segments.map((segment, index) => (
//             <div key={index} className="relative">
//               {/* Timeline Bar */}
//               <div className="flex items-center space-x-4 p-4 bg-white border-l-4 border-indigo-500 shadow-sm rounded-r-lg">
                
//                 {/* Time & Emoji */}
//                 <div className="flex-shrink-0 text-center">
//                   <div className="text-2xl mb-1">{getEmoji(segment.final_emotion)}</div>
//                   <div className="text-xs font-mono text-gray-500">
//                     {formatTime(segment.start_time)}
//                   </div>
//                   <div className="text-xs text-gray-400">
//                     {formatTime(segment.end_time)}
//                   </div>
//                 </div>

//                 {/* Speaker & Content */}
//                 <div className="flex-grow">
//                   <div className="flex items-center space-x-3 mb-2">
//                     <span className="bg-indigo-100 text-indigo-800 px-3 py-1 rounded-full text-sm font-medium">
//                       {segment.speaker}
//                     </span>
//                     <span className={`px-2 py-1 rounded-full text-xs font-medium text-white ${getColor(segment.final_emotion)}`}>
//                       {segment.final_emotion.toUpperCase()}
//                     </span>
//                     <span className="text-xs text-gray-500">
//                       {Math.round(segment.confidence * 100)}% confidence
//                     </span>
//                   </div>
                  
//                   <p className="text-gray-800 bg-gray-50 p-3 rounded-lg">
//                     "{segment.text || 'No transcript available'}"
//                   </p>

//                   {/* Emotion Breakdown */}
//                   <div className="mt-3 flex flex-wrap gap-2">
//                     {segment.audio_emotions && Object.entries(segment.audio_emotions)
//                       .filter(([_, score]) => score > 0.1)
//                       .sort(([_, a], [__, b]) => b - a)
//                       .slice(0, 3)
//                       .map(([emotion, score]) => (
//                         <span key={emotion} className="text-xs bg-gray-200 px-2 py-1 rounded">
//                           {getEmoji(emotion)} {emotion} ({Math.round(score * 100)}%)
//                         </span>
//                       ))
//                     }
//                   </div>
//                 </div>

//                 {/* Duration Bar */}
//                 <div className="flex-shrink-0 w-20">
//                   <div className="text-xs text-gray-500 mb-1">Duration</div>
//                   <div className="h-2 bg-gray-200 rounded-full">
//                     <div 
//                       className={`h-2 rounded-full ${getColor(segment.final_emotion)}`}
//                       style={{width: `${Math.min((segment.end_time - segment.start_time) * 10, 100)}%`}}
//                     ></div>
//                   </div>
//                   <div className="text-xs text-gray-400 mt-1">
//                     {Math.round(segment.end_time - segment.start_time)}s
//                   </div>
//                 </div>
//               </div>

//               {/* Connector Line */}
//               {index < data.segments.length - 1 && (
//                 <div className="w-px h-4 bg-gray-300 ml-8 my-2"></div>
//               )}
//             </div>
//           ))}
//         </div>
//       </div>
//     </div>
//   );

//   const renderTranscript = () => (
//     <div className="card">
//       <div className="flex justify-between items-center mb-6">
//         <h3 className="text-xl font-semibold">Full Transcript with Emotions</h3>
//         <button className="btn-secondary text-sm">
//           📄 Export Transcript
//         </button>
//       </div>
  
  

//       <div className="space-y-4 max-h-96 overflow-y-auto">
//         {data.segments && data.segments.map((segment, index) => (
//           <div key={index} className={`p-4 rounded-lg border-l-4 ${
//             segment.speaker === 'Speaker_1' ? 'bg-blue-50 border-blue-500' : 'bg-green-50 border-green-500'
//           }`}>
            
//             <div className="flex items-center justify-between mb-2">
//               <div className="flex items-center space-x-3">
//                 <span className="font-semibold text-gray-800">{segment.speaker}</span>
//                 <span className="text-2xl">{getEmoji(segment.final_emotion)}</span>
//                 <span className={`px-2 py-1 rounded text-xs font-medium text-white ${getColor(segment.final_emotion)}`}>
//                   {segment.final_emotion}
//                 </span>
//               </div>
//               <span className="text-sm text-gray-500">
//                 {formatTime(segment.start_time)} - {formatTime(segment.end_time)}
//               </span>
//             </div>

//             <p className="text-gray-800 leading-relaxed">
//               "{segment.text || 'No transcript available'}"
//             </p>

//             <div className="mt-2 text-xs text-gray-500">
//               Confidence: {Math.round(segment.confidence * 100)}%
//             </div>
//           </div>
//         ))}
//       </div>
//     </div>
//   );

//   const renderInsights = () => (
//     <div className="space-y-6">
//       <div className="card">
//         <h3 className="text-xl font-semibold mb-4">🔍 Conversation Insights</h3>
        
//         <div className="grid md:grid-cols-2 gap-6">
//           {/* Emotional Patterns */}
//           <div className="bg-gradient-to-br from-blue-50 to-indigo-50 p-6 rounded-lg">
//             <h4 className="font-semibold mb-3 flex items-center">
//               📊 Emotional Patterns
//             </h4>
//             <ul className="space-y-2 text-sm">
//               <li className="flex items-center space-x-2">
//                 <span>🎯</span>
//                 <span>Most emotional speaker: <strong>
//                   {data.summary?.speaker_summaries && 
//                     Object.entries(data.summary.speaker_summaries)
//                       .reduce((prev, [speaker, info]) => 
//                         info.total_segments > (prev.segments || 0) ? {speaker, segments: info.total_segments} : prev, {}
//                       ).speaker
//                   }
//                 </strong></span>
//               </li>
//               <li className="flex items-center space-x-2">
//                 <span>⏱️</span>
//                 <span>Average segment duration: <strong>
//                   {data.segments && Math.round(
//                     data.segments.reduce((sum, seg) => sum + (seg.end_time - seg.start_time), 0) / data.segments.length
//                   )}s</strong>
//                 </span>
//               </li>
//               <li className="flex items-center space-x-2">
//                 <span>🗣️</span>
//                 <span>Total conversation time: <strong>
//                   {data.segments && Math.round(Math.max(...data.segments.map(seg => seg.end_time)))}s
//                 </strong></span>
//               </li>
//             </ul>
//           </div>

//           {/* Accessibility Features */}
//           <div className="bg-gradient-to-br from-green-50 to-emerald-50 p-6 rounded-lg">
//             <h4 className="font-semibold mb-3 flex items-center">
//               ♿ Accessibility Summary
//             </h4>
//             <ul className="space-y-2 text-sm">
//               <li className="flex items-center space-x-2">
//                 <span>👁️</span>
//                 <span>Visual timeline: <strong>Available</strong></span>
//               </li>
//               <li className="flex items-center space-x-2">
//                 <span>😊</span>
//                 <span>Emotion emojis: <strong>Enabled</strong></span>
//               </li>
//               <li className="flex items-center space-x-2">
//                 <span>🎨</span>
//                 <span>Color-coded speakers: <strong>Active</strong></span>
//               </li>
//               <li className="flex items-center space-x-2">
//                 <span>📱</span>
//                 <span>Mobile-friendly: <strong>Optimized</strong></span>
//               </li>
//             </ul>
//           </div>
//         </div>
//       </div>
//     </div>
//   );

//   return (
//     <div className="animate-fade-in">
//       {/* Header */}
//       <div className="mb-8">
//         <div className="flex justify-between items-center mb-4">
//           <h2 className="text-2xl font-bold text-gray-800">Analysis Results</h2>
//           <button 
//             onClick={onNew}
//             className="btn-primary flex items-center space-x-2"
//           >
//             <span>📤</span>
//             <span>Upload New File</span>
//           </button>
//         </div>
        
//         <div className="bg-white p-4 rounded-lg shadow-sm border">
//           <p className="text-sm text-gray-600 mb-2">File: <span className="font-medium">{data.filename}</span></p>
//           <p className="text-sm text-gray-600">Analysis ID: <span className="font-mono text-xs">{data.analysis_id}</span></p>
//         </div>
//       </div>

//       {/* Navigation Tabs */}
//       <div className="mb-6">
//         <nav className="flex space-x-1 bg-gray-100 p-1 rounded-lg">
//           {[
//             { id: 'overview', label: '📊 Overview', icon: '📊' },
//             { id: 'timeline', label: '⏰ Timeline', icon: '⏰' },
//             { id: 'transcript', label: '📝 Transcript', icon: '📝' },
//             { id: 'insights', label: '💡 Insights', icon: '💡' }
//           ].map(tab => (
//             <button
//               key={tab.id}
//               onClick={() => setActiveTab(tab.id)}
//               className={`flex items-center space-x-2 px-4 py-2 rounded-md font-medium transition-colors ${
//                 activeTab === tab.id
//                   ? 'bg-white text-indigo-600 shadow-sm'
//                   : 'text-gray-600 hover:text-gray-800'
//               }`}
//             >
//               <span>{tab.icon}</span>
//               <span>{tab.label}</span>
//             </button>
//           ))}
//         </nav>
//       </div>

//       {/* Tab Content */}
//       <div className="tab-content">
//         {activeTab === 'overview' && renderOverview()}
//         {activeTab === 'timeline' && renderTimeline()}
//         {activeTab === 'transcript' && renderTranscript()}
//         {activeTab === 'insights' && renderInsights()}
//       </div>
//     </div>
//   );
// };

// export default Dashboard;
// // ======================================================================================

import React, { useState } from 'react';

const Dashboard = ({ data, onNew }) => {
  const [activeTab, setActiveTab] = useState('overview');

  // Emotion to emoji mapping
  const emotionEmojis = {
    happy: '😊',
    sad: '😢',
    angry: '😠',
    neutral: '😐',
    fear: '😨',
    surprise: '😲',
    disgust: '🤢'
  };

  // Emotion to color mapping for timeline
  const emotionColors = {
    happy: 'bg-green-500',
    sad: 'bg-blue-500',
    angry: 'bg-red-500',
    neutral: 'bg-gray-500',
    fear: 'bg-purple-500',
    surprise: 'bg-yellow-500',
    disgust: 'bg-orange-500'
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getEmoji = (emotion) => emotionEmojis[emotion] || '😐';
  const getColor = (emotion) => emotionColors[emotion] || 'bg-gray-500';

  // 🆕 Export Transcript Function - NOW WORKING!
  const handleExportTranscript = () => {
    try {
      let content = '';
      
      // Header with file info
      content += `MULTIMODAL EMOTION RECOGNITION TRANSCRIPT\n`;
      content += `==========================================\n\n`;
      content += `File: ${data.filename || 'Unknown'}\n`;
      content += `Analysis ID: ${data.analysis_id || 'N/A'}\n`;
      content += `Generated: ${new Date().toLocaleString()}\n`;
      content += `Total Speakers: ${data.summary?.total_speakers || 0}\n`;
      content += `Overall Emotion: ${data.summary?.dominant_conversation_emotion || 'neutral'}\n\n`;
      
      // Transcript content
      if (data.segments && data.segments.length > 0) {
        content += `CONVERSATION TRANSCRIPT\n`;
        content += `======================\n\n`;
        
        data.segments.forEach((segment, index) => {
          content += `[${index + 1}] ${segment.speaker}\n`;
          content += `Time: ${formatTime(segment.start_time)} - ${formatTime(segment.end_time)}\n`;
          content += `Emotion: ${getEmoji(segment.final_emotion)} ${segment.final_emotion.toUpperCase()}\n`;
          content += `Confidence: ${Math.round(segment.confidence * 100)}%\n`;
          content += `Text: "${segment.text || 'No transcript available'}"\n`;
          
          // Add emotion breakdown if available
          if (segment.audio_emotions) {
            content += `Emotion Breakdown:\n`;
            Object.entries(segment.audio_emotions)
              .filter(([_, score]) => score > 0.1)
              .sort(([_, a], [__, b]) => b - a)
              .slice(0, 3)
              .forEach(([emotion, score]) => {
                content += `  - ${emotion}: ${Math.round(score * 100)}%\n`;
              });
          }
          content += `\n${'-'.repeat(50)}\n\n`;
        });
        
        // Summary statistics
        content += `CONVERSATION SUMMARY\n`;
        content += `===================\n\n`;
        
        if (data.summary?.speaker_summaries) {
          Object.entries(data.summary.speaker_summaries).forEach(([speaker, info]) => {
            content += `${speaker}:\n`;
            content += `  - Dominant Emotion: ${getEmoji(info.dominant_emotion)} ${info.dominant_emotion}\n`;
            content += `  - Total Segments: ${info.total_segments}\n\n`;
          });
        }
        
        if (data.summary?.overall_emotion_distribution) {
          content += `Emotion Distribution:\n`;
          Object.entries(data.summary.overall_emotion_distribution).forEach(([emotion, percentage]) => {
            content += `  - ${getEmoji(emotion)} ${emotion}: ${Math.round(percentage * 100)}%\n`;
          });
        }
        
      } else {
        content += `No transcript segments available.\n`;
      }
      
      // Create and download file
      const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      
      // Generate filename
      const baseFilename = data.filename ? 
        data.filename.replace(/\.[^/.]+$/, '') : 
        'emotion_analysis';
      const timestamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-');
      link.download = `${baseFilename}_transcript_${timestamp}.txt`;
      
      // Trigger download
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      
      console.log('Transcript exported successfully');
      
    } catch (error) {
      console.error('Export failed:', error);
      alert('Failed to export transcript. Please try again.');
    }
  };

  // 🆕 Export JSON Function (Bonus Feature)
  const handleExportJSON = () => {
    try {
      const exportData = {
        metadata: {
          filename: data.filename,
          analysis_id: data.analysis_id,
          export_timestamp: new Date().toISOString(),
          total_speakers: data.summary?.total_speakers || 0,
          dominant_emotion: data.summary?.dominant_conversation_emotion || 'neutral'
        },
        segments: data.segments || [],
        summary: data.summary || {},
        audio_analysis: data.audio_analysis || {},
        video_analysis: data.video_analysis || {}
      };
      
      const blob = new Blob([JSON.stringify(exportData, null, 2)], { 
        type: 'application/json' 
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      
      const baseFilename = data.filename ? 
        data.filename.replace(/\.[^/.]+$/, '') : 
        'emotion_analysis';
      const timestamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-');
      link.download = `${baseFilename}_data_${timestamp}.json`;
      
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      
    } catch (error) {
      console.error('JSON export failed:', error);
      alert('Failed to export JSON data. Please try again.');
    }
  };

  const renderOverview = () => (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card text-center">
          <h3 className="text-lg font-semibold text-gray-700 mb-2">Overall Emotion</h3>
          <div className="text-4xl mb-2 emoji-hover">
            {getEmoji(data.summary?.dominant_conversation_emotion)}
          </div>
          <p className="text-2xl font-bold text-indigo-600 uppercase">
            {data.summary?.dominant_conversation_emotion || 'NEUTRAL'}
          </p>
        </div>

        <div className="card text-center">
          <h3 className="text-lg font-semibold text-green-700 mb-2">Total Speakers</h3>
          <p className="text-4xl font-bold text-green-600">
            {data.summary?.total_speakers || 0}
          </p>
        </div>

        <div className="card text-center">
          <h3 className="text-lg font-semibold text-purple-700 mb-2">Segments Analyzed</h3>
          <p className="text-4xl font-bold text-purple-600">
            {data.segments?.length || 0}
          </p>
        </div>
      </div>

      {/* Emotion Distribution Chart */}
      <div className="card">
        <h3 className="text-xl font-semibold mb-4">Conversation Emotion Breakdown</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {data.summary?.overall_emotion_distribution && 
            Object.entries(data.summary.overall_emotion_distribution).map(([emotion, percentage]) => (
              <div key={emotion} className="emotion-card text-center p-4 rounded-lg bg-gray-50">
                <div className="text-3xl mb-2 emoji-hover">{getEmoji(emotion)}</div>
                <div className="font-semibold capitalize text-gray-700">{emotion}</div>
                <div className="text-2xl font-bold text-indigo-600">
                  {Math.round(percentage * 100)}%
                </div>
              </div>
            ))
          }
        </div>
      </div>

      {/* Speaker Emotions */}
      <div className="card">
        <h3 className="text-xl font-semibold mb-4">Speaker Emotions</h3>
        <div className="space-y-4">
          {data.summary?.speaker_summaries && 
            Object.entries(data.summary.speaker_summaries).map(([speaker, info]) => (
              <div key={speaker} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                <div className="flex items-center space-x-4">
                  <div className="text-3xl emoji-hover">{getEmoji(info.dominant_emotion)}</div>
                  <div>
                    <h4 className="font-semibold text-lg">{speaker}</h4>
                    <p className="text-gray-600 capitalize">
                      Dominant Emotion: <span className="font-medium">{info.dominant_emotion}</span>
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm text-gray-500">Total Segments</p>
                  <p className="text-2xl font-bold text-indigo-600">{info.total_segments}</p>
                </div>
              </div>
            ))
          }
        </div>
      </div>
    </div>
  );

  const renderTimeline = () => (
    <div className="space-y-6">
      <div className="card">
        <h3 className="text-xl font-semibold mb-6">Conversation Timeline</h3>
        
        {/* Timeline Header */}
        <div className="mb-6">
          <h4 className="text-lg font-medium mb-3">Emotion Legend:</h4>
          <div className="flex flex-wrap gap-3">
            {Object.entries(emotionEmojis).map(([emotion, emoji]) => (
              <div key={emotion} className="flex items-center space-x-2 bg-gray-100 px-3 py-1 rounded-full hover:bg-gray-200 transition-colors">
                <span className="text-lg emoji-hover">{emoji}</span>
                <span className="capitalize text-sm font-medium">{emotion}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Visual Timeline */}
        <div className="space-y-4">
          {data.segments && data.segments.map((segment, index) => (
            <div key={index} className="timeline-segment">
              {/* Timeline Bar */}
              <div className="flex items-center space-x-4 p-4 bg-white border-l-4 border-indigo-500 shadow-sm rounded-r-lg hover:shadow-md transition-shadow">
                
                {/* Time & Emoji */}
                <div className="flex-shrink-0 text-center">
                  <div className="text-2xl mb-1 emoji-hover">{getEmoji(segment.final_emotion)}</div>
                  <div className="text-xs font-mono text-gray-500">
                    {formatTime(segment.start_time)}
                  </div>
                  <div className="text-xs text-gray-400">
                    {formatTime(segment.end_time)}
                  </div>
                </div>

                {/* Speaker & Content */}
                <div className="flex-grow">
                  <div className="flex items-center space-x-3 mb-2">
                    <span className="speaker-badge bg-indigo-100 text-indigo-800 px-3 py-1 rounded-full text-sm font-medium">
                      {segment.speaker}
                    </span>
                    <span className={`emotion-badge px-2 py-1 rounded-full text-xs font-medium text-white ${getColor(segment.final_emotion)}`}>
                      {segment.final_emotion.toUpperCase()}
                    </span>
                    <span className="confidence-badge text-xs text-gray-500">
                      {Math.round(segment.confidence * 100)}% confidence
                    </span>
                  </div>
                  
                  <p className="text-gray-800 bg-gray-50 p-3 rounded-lg">
                    "{segment.text || 'No transcript available'}"
                  </p>

                  {/* Emotion Breakdown */}
                  <div className="mt-3 flex flex-wrap gap-2">
                    {segment.audio_emotions && Object.entries(segment.audio_emotions)
                      .filter(([_, score]) => score > 0.1)
                      .sort(([_, a], [__, b]) => b - a)
                      .slice(0, 3)
                      .map(([emotion, score]) => (
                        <span key={emotion} className="text-xs bg-gray-200 px-2 py-1 rounded hover:bg-gray-300 transition-colors">
                          <span className="emoji-hover">{getEmoji(emotion)}</span> {emotion} ({Math.round(score * 100)}%)
                        </span>
                      ))
                    }
                  </div>
                </div>

                {/* Duration Bar */}
                <div className="flex-shrink-0 w-20">
                  <div className="text-xs text-gray-500 mb-1">Duration</div>
                  <div className="duration-bar h-2 bg-gray-200 rounded-full">
                    <div 
                      className={`duration-fill h-2 rounded-full ${getColor(segment.final_emotion)}`}
                      style={{width: `${Math.min((segment.end_time - segment.start_time) * 10, 100)}%`}}
                    ></div>
                  </div>
                  <div className="text-xs text-gray-400 mt-1">
                    {Math.round(segment.end_time - segment.start_time)}s
                  </div>
                </div>
              </div>

              {/* Connector Line */}
              {index < data.segments.length - 1 && (
                <div className="timeline-connector w-px h-4 bg-gray-300 ml-8 my-2"></div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const renderTranscript = () => (
    <div className="card">
      <div className="flex justify-between items-center mb-6">
        <h3 className="text-xl font-semibold">Full Transcript with Emotions</h3>
        <div className="flex space-x-2">
          <button 
            onClick={handleExportTranscript}
            className="btn-secondary text-sm hover:bg-gray-800 transition-colors"
          >
            📄 Export Transcript
          </button>
          <button 
            onClick={handleExportJSON}
            className="btn-secondary text-sm hover:bg-gray-800 transition-colors"
          >
            💾 Export JSON
          </button>
        </div>
      </div>

      <div className="space-y-4 max-h-96 overflow-y-auto">
        {data.segments && data.segments.map((segment, index) => (
          <div key={index} className={`transcript-bubble p-4 rounded-lg border-l-4 ${
            segment.speaker === 'Speaker_1' ? 'speaker-1-bubble bg-blue-50 border-blue-500' : 
            segment.speaker === 'Speaker_2' ? 'speaker-2-bubble bg-green-50 border-green-500' :
            'speaker-3-bubble bg-purple-50 border-purple-500'
          }`}>
            
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center space-x-3">
                <span className="font-semibold text-gray-800">{segment.speaker}</span>
                <span className="text-2xl emoji-hover">{getEmoji(segment.final_emotion)}</span>
                <span className={`px-2 py-1 rounded text-xs font-medium text-white ${getColor(segment.final_emotion)}`}>
                  {segment.final_emotion}
                </span>
              </div>
              <span className="text-sm text-gray-500">
                {formatTime(segment.start_time)} - {formatTime(segment.end_time)}
              </span>
            </div>

            <p className="text-gray-800 leading-relaxed">
              "{segment.text || 'No transcript available'}"
            </p>

            <div className="mt-2 text-xs text-gray-500">
              Confidence: {Math.round(segment.confidence * 100)}%
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderInsights = () => (
    <div className="space-y-6">
      <div className="card">
        <h3 className="text-xl font-semibold mb-4">🔍 Conversation Insights</h3>
        
        <div className="grid md:grid-cols-2 gap-6">
          {/* Emotional Patterns */}
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 p-6 rounded-lg">
            <h4 className="font-semibold mb-3 flex items-center">
              📊 Emotional Patterns
            </h4>
            <ul className="space-y-2 text-sm">
              <li className="flex items-center space-x-2">
                <span>🎯</span>
                <span>Most emotional speaker: <strong>
                  {data.summary?.speaker_summaries && 
                    Object.entries(data.summary.speaker_summaries)
                      .reduce((prev, [speaker, info]) => 
                        info.total_segments > (prev.segments || 0) ? {speaker, segments: info.total_segments} : prev, {}
                      ).speaker
                  }
                </strong></span>
              </li>
              <li className="flex items-center space-x-2">
                <span>⏱️</span>
                <span>Average segment duration: <strong>
                  {data.segments && Math.round(
                    data.segments.reduce((sum, seg) => sum + (seg.end_time - seg.start_time), 0) / data.segments.length
                  )}s</strong>
                </span>
              </li>
              <li className="flex items-center space-x-2">
                <span>🗣️</span>
                <span>Total conversation time: <strong>
                  {data.segments && Math.round(Math.max(...data.segments.map(seg => seg.end_time)))}s
                </strong></span>
              </li>
            </ul>
          </div>

          {/* Accessibility Features */}
          <div className="bg-gradient-to-br from-green-50 to-emerald-50 p-6 rounded-lg">
            <h4 className="font-semibold mb-3 flex items-center">
              ♿ Accessibility Summary
            </h4>
            <ul className="space-y-2 text-sm">
              <li className="flex items-center space-x-2">
                <span>👁️</span>
                <span>Visual timeline: <strong>Available</strong></span>
              </li>
              <li className="flex items-center space-x-2">
                <span>😊</span>
                <span>Emotion emojis: <strong>Enabled</strong></span>
              </li>
              <li className="flex items-center space-x-2">
                <span>🎨</span>
                <span>Color-coded speakers: <strong>Active</strong></span>
              </li>
              <li className="flex items-center space-x-2">
                <span>📱</span>
                <span>Mobile-friendly: <strong>Optimized</strong></span>
              </li>
              <li className="flex items-center space-x-2">
                <span>📄</span>
                <span>Export functionality: <strong>Working</strong></span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="animate-fade-in">
      {/* Header */}
      <div className="mb-8">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-bold text-gray-800">Analysis Results</h2>
          <button 
            onClick={onNew}
            className="btn-primary"
          >
            <span>📤</span>
            <span>Upload New File</span>
          </button>
        </div>
        
        <div className="bg-white p-4 rounded-lg shadow-sm border">
          <p className="text-sm text-gray-600 mb-2">File: <span className="font-medium">{data.filename}</span></p>
          <p className="text-sm text-gray-600">Analysis ID: <span className="font-mono text-xs">{data.analysis_id}</span></p>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="mb-6">
        <nav className="nav-tabs flex space-x-1 bg-gray-100 p-1 rounded-lg">
          {[
            { id: 'overview', label: '📊 Overview', icon: '📊' },
            { id: 'timeline', label: '⏰ Timeline', icon: '⏰' },
            { id: 'transcript', label: '📝 Transcript', icon: '📝' },
            { id: 'insights', label: '💡 Insights', icon: '💡' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`nav-tab flex items-center space-x-2 px-4 py-2 rounded-md font-medium transition-colors ${
                activeTab === tab.id
                  ? 'nav-tab-active bg-white text-indigo-600 shadow-sm'
                  : 'nav-tab-inactive text-gray-600 hover:text-gray-800'
              }`}
            >
              <span>{tab.icon}</span>
              <span>{tab.label}</span>
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="tab-content">
        {activeTab === 'overview' && renderOverview()}
        {activeTab === 'timeline' && renderTimeline()}
        {activeTab === 'transcript' && renderTranscript()}
        {activeTab === 'insights' && renderInsights()}
      </div>
    </div>
  );
};

export default Dashboard;
