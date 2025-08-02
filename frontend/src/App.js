// import React, { useState } from 'react';
// import './App.css';
// import Dashboard from './components/Dashboard';
// import FileUpload from './components/FileUpload';

// function App() {
//   const [analysisResult, setAnalysisResult] = useState(null);
//   const [isLoading, setIsLoading] = useState(false);

//   const handleAnalysisComplete = (result) => {
//     setAnalysisResult(result);
//     setIsLoading(false);
//   };

//   const handleAnalysisStart = () => {
//     setIsLoading(true);
//     setAnalysisResult(null);
//   };

//   return (
//     <div className="App">
//       <header className="App-header">
//         <h1 className="text-4xl font-bold text-blue-600 mb-2">
//           Multimodal Emotion Recognition
//         </h1>
//         <p className="text-lg text-gray-600 mb-8">
//           AI-Powered Conversational Emotion Analysis for Group Interactions
//         </p>
//       </header>

//       <main className="container mx-auto px-4 py-8">
//         {!analysisResult && !isLoading && (
//           <FileUpload 
//             onAnalysisStart={handleAnalysisStart}
//             onAnalysisComplete={handleAnalysisComplete}
//           />
//         )}

//         {isLoading && (
//           <div className="flex justify-center items-center py-16">
//             <div className="text-center">
//               <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
//               <p className="text-lg text-gray-600">Processing your file...</p>
//               <p className="text-sm text-gray-500 mt-2">
//                 This may take a few minutes depending on file size
//               </p>
//             </div>
//           </div>
//         )}

//         {analysisResult && (
//           <Dashboard 
//             analysisResult={analysisResult}
//             onNewAnalysis={() => setAnalysisResult(null)}
//           />
//         )}
//       </main>
//     </div>
//   );
// }

// export default App;


import React, { useState } from 'react';
import Dashboard from './components/Dashboard';
import FileUpload from './components/FileUpload';

function App() {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);

  // Called when user starts uploading
  const handleStart = () => {
    setLoading(true);
    setAnalysis(null);
  };

  // Called when upload and analysis complete
  const handleComplete = (result) => {
    setAnalysis(result);
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-blue-100 to-indigo-50 p-4 font-sans">
      {/* Header */}
      <header className="text-center mb-6">
        <h1 className="text-4xl font-bold text-indigo-700 mb-2">🎤 Multimodal Emotion Dashboard</h1>
        <p className="text-lg text-gray-700">AI-powered analysis of group conversations for emotional insights</p>
      </header>
      {/* Main Content */}
      <main className="max-w-6xl mx-auto">
        {!analysis && !loading && (
          <FileUpload onStart={handleStart} onComplete={handleComplete} />
        )}
        {loading && (
          <div className="flex justify-center items-center py-20">
            <div className="text-center">
              {/* Simple spinner */}
              <div className="border-4 border-t-4 border-indigo-400 border-t-indigo-600 rounded-full w-12 h-12 mx-auto mb-4 animate-spin"></div>
              <p className="text-gray-600 text-lg">Processing your file...</p>
            </div>
          </div>
        )}
        {analysis && (
          <Dashboard data={analysis} onNew={() => setAnalysis(null)} />
        )}
      </main>
    </div>
  );
}

export default App;
