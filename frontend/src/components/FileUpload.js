import React, { useRef, useState } from 'react';
import { uploadFile } from '../utils/api';

const FileUpload = ({ onStart, onComplete }) => {
  const [dragActive, setDragActive] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const fileRef = useRef();

  const handleDrag = (e) => {
    e.preventDefault(); e.stopPropagation();
    if (['dragenter', 'dragover'].includes(e.type)) {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault(); e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = async (file) => {
    const allowedTypes = [
      'audio/wav', 'audio/mp3', 'audio/mpeg',
      'video/mp4', 'video/avi', 'video/mov', 'video/quicktime', 'video/x-msvideo'
    ];
    // Validate file type
    if (!allowedTypes.includes(file.type)) {
      alert('Please select a valid audio or video file (WAV, MP3, MP4, AVI, MOV)');
      return;
    }
    // Validate size (max 500MB)
    if (file.size > 500 * 1024 * 1024) {
      alert('File size must be less than 500MB');
      return;
    }
    // Notify parent upload is starting
    if (onStart && typeof onStart === 'function') onStart();

    try {
      const result = await uploadFile(file, (progress) => {
        setUploadProgress(progress);
      });
      // Call parent with analysis result
      if (onComplete && typeof onComplete === 'function') onComplete(result);
    } catch (err) {
      console.error('Upload failed:', err);
      alert('Failed to process file. Please try again.');
      if (onComplete && typeof onComplete === 'function') onComplete(null);
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <div
        className={`relative border-2 border-dashed rounded-lg p-12 text-center transition-colors duration-200 ${
          dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
        }`}
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={fileRef}
          type="file"
          accept="audio/*,video/*"
          className="hidden"
          onChange={handleFileSelect}
        />
        {/* UI content */}
        <div className="space-y-4">
          {/* SVG Icon */}
          <div className="mx-auto w-12 h-12 text-gray-400 cursor-pointer" onClick={() => fileRef.current?.click()}>
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6H16a4 4 0 011 7.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
          </div>
          <div>
            <p className="text-lg font-medium text-gray-900">Drop your audio or video file here</p>
            <p className="text-sm text-gray-600 mt-1">or click to select a file</p>
          </div>
          <div className="text-xs text-gray-500">
            <p>Supported formats: WAV, MP3, MP4, AVI, MOV</p>
            <p>Maximum file size: 500MB</p>
          </div>
        </div>
        {/* Upload progress */}
        {uploadProgress > 0 && uploadProgress < 100 && (
          <div className="mt-4">
            <div className="flex justify-between text-sm text-gray-600 mb-1">
              <span>Uploading...</span>
              <span>{uploadProgress}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FileUpload;
