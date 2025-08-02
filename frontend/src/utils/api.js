const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

export const uploadFile = async (file, onProgress) => {
  const formData = new FormData();
  formData.append('file', file);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();

    xhr.upload.addEventListener('progress', (event) => {
      if (event.lengthComputable && onProgress) {
        const progress = Math.round((event.loaded / event.total) * 100);
        onProgress(progress);
      }
    });

    xhr.onload = () => {
      if (xhr.status === 200) {
        try {
          const response = JSON.parse(xhr.responseText);
          resolve(response);
        } catch (err) {
          reject(new Error('Invalid response'));
        }
      } else {
        reject(new Error(`Failed: ${xhr.status} ${xhr.statusText}`));
      }
    };

    xhr.onerror = () => {
      reject(new Error('Network error during upload'));
    };

    xhr.open('POST', `${API_BASE_URL}/upload`);
    xhr.send(formData);
  });
};

export const getAnalysis = async (analysisId) => {
  try {
    const resp = await fetch(`${API_BASE_URL}/analyze/${analysisId}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return await resp.json();
  } catch (err) {
    console.error('Fetch analysis error', err);
    throw err;
  }
};
export const getEmotionLabels = async () => {
  try {
    const resp = await fetch(`${API_BASE_URL}/emotions`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return await resp.json();
  } catch (err) {
    console.error('Fetch emotion labels error', err);
    throw err;
  }
};
export const healthCheck = async () => {
  try {
    const resp = await fetch(`${API_BASE_URL}/health`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return await resp.json();
  } catch (err) {
    console.error('Health check error', err);
    throw err;
  }
};
