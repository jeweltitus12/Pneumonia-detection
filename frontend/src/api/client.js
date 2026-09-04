import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
});

export const predictImage = (file, model = 'MobileNetV2') => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('model', model);
  return api.post('/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

export const fetchModels = () => api.get('/models');
export const fetchHistory = () => api.get('/history');
export const fetchStats = () => api.get('/stats');
export const fetchHealth = () => api.get('/health');
