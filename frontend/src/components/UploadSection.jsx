import { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, CheckCircle, AlertCircle, FileImage, Loader2, Sparkles } from 'lucide-react';
import { predictImage, fetchModels } from '../api/client';

const FALLBACK_MODELS = [
  'MobileNetV2',
  'DenseNet121',
  'ResNet50',
  'EfficientNetB0',
  'VGG16',
  'InceptionV3',
];

function ConfidenceBar({ confidence, prediction }) {
  const isPneumonia = prediction === 'Pneumonia';
  const barColor = isPneumonia ? 'bg-red-500' : 'bg-emerald-500';

  return (
    <div className="mt-4 space-y-2">
      <div className="flex justify-between text-xs font-medium text-slate-500 dark:text-slate-400">
        <span>Confidence score</span>
        <span>{confidence.toFixed(1)}%</span>
      </div>
      <div className="h-2.5 w-full rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ease-out ${barColor}`}
          style={{ width: `${Math.min(confidence, 100)}%` }}
        />
      </div>
    </div>
  );
}

export default function UploadSection({ onUploadSuccess }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [selectedModel, setSelectedModel] = useState('MobileNetV2');
  const [modelOptions, setModelOptions] = useState(
    FALLBACK_MODELS.map((name) => ({ name, available: true }))
  );
  const [modelsError, setModelsError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    const loadModels = async (attempt = 0) => {
      try {
        const res = await fetchModels();
        if (cancelled) return;

        if (Array.isArray(res.data?.models) && res.data.models.length > 0) {
          setModelOptions(res.data.models);
          const defaultModel = res.data.default;
          const defaultAvailable = res.data.models.find(
            (model) => model.name === defaultModel && model.available !== false
          );
          const firstAvailable = res.data.models.find((model) => model.available !== false);
          if (defaultAvailable) {
            setSelectedModel(defaultModel);
          } else if (firstAvailable) {
            setSelectedModel(firstAvailable.name);
          }
          setModelsError(null);
        }
      } catch {
        if (cancelled) return;
        if (attempt < 5) {
          window.setTimeout(() => loadModels(attempt + 1), 800 * (attempt + 1));
          return;
        }
        setModelsError(
          'Could not reach the backend. Start it with: cd backend && python app.py (port 5000), then refresh.'
        );
        setModelOptions(FALLBACK_MODELS.map((name) => ({ name, available: true })));
      }
    };

    loadModels();
    return () => {
      cancelled = true;
    };
  }, []);

  const onDrop = useCallback((acceptedFiles) => {
    const selectedFile = acceptedFiles[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/jpeg': ['.jpeg', '.jpg'],
      'image/png': ['.png'],
    },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024,
  });

  const handlePredict = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await predictImage(file, selectedModel);
      setResult(response.data);
      if (onUploadSuccess) onUploadSuccess();
    } catch (err) {
      setError(
        err.response?.data?.error ||
          'Failed to process image. Make sure the backend is running and the model is loaded.'
      );
    } finally {
      setLoading(false);
    }
  };

  const clearSelection = () => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-6 space-y-8">
      <div className="text-center space-y-3">
        <div className="inline-flex items-center gap-2 rounded-full bg-medical-50 dark:bg-medical-900/30 px-3 py-1 text-sm font-medium text-medical-700 dark:text-medical-300">
          <Sparkles className="h-4 w-4" />
          AI-powered chest X-ray analysis
        </div>
        <h2 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-white">
          Pneumonia Detection
        </h2>
        <p className="text-slate-500 dark:text-slate-400 max-w-xl mx-auto">
          Upload a chest X-ray image to receive an instant prediction with a confidence score.
        </p>
      </div>

      <div className="max-w-md mx-auto">
        <label
          htmlFor="model-select"
          className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2"
        >
          Detection model
        </label>
        <select
          id="model-select"
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          disabled={loading}
          className="w-full rounded-xl border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-900 dark:text-white px-4 py-3 text-sm shadow-sm focus:outline-none focus:ring-2 focus:ring-medical-500 disabled:opacity-60"
        >
          {modelOptions.map((model) => (
            <option key={model.name} value={model.name} disabled={model.available === false}>
              {model.name}
              {model.available === false ? ' (not trained)' : ''}
            </option>
          ))}
        </select>
        <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
          Select the model to use before uploading and analyzing your X-ray.
        </p>
        {modelsError && (
          <p className="mt-2 text-xs text-amber-600 dark:text-amber-400">{modelsError}</p>
        )}
      </div>

      {!file ? (
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all duration-200
            ${
              isDragActive
                ? 'border-medical-500 bg-medical-50 dark:bg-medical-900/20 scale-[1.01]'
                : 'border-slate-300 dark:border-slate-700 hover:border-medical-400 hover:bg-slate-50 dark:hover:bg-slate-800/50'
            }
          `}
        >
          <input {...getInputProps()} />
          <UploadCloud
            className={`mx-auto h-16 w-16 mb-4 ${isDragActive ? 'text-medical-500' : 'text-slate-400'}`}
          />
          <p className="text-lg font-medium text-slate-700 dark:text-slate-200">
            {isDragActive ? 'Drop the X-ray here...' : 'Drag & drop an X-ray image here'}
          </p>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-2">
            or click to select a file (JPG, PNG, max 10 MB)
          </p>
        </div>
      ) : (
        <div className="bg-white dark:bg-slate-800 rounded-2xl shadow-sm border border-slate-200 dark:border-slate-700 overflow-hidden">
          {result?.gradcam?.overlay ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 border-b border-slate-200 dark:border-slate-700">
              <div className="bg-slate-100 dark:bg-slate-900 p-6 flex flex-col items-center justify-center min-h-[300px]">
                <p className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-3 self-start">
                  Original X-ray
                </p>
                <img
                  src={preview}
                  alt="Original chest X-ray"
                  className="max-w-full max-h-[360px] object-contain rounded shadow-sm"
                />
              </div>
              <div className="bg-slate-100 dark:bg-slate-900 p-6 flex flex-col items-center justify-center min-h-[300px]">
                <p className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-3 self-start">
                  Grad-CAM heatmap
                </p>
                <img
                  src={result.gradcam.overlay}
                  alt={`Grad-CAM overlay for ${result.model_name || result.model || selectedModel}`}
                  className="max-w-full max-h-[360px] object-contain rounded shadow-sm"
                />
              </div>
            </div>
          ) : (
            <div className="bg-slate-100 dark:bg-slate-900 p-6 flex items-center justify-center min-h-[300px] border-b border-slate-200 dark:border-slate-700">
              <img
                src={preview}
                alt="X-ray preview"
                className="max-w-full max-h-[400px] object-contain rounded shadow-sm"
              />
            </div>
          )}

          <div className="p-8 flex flex-col justify-center">
            <div className="flex items-center space-x-3 mb-6">
              <FileImage className="h-6 w-6 text-medical-500 shrink-0" />
              <span
                className="font-medium text-slate-700 dark:text-slate-300 truncate"
                title={file.name}
              >
                {file.name}
              </span>
            </div>

            {!result && !loading && !error && (
              <div className="space-y-4 mt-auto">
                <button
                  onClick={handlePredict}
                  className="w-full bg-medical-600 hover:bg-medical-700 text-white font-medium py-3 px-4 rounded-xl transition-colors shadow-sm"
                >
                  Analyze Image
                </button>
                <button
                  onClick={clearSelection}
                  className="w-full bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 hover:bg-slate-50 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 font-medium py-3 px-4 rounded-xl transition-colors"
                >
                  Select Different File
                </button>
              </div>
            )}

            {loading && (
              <div className="flex flex-col items-center justify-center space-y-4 py-8">
                <Loader2 className="h-10 w-10 text-medical-500 animate-spin" />
                <p className="text-slate-600 dark:text-slate-400 font-medium animate-pulse">
                  Running inference and generating Grad-CAM...
                </p>
              </div>
            )}

            {error && (
              <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-4 mt-auto">
                <div className="flex items-start space-x-3">
                  <AlertCircle className="h-5 w-5 text-red-500 mt-0.5 shrink-0" />
                  <div>
                    <h4 className="text-red-800 dark:text-red-200 font-medium">Analysis failed</h4>
                    <p className="text-sm text-red-600 dark:text-red-300 mt-1">{error}</p>
                  </div>
                </div>
                <button
                  onClick={clearSelection}
                  className="mt-4 text-sm font-medium text-red-600 dark:text-red-400 hover:underline"
                >
                  Try Again
                </button>
              </div>
            )}

            {result && (
              <div
                className={`rounded-xl p-6 mt-auto border ${
                  result.prediction === 'Pneumonia'
                    ? 'bg-red-50 border-red-200 dark:bg-red-900/10 dark:border-red-900/50'
                    : 'bg-emerald-50 border-emerald-200 dark:bg-emerald-900/10 dark:border-emerald-900/50'
                }`}
              >
                <h3 className="text-lg font-semibold text-slate-800 dark:text-slate-200 mb-4">
                  Prediction Result
                </h3>
                <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">
                  Model used:{' '}
                  <span className="font-medium text-slate-700 dark:text-slate-300">
                    {result.model_name || result.model || selectedModel}
                  </span>
                </p>
                <div className="flex items-center justify-between gap-4">
                  <div>
                    <p className="text-sm text-slate-500 dark:text-slate-400 uppercase tracking-wider font-semibold mb-1">
                      Diagnosis
                    </p>
                    <div className="flex items-center space-x-2">
                      {result.prediction === 'Normal' ? (
                        <CheckCircle className="h-6 w-6 text-emerald-500" />
                      ) : (
                        <AlertCircle className="h-6 w-6 text-red-500" />
                      )}
                      <span
                        className={`text-2xl font-bold ${
                          result.prediction === 'Pneumonia'
                            ? 'text-red-600 dark:text-red-400'
                            : 'text-emerald-600 dark:text-emerald-400'
                        }`}
                      >
                        {result.prediction}
                      </span>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-slate-500 dark:text-slate-400 uppercase tracking-wider font-semibold mb-1">
                      Confidence
                    </p>
                    <span className="text-2xl font-bold text-slate-800 dark:text-slate-200">
                      {Number(result.confidence).toFixed(1)}%
                    </span>
                  </div>
                </div>

                <ConfidenceBar confidence={Number(result.confidence)} prediction={result.prediction} />

                {result.gradcam?.layer && (
                  <p className="mt-4 text-xs text-slate-500 dark:text-slate-400">
                    Grad-CAM layer: {result.gradcam.backbone_layer || result.gradcam.layer}
                  </p>
                )}

                {result.gradcam_error && (
                  <div className="mt-6 rounded-lg border border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-900/20 p-4">
                    <p className="text-sm font-medium text-amber-800 dark:text-amber-200">
                      Grad-CAM unavailable
                    </p>
                    <p className="text-sm text-amber-700 dark:text-amber-300 mt-1">
                      {result.gradcam_error}
                    </p>
                  </div>
                )}

                <button
                  onClick={clearSelection}
                  className="w-full mt-6 bg-slate-900 hover:bg-slate-800 dark:bg-slate-700 dark:hover:bg-slate-600 text-white font-medium py-2.5 px-4 rounded-lg transition-colors"
                >
                  Upload Another
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
