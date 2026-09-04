import { useState, useEffect } from 'react';
import { Stethoscope, Moon, Sun, Circle } from 'lucide-react';
import UploadSection from './components/UploadSection';
import Dashboard from './components/Dashboard';
import { fetchHealth } from './api/client';

function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [activeTab, setActiveTab] = useState('upload'); // 'upload' or 'dashboard'
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [backendStatus, setBackendStatus] = useState('checking');

  useEffect(() => {
    let cancelled = false;
    let attempts = 0;
    const maxAttempts = 40;
    let timer;

    const poll = () => {
      fetchHealth()
        .then((res) => {
          if (cancelled) return;
          const ready = Boolean(res.data?.model_loaded);
          setBackendStatus(ready ? 'ready' : 'no-model');
          if (!ready && attempts < maxAttempts) {
            attempts += 1;
            timer = setTimeout(poll, 3000);
          }
        })
        .catch(() => {
          if (cancelled) return;
          setBackendStatus('offline');
          if (attempts < maxAttempts) {
            attempts += 1;
            timer = setTimeout(poll, 3000);
          }
        });
    };

    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    // Check system preference on load
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      setDarkMode(true);
    }
  }, []);

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  const toggleDarkMode = () => setDarkMode(!darkMode);

  const handleUploadSuccess = () => {
    setRefreshTrigger(prev => prev + 1);
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 sticky top-0 z-10 transition-colors">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            
            {/* Logo */}
            <div className="flex items-center space-x-3">
              <div className="bg-medical-500 p-2 rounded-lg text-white">
                <Stethoscope className="h-6 w-6" />
              </div>
              <span className="font-bold text-xl text-slate-900 dark:text-white tracking-tight">
                Pneumo<span className="text-medical-500">Detect</span>
              </span>
            </div>

            {/* Navigation */}
            <nav className="hidden md:flex space-x-1">
              <button
                onClick={() => setActiveTab('upload')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'upload' 
                    ? 'bg-slate-100 text-slate-900 dark:bg-slate-800 dark:text-white' 
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50 dark:text-slate-400 dark:hover:text-white dark:hover:bg-slate-800/50'
                }`}
              >
                Scan Image
              </button>
              <button
                onClick={() => setActiveTab('dashboard')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'dashboard' 
                    ? 'bg-slate-100 text-slate-900 dark:bg-slate-800 dark:text-white' 
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50 dark:text-slate-400 dark:hover:text-white dark:hover:bg-slate-800/50'
                }`}
              >
                Dashboard
              </button>
            </nav>

            {/* Theme Toggle + Status */}
            <div className="flex items-center gap-3">
              <div
                className="hidden sm:flex items-center gap-1.5 text-xs font-medium text-slate-500 dark:text-slate-400"
                title={
                  backendStatus === 'ready'
                    ? 'Backend and model ready'
                    : backendStatus === 'no-model'
                      ? 'Backend running but model not loaded'
                      : backendStatus === 'offline'
                        ? 'Backend offline'
                        : 'Checking backend...'
                }
              >
                <Circle
                  className={`h-2.5 w-2.5 fill-current ${
                    backendStatus === 'ready'
                      ? 'text-emerald-500'
                      : backendStatus === 'no-model'
                        ? 'text-amber-500'
                        : backendStatus === 'offline'
                          ? 'text-red-500'
                          : 'text-slate-400'
                  }`}
                />
                {backendStatus === 'ready' && 'Model ready'}
                {backendStatus === 'no-model' && 'Model missing'}
                {backendStatus === 'offline' && 'Backend offline'}
                {backendStatus === 'checking' && 'Connecting...'}
              </div>
              <button
                onClick={toggleDarkMode}
                className="p-2 rounded-full text-slate-500 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800 transition-colors"
                aria-label="Toggle Dark Mode"
              >
                {darkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 py-8 px-4 sm:px-6 lg:px-8">
        {/* Mobile Navigation (shows only on small screens) */}
        <div className="md:hidden flex rounded-lg p-1 bg-slate-100 dark:bg-slate-800 mb-8 max-w-md mx-auto">
          <button
            onClick={() => setActiveTab('upload')}
            className={`flex-1 py-2 text-sm font-medium rounded-md transition-all ${
              activeTab === 'upload' 
                ? 'bg-white text-slate-900 shadow-sm dark:bg-slate-700 dark:text-white' 
                : 'text-slate-500 dark:text-slate-400'
            }`}
          >
            Scan Image
          </button>
          <button
            onClick={() => setActiveTab('dashboard')}
            className={`flex-1 py-2 text-sm font-medium rounded-md transition-all ${
              activeTab === 'dashboard' 
                ? 'bg-white text-slate-900 shadow-sm dark:bg-slate-700 dark:text-white' 
                : 'text-slate-500 dark:text-slate-400'
            }`}
          >
            Dashboard
          </button>
        </div>

        {/* Tab Content */}
        <div className="animate-in fade-in slide-in-from-bottom-4 duration-500 ease-out">
          {activeTab === 'upload' ? (
            <UploadSection onUploadSuccess={handleUploadSuccess} />
          ) : (
            <Dashboard refreshTrigger={refreshTrigger} />
          )}
        </div>
      </main>
      
      {/* Footer */}
      <footer className="border-t border-slate-200 dark:border-slate-800 py-6 text-center text-slate-500 dark:text-slate-400 text-sm">
        <p>PneumoDetect AI • For demonstration purposes only. Not a substitute for professional medical advice.</p>
      </footer>
    </div>
  );
}

export default App;
