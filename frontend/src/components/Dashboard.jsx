import { useState, useEffect } from 'react';
import { Activity, ShieldCheck, AlertTriangle, RefreshCw, AlertCircle } from 'lucide-react';
import { format } from 'date-fns';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';
import { fetchHistory, fetchStats } from '../api/client';

const CHART_COLORS = ['#10b981', '#ef4444'];

export default function Dashboard({ refreshTrigger }) {
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState({ total: 0, pneumonia: 0, normal: 0 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [historyRes, statsRes] = await Promise.all([fetchHistory(), fetchStats()]);
      setHistory(historyRes.data);
      setStats(statsRes.data);
    } catch (err) {
      setError('Unable to load dashboard data. Ensure the backend server is running.');
      console.error('Failed to fetch dashboard data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [refreshTrigger]);

  const chartData = [
    { name: 'Normal', value: stats.normal },
    { name: 'Pneumonia', value: stats.pneumonia },
  ].filter((item) => item.value > 0);

  if (loading) {
    return (
      <div className="p-12 text-center">
        <RefreshCw className="h-8 w-8 text-medical-500 animate-spin mx-auto mb-3" />
        <p className="text-slate-500 dark:text-slate-400">Loading dashboard...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-lg mx-auto p-8 text-center bg-white dark:bg-slate-800 rounded-2xl border border-red-200 dark:border-red-900/50">
        <AlertCircle className="h-10 w-10 text-red-500 mx-auto mb-3" />
        <p className="text-slate-700 dark:text-slate-300 mb-4">{error}</p>
        <button
          onClick={fetchData}
          className="inline-flex items-center gap-2 bg-medical-600 hover:bg-medical-700 text-white px-4 py-2 rounded-lg text-sm font-medium"
        >
          <RefreshCw className="h-4 w-4" />
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="w-full max-w-6xl mx-auto p-6 space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-slate-900 dark:text-white">Analytics Dashboard</h2>
          <p className="text-slate-500 dark:text-slate-400 text-sm mt-1">
            Overview of all prediction history
          </p>
        </div>
        <button
          onClick={fetchData}
          className="inline-flex items-center gap-2 text-sm font-medium text-medical-600 hover:text-medical-700 dark:text-medical-400"
        >
          <RefreshCw className="h-4 w-4" />
          Refresh
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white dark:bg-slate-800 rounded-2xl p-6 border border-slate-200 dark:border-slate-700 shadow-sm flex items-center space-x-4">
          <div className="p-4 bg-medical-100 dark:bg-medical-900/30 text-medical-600 dark:text-medical-400 rounded-xl">
            <Activity className="h-8 w-8" />
          </div>
          <div>
            <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Total Scans</p>
            <h3 className="text-3xl font-bold text-slate-900 dark:text-white">{stats.total}</h3>
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-2xl p-6 border border-slate-200 dark:border-slate-700 shadow-sm flex items-center space-x-4">
          <div className="p-4 bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 rounded-xl">
            <AlertTriangle className="h-8 w-8" />
          </div>
          <div>
            <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Pneumonia Detected</p>
            <h3 className="text-3xl font-bold text-slate-900 dark:text-white">{stats.pneumonia}</h3>
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-2xl p-6 border border-slate-200 dark:border-slate-700 shadow-sm flex items-center space-x-4">
          <div className="p-4 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400 rounded-xl">
            <ShieldCheck className="h-8 w-8" />
          </div>
          <div>
            <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Normal Scans</p>
            <h3 className="text-3xl font-bold text-slate-900 dark:text-white">{stats.normal}</h3>
          </div>
        </div>
      </div>

      {stats.total > 0 && (
        <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-sm p-6">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">Result Distribution</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={chartData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={90}
                  paddingAngle={4}
                >
                  {chartData.map((entry, index) => (
                    <Cell key={entry.name} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      <div className="bg-white dark:bg-slate-800 rounded-2xl border border-slate-200 dark:border-slate-700 shadow-sm overflow-hidden">
        <div className="px-6 py-5 border-b border-slate-200 dark:border-slate-700">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white">Recent Predictions</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="bg-slate-50 dark:bg-slate-900/50 text-slate-500 dark:text-slate-400 text-sm uppercase tracking-wider">
                <th className="px-6 py-4 font-medium">Date</th>
                <th className="px-6 py-4 font-medium">File Name</th>
                <th className="px-6 py-4 font-medium">Result</th>
                <th className="px-6 py-4 font-medium">Confidence</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
              {history.length === 0 ? (
                <tr>
                  <td colSpan="4" className="px-6 py-8 text-center text-slate-500 dark:text-slate-400">
                    No predictions yet. Upload an image to see history.
                  </td>
                </tr>
              ) : (
                history.map((item) => (
                  <tr
                    key={item.id}
                    className="hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors"
                  >
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-600 dark:text-slate-300">
                      {format(new Date(item.timestamp), 'MMM d, yyyy h:mm a')}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-slate-900 dark:text-white">
                      {item.filename}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span
                        className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          item.prediction === 'Normal'
                            ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-400'
                            : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400'
                        }`}
                      >
                        {item.prediction}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-600 dark:text-slate-300">
                      {Number(item.confidence).toFixed(1)}%
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
