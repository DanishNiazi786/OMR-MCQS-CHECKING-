import React, { useState, useEffect } from 'react';
import { 
  Save, 
  RefreshCw, 
  Database, 
  Scan, 
  Settings as SettingsIcon,
  AlertCircle,
  CheckCircle,
  Monitor
} from 'lucide-react';
import { useApi } from '../context/ApiContext';

interface AppSettings {
  scanner: {
    resolution: number;
    colorMode: string;
    autoFeed: boolean;
    duplex: boolean;
  };
  processing: {
    confidenceThreshold: number;
    autoProcessing: boolean;
    batchSize: number;
  };
  exam: {
    defaultQuestions: number;
    passingScore: number;
    allowPartialCredit: boolean;
  };
  database: {
    connectionString: string;
    connected: boolean;
  };
}

interface DatabaseStatus {
  connected: boolean;
  status: string;
  host?: string;
  database?: string;
}

export const Settings: React.FC = () => {
  const { api } = useApi();
  
  const [settings, setSettings] = useState<AppSettings>({
    scanner: {
      resolution: 300,
      colorMode: 'grayscale',
      autoFeed: true,
      duplex: false,
    },
    processing: {
      confidenceThreshold: 0.7,
      autoProcessing: true,
      batchSize: 50,
    },
    exam: {
      defaultQuestions: 100,
      passingScore: 60,
      allowPartialCredit: false,
    },
    database: {
      connectionString: 'mongodb+srv://dani:123@cluster0.e1vm0zs.mongodb.net/',
      connected: true,
    },
  });

  const [dbStatus, setDbStatus] = useState<DatabaseStatus>({
    connected: false,
    status: 'Unknown',
  });

  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    fetchSettings();
    checkDatabaseStatus();
  }, []);

  const fetchSettings = async () => {
    try {
      const response = await api.get('/settings');
      setSettings(response.data);
    } catch (error) {
      console.error('Failed to fetch settings:', error);
    }
  };

  const checkDatabaseStatus = async () => {
    try {
      const response = await api.post('/settings/test-db');
      setDbStatus(response.data);
    } catch (error) {
      setDbStatus({
        connected: false,
        status: 'Connection failed',
      });
    }
  };

  const handleSave = async () => {
    setLoading(true);
    setError('');
    
    try {
      await api.put('/settings', settings);
      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
    } catch (error: any) {
      setError(error.response?.data?.error || 'Failed to save settings');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async () => {
    if (confirm('Are you sure you want to reset all settings to defaults?')) {
      setLoading(true);
      try {
        const response = await api.post('/settings/reset');
        setSettings(response.data.settings);
        setSuccess(true);
        setTimeout(() => setSuccess(false), 3000);
      } catch (error: any) {
        setError(error.response?.data?.error || 'Failed to reset settings');
      } finally {
        setLoading(false);
      }
    }
  };

  const updateSettings = (section: keyof AppSettings, key: string, value: any) => {
    setSettings(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: value,
      },
    }));
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Settings</h1>
        <p className="text-gray-600 mt-2">
          Configure application settings and manage system preferences
        </p>
      </div>

      {/* Status Messages */}
      {success && (
        <div className="mb-6 bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center space-x-3">
            <CheckCircle className="h-5 w-5 text-green-600" />
            <span className="text-green-800">Settings saved successfully!</span>
          </div>
        </div>
      )}

      {error && (
        <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center space-x-3">
            <AlertCircle className="h-5 w-5 text-red-600" />
            <span className="text-red-800">{error}</span>
          </div>
        </div>
      )}

      <div className="space-y-8">
        {/* Scanner Settings */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center space-x-3 mb-6">
            <div className="bg-blue-100 p-2 rounded-lg">
              <Scan className="h-5 w-5 text-blue-600" />
            </div>
            <h2 className="text-lg font-semibold text-gray-900">Scanner Settings</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Resolution (DPI)
              </label>
              <select
                value={settings.scanner.resolution}
                onChange={(e) => updateSettings('scanner', 'resolution', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value={150}>150 DPI</option>
                <option value={300}>300 DPI</option>
                <option value={600}>600 DPI</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Color Mode
              </label>
              <select
                value={settings.scanner.colorMode}
                onChange={(e) => updateSettings('scanner', 'colorMode', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="color">Color</option>
                <option value="grayscale">Grayscale</option>
                <option value="black-white">Black & White</option>
              </select>
            </div>

            <div className="flex items-center">
              <input
                type="checkbox"
                id="autoFeed"
                checked={settings.scanner.autoFeed}
                onChange={(e) => updateSettings('scanner', 'autoFeed', e.target.checked)}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <label htmlFor="autoFeed" className="ml-2 text-sm text-gray-700">
                Auto Document Feeder
              </label>
            </div>

            <div className="flex items-center">
              <input
                type="checkbox"
                id="duplex"
                checked={settings.scanner.duplex}
                onChange={(e) => updateSettings('scanner', 'duplex', e.target.checked)}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <label htmlFor="duplex" className="ml-2 text-sm text-gray-700">
                Duplex Scanning
              </label>
            </div>
          </div>
        </div>

        {/* Processing Settings */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center space-x-3 mb-6">
            <div className="bg-emerald-100 p-2 rounded-lg">
              <Monitor className="h-5 w-5 text-emerald-600" />
            </div>
            <h2 className="text-lg font-semibold text-gray-900">Processing Settings</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Confidence Threshold
              </label>
              <div className="flex items-center space-x-3">
                <input
                  type="range"
                  min="0.5"
                  max="1"
                  step="0.05"
                  value={settings.processing.confidenceThreshold}
                  onChange={(e) => updateSettings('processing', 'confidenceThreshold', parseFloat(e.target.value))}
                  className="flex-1"
                />
                <span className="text-sm text-gray-600 w-12">
                  {Math.round(settings.processing.confidenceThreshold * 100)}%
                </span>
              </div>
              <p className="text-xs text-gray-500 mt-1">
                Minimum confidence required for bubble detection
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Batch Size
              </label>
              <input
                type="number"
                min="1"
                max="100"
                value={settings.processing.batchSize}
                onChange={(e) => updateSettings('processing', 'batchSize', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
              <p className="text-xs text-gray-500 mt-1">
                Number of sheets to process in one batch
              </p>
            </div>

            <div className="flex items-center">
              <input
                type="checkbox"
                id="autoProcessing"
                checked={settings.processing.autoProcessing}
                onChange={(e) => updateSettings('processing', 'autoProcessing', e.target.checked)}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <label htmlFor="autoProcessing" className="ml-2 text-sm text-gray-700">
                Auto-process scanned sheets
              </label>
            </div>
          </div>
        </div>

        {/* Exam Settings */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center space-x-3 mb-6">
            <div className="bg-orange-100 p-2 rounded-lg">
              <SettingsIcon className="h-5 w-5 text-orange-600" />
            </div>
            <h2 className="text-lg font-semibold text-gray-900">Exam Settings</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Default Number of Questions
              </label>
              <select
                value={settings.exam.defaultQuestions}
                onChange={(e) => updateSettings('exam', 'defaultQuestions', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value={100}>100 Questions</option>
                <option value={200}>200 Questions</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Passing Score (%)
              </label>
              <input
                type="number"
                min="0"
                max="100"
                value={settings.exam.passingScore}
                onChange={(e) => updateSettings('exam', 'passingScore', parseInt(e.target.value))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            <div className="flex items-center">
              <input
                type="checkbox"
                id="allowPartialCredit"
                checked={settings.exam.allowPartialCredit}
                onChange={(e) => updateSettings('exam', 'allowPartialCredit', e.target.checked)}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <label htmlFor="allowPartialCredit" className="ml-2 text-sm text-gray-700">
                Allow Partial Credit
              </label>
            </div>
          </div>
        </div>

        {/* Database Settings */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center space-x-3 mb-6">
            <div className="bg-purple-100 p-2 rounded-lg">
              <Database className="h-5 w-5 text-purple-600" />
            </div>
            <h2 className="text-lg font-semibold text-gray-900">Database Settings</h2>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Connection String
              </label>
              <input
                type="text"
                value={settings.database.connectionString}
                onChange={(e) => updateSettings('database', 'connectionString', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="MongoDB connection string"
              />
            </div>

            <div className="bg-gray-50 rounded-lg p-4">
              <h3 className="text-sm font-medium text-gray-700 mb-3">Connection Status</h3>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Status</span>
                  <div className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${dbStatus.connected ? 'bg-green-500' : 'bg-red-500'}`}></div>
                    <span className={`text-sm ${dbStatus.connected ? 'text-green-600' : 'text-red-600'}`}>
                      {dbStatus.status}
                    </span>
                  </div>
                </div>
                {dbStatus.host && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Host</span>
                    <span className="text-sm text-gray-900">{dbStatus.host}</span>
                  </div>
                )}
                {dbStatus.database && (
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Database</span>
                    <span className="text-sm text-gray-900">{dbStatus.database}</span>
                  </div>
                )}
              </div>
              
              <button
                onClick={checkDatabaseStatus}
                className="mt-3 px-3 py-1 text-sm border border-gray-300 text-gray-700 rounded hover:bg-gray-50 transition-colors"
              >
                Test Connection
              </button>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex justify-end space-x-4">
          <button
            onClick={handleReset}
            disabled={loading}
            className="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <RefreshCw className="h-4 w-4 inline mr-2" />
            Reset to Defaults
          </button>
          
          <button
            onClick={handleSave}
            disabled={loading}
            className="flex items-center space-x-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? (
              <RefreshCw className="h-4 w-4 animate-spin" />
            ) : (
              <Save className="h-4 w-4" />
            )}
            <span>{loading ? 'Saving...' : 'Save Settings'}</span>
          </button>
        </div>
      </div>
    </div>
  );
};