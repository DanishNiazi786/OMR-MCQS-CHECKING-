import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Navigation } from './components/Navigation';
import { Dashboard } from './pages/Dashboard';
import { ExamCreation } from './pages/ExamCreation';
import { OMRManagement } from './pages/OMRManagement';
import { ScanProcess } from './pages/ScanProcess';
import { StudentResults } from './pages/StudentResults';
import { Settings } from './pages/Settings';
import { Login } from './pages/Login';
import { ApiProvider } from './context/ApiContext';

// Simple authentication state
const isAuthenticated = () => {
  return localStorage.getItem('isAuthenticated') === 'true';
};

// ProtectedRoute component
const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return isAuthenticated() ? <>{children}</> : <Navigate to="/login" replace />;
};

function App() {
  return (
    <ApiProvider>
      <Router>
        <div className="min-h-screen bg-gray-50">
          {isAuthenticated() && <Navigation />}
          <main className={isAuthenticated() ? 'pt-16' : ''}>
            <Routes>
              <Route path="/login" element={<Login />} />
              <Route
                path="/"
                element={
                  <ProtectedRoute>
                    <Dashboard />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/exam-creation"
                element={
                  <ProtectedRoute>
                    <ExamCreation />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/omr-management"
                element={
                  <ProtectedRoute>
                    <OMRManagement />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/scan-process"
                element={
                  <ProtectedRoute>
                    <ScanProcess />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/student-results"
                element={
                  <ProtectedRoute>
                    <StudentResults />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/settings"
                element={
                  <ProtectedRoute>
                    <Settings />
                  </ProtectedRoute>
                }
              />
              <Route path="*" element={<Navigate to={isAuthenticated() ? "/" : "/login"} replace />} />
            </Routes>
          </main>
        </div>
      </Router>
    </ApiProvider>
  );
}

export default App;