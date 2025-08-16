import React, { useState, useEffect, useContext, createContext } from 'react';
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

interface AuthContextType {
  isAuth: boolean;
  setIsAuth: React.Dispatch<React.SetStateAction<boolean>>;
}

export const AuthContext = createContext<AuthContextType | undefined>(undefined);

// ProtectedRoute component
const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const authContext = useContext(AuthContext);
  if (!authContext) {
    throw new Error('ProtectedRoute must be used within an AuthContext.Provider');
  }
  const isAuthenticated = authContext.isAuth;
  return isAuthenticated ? <>{children}</> : <Navigate to="/login" replace />;
};

function App() {
  const [isAuth, setIsAuth] = useState(() => sessionStorage.getItem('isAuthenticated') === 'true');

  useEffect(() => {
    const handleStorageChange = () => {
      setIsAuth(sessionStorage.getItem('isAuthenticated') === 'true');
    };
    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  return (
    <AuthContext.Provider value={{ isAuth, setIsAuth }}>
      <ApiProvider>
        <Router>
          <div className="min-h-screen bg-gray-50">
            {isAuth && <Navigation />}
            <main className={isAuth ? 'pt-16' : ''}>
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
                <Route path="*" element={<Navigate to={isAuth ? "/" : "/login"} replace />} />
              </Routes>
            </main>
          </div>
        </Router>
      </ApiProvider>
    </AuthContext.Provider>
  );
}

export default App;