import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { 
  Plus, 
  Scan, 
  FileText, 
  Users, 
  TrendingUp, 
  Clock,
  Award,
  AlertCircle 
} from 'lucide-react';
import { useApi } from '../context/ApiContext';

interface Exam {
  examId: string;
  name: string;
  numQuestions: number;
  createdAt: string;
}

interface DashboardStats {
  totalExams: number;
  totalStudents: number;
  recentScans: number;
  averageScore: number;
}

export const Dashboard: React.FC = () => {
  const { api } = useApi();
  const [recentExams, setRecentExams] = useState<Exam[]>([]);
  const [stats, setStats] = useState<DashboardStats>({
    totalExams: 0,
    totalStudents: 0,
    recentScans: 0,
    averageScore: 0,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      
      // Fetch recent exams
      const examsResponse = await api.get('/exams');
      const exams = examsResponse.data.slice(0, 5); // Get last 5 exams
      setRecentExams(exams);

      // Calculate basic stats
      const totalExams = exams.length;
      let totalStudents = 0;
      let totalScore = 0;
      let studentCount = 0;

      // Fetch results for each exam to calculate stats
      for (const exam of exams) {
        try {
          const resultsResponse = await api.get(`/results/exam/${exam.examId}`);
          const examStudents = resultsResponse.data.responses?.length || 0;
          totalStudents += examStudents;

          if (resultsResponse.data.responses) {
            resultsResponse.data.responses.forEach((response: any) => {
              totalScore += response.accuracy;
              studentCount++;
            });
          }
        } catch (error) {
          // Skip if no results yet
        }
      }

      setStats({
        totalExams,
        totalStudents,
        recentScans: totalStudents,
        averageScore: studentCount > 0 ? totalScore / studentCount : 0,
      });

    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const statCards = [
    {
      title: 'Total Exams',
      value: stats.totalExams,
      icon: FileText,
      color: 'bg-blue-500',
      textColor: 'text-blue-600',
      bgColor: 'bg-blue-50',
    },
    {
      title: 'Students Processed',
      value: stats.totalStudents,
      icon: Users,
      color: 'bg-emerald-500',
      textColor: 'text-emerald-600',
      bgColor: 'bg-emerald-50',
    },
    {
      title: 'Recent Scans',
      value: stats.recentScans,
      icon: Scan,
      color: 'bg-orange-500',
      textColor: 'text-orange-600',
      bgColor: 'bg-orange-50',
    },
    {
      title: 'Average Score',
      value: `${Math.round(stats.averageScore)}%`,
      icon: Award,
      color: 'bg-purple-500',
      textColor: 'text-purple-600',
      bgColor: 'bg-purple-50',
    },
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600 mt-2">
          Welcome to EFSoft OMR Software - Your complete solution for optical mark recognition
        </p>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        <Link
          to="/exam-creation"
          className="bg-gradient-to-br from-blue-500 to-blue-600 p-6 rounded-xl text-white hover:from-blue-600 hover:to-blue-700 transition-all duration-200 transform hover:scale-105 shadow-lg"
        >
          <div className="flex items-center space-x-4">
            <div className="bg-white/20 p-3 rounded-lg">
              <Plus className="h-8 w-8" />
            </div>
            <div>
              <h3 className="text-lg font-semibold">Create New Exam</h3>
              <p className="text-blue-100">Set up answer keys and exam details</p>
            </div>
          </div>
        </Link>

        <Link
          to="/scan-process"
          className="bg-gradient-to-br from-emerald-500 to-emerald-600 p-6 rounded-xl text-white hover:from-emerald-600 hover:to-emerald-700 transition-all duration-200 transform hover:scale-105 shadow-lg"
        >
          <div className="flex items-center space-x-4">
            <div className="bg-white/20 p-3 rounded-lg">
              <Scan className="h-8 w-8" />
            </div>
            <div>
              <h3 className="text-lg font-semibold">Start Scanning</h3>
              <p className="text-emerald-100">Process answer sheets with scanner</p>
            </div>
          </div>
        </Link>

        <Link
          to="/student-results"
          className="bg-gradient-to-br from-orange-500 to-orange-600 p-6 rounded-xl text-white hover:from-orange-600 hover:to-orange-700 transition-all duration-200 transform hover:scale-105 shadow-lg"
        >
          <div className="flex items-center space-x-4">
            <div className="bg-white/20 p-3 rounded-lg">
              <TrendingUp className="h-8 w-8" />
            </div>
            <div>
              <h3 className="text-lg font-semibold">Student Results</h3>
              <p className="text-orange-100">Analyze and export reports</p>
            </div>
          </div>
        </Link>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {statCards.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <div key={index} className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">{stat.title}</p>
                  <p className="text-2xl font-bold text-gray-900 mt-1">{stat.value}</p>
                </div>
                <div className={`${stat.bgColor} p-3 rounded-lg`}>
                  <Icon className={`h-6 w-6 ${stat.textColor}`} />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Recent Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Recent Exams */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200">
          <div className="p-6 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900">Recent Exams</h2>
          </div>
          <div className="p-6">
            {recentExams.length > 0 ? (
              <div className="space-y-4">
                {recentExams.map((exam) => (
                  <div
                    key={exam.examId}
                    className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                  >
                    <div className="flex items-center space-x-3">
                      <div className="bg-blue-100 p-2 rounded-lg">
                        <FileText className="h-5 w-5 text-blue-600" />
                      </div>
                      <div>
                        <h3 className="font-medium text-gray-900">{exam.name}</h3>
                        <p className="text-sm text-gray-500">
                          {exam.numQuestions} questions
                        </p>
                      </div>
                    </div>
                    <div className="text-sm text-gray-500">
                      <Clock className="h-4 w-4 inline mr-1" />
                      {new Date(exam.createdAt).toLocaleDateString()}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8">
                <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-500">No exams created yet</p>
                <Link
                  to="/exam-creation"
                  className="text-blue-600 hover:text-blue-700 font-medium"
                >
                  Create your first exam
                </Link>
              </div>
            )}
          </div>
        </div>

        {/* System Status */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-200">
          <div className="p-6 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900">System Status</h2>
          </div>
          <div className="p-6 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-600">Database Connection</span>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span className="text-sm text-green-600">Connected</span>
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-600">Scanner Status</span>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span className="text-sm text-green-600">Ready</span>
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-600">Processing Engine</span>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span className="text-sm text-green-600">Operational</span>
              </div>
            </div>

            <div className="pt-4 border-t border-gray-200">
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <div className="flex items-start space-x-3">
                  <AlertCircle className="h-5 w-5 text-blue-600 mt-0.5" />
                  <div>
                    <h4 className="text-sm font-medium text-blue-900">System Ready</h4>
                    <p className="text-sm text-blue-700">
                      All systems operational. Ready to process OMR sheets.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};