import React, { useState, useEffect } from 'react';
import { 
  Download, 
  Filter, 
  Search, 
  Eye, 
  FileText,
  TrendingUp,
  Users,
  Award,
  BarChart3,
  RefreshCw
} from 'lucide-react';
import { useApi } from '../context/ApiContext';

interface StudentInfo {
  name: string;
  lockerNumber: string;
  rank: string;
  ocrConfidence: number;
  rawOcrText: string;
  ocrAvailable: boolean;
}

interface StudentResult {
  _id: string;
  examId: string;
  studentId: string;
  studentName?: string;
  examName: string;
  score: number;
  totalMarks: number;
  percentage: number;
  passFailStatus: 'Pass' | 'Fail';
  correctAnswers: number;
  incorrectAnswers: number;
  blankAnswers: number;
  multipleMarks: number;
  sponsorDS?: string;
  course?: string;
  wing?: string;
  module?: string;
  processedAt: string;
  studentInfo?: StudentInfo;
  passingPercentage?: number;
}

interface FilterOptions {
  examId: string;
  sponsorDS: string;
  course: string;
  passFailStatus: 'all' | 'pass' | 'fail';
}

export const StudentResults: React.FC = () => {
  const { api } = useApi();
  
  const [results, setResults] = useState<StudentResult[]>([]);
  const [filteredResults, setFilteredResults] = useState<StudentResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filters, setFilters] = useState<FilterOptions>({
    examId: '',
    sponsorDS: '',
    course: '',
    passFailStatus: 'all',
  });
  
  const [availableFilters, setAvailableFilters] = useState({
    exams: [] as Array<{examId: string, examName: string}>,
    sponsorDS: [] as string[],
    courses: [] as string[]
  });

  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(20);
  const [sortBy, setSortBy] = useState<'studentName' | 'percentage' | 'processedAt'>('processedAt');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  useEffect(() => {
    fetchResults();
  }, []);

  useEffect(() => {
    applyFilters();
  }, [results, searchTerm, filters, sortBy, sortOrder]);

  const fetchResults = async () => {
    setLoading(true);
    try {
      const response = await api.get('/results/all');
      const resultsData = response.data;
      setResults(resultsData);
      
      const uniqueExams = Array.from(
        new Map(
          resultsData.map((r: StudentResult) => [r.examId, { examId: r.examId, examName: r.examName }])
        ).values()
      ) as Array<{ examId: string; examName: string }>;
      const uniqueSponsorDS = Array.from(
        new Set(resultsData.map((r: StudentResult) => r.sponsorDS).filter(Boolean))
      ) as string[];
      const uniqueCourses = Array.from(
        new Set(resultsData.map((r: StudentResult) => r.course).filter(Boolean))
      ) as string[];
      
      setAvailableFilters({
        exams: uniqueExams,
        sponsorDS: uniqueSponsorDS,
        courses: uniqueCourses
      });
      
    } catch (error) {
      console.error('Failed to fetch results:', error);
    } finally {
      setLoading(false);
    }
  };

  const applyFilters = () => {
    let filtered = results.filter(result => {
      const matchesSearch = !searchTerm || 
        (result.studentName && result.studentName.toLowerCase().includes(searchTerm.toLowerCase())) ||
        result.studentId.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (result.studentInfo?.lockerNumber && result.studentInfo.lockerNumber.toLowerCase().includes(searchTerm.toLowerCase())) ||
        (result.studentInfo?.rank && result.studentInfo.rank.toLowerCase().includes(searchTerm.toLowerCase()));
      
      const matchesSponsorDS = !filters.sponsorDS || result.sponsorDS === filters.sponsorDS;
      const matchesCourse = !filters.course || result.course === filters.course;
      const matchesPassFail = filters.passFailStatus === 'all' || 
        (filters.passFailStatus === 'pass' && result.passFailStatus === 'Pass') ||
        (filters.passFailStatus === 'fail' && result.passFailStatus === 'Fail');
      const matchesExam = !filters.examId || result.examId === filters.examId;
      
      return matchesSearch && matchesSponsorDS && matchesCourse && matchesPassFail && matchesExam;
    });

    filtered.sort((a, b) => {
      let aValue, bValue;
      
      switch (sortBy) {
        case 'studentName':
          aValue = a.studentName || a.studentInfo?.name || '';
          bValue = b.studentName || b.studentInfo?.name || '';
          break;
        case 'percentage':
          aValue = a.percentage;
          bValue = b.percentage;
          break;
        case 'processedAt':
          aValue = new Date(a.processedAt).getTime();
          bValue = new Date(b.processedAt).getTime();
          break;
        default:
          return 0;
      }
      
      if (typeof aValue === 'string' && typeof bValue === 'string') {
        return sortOrder === 'asc' 
          ? aValue.localeCompare(bValue)
          : bValue.localeCompare(aValue);
      } else if (typeof aValue === 'number' && typeof bValue === 'number') {
        return sortOrder === 'asc' 
          ? aValue - bValue
          : bValue - aValue;
      } else {
        return 0;
      }
    });

    setFilteredResults(filtered);
    setCurrentPage(1);
  };

  const downloadAllResultsPDF = async () => {
    try {
      const response = await api.post('/results/download-all-pdf', {
        results: filteredResults,
        filters: filters
      }, {
        responseType: 'blob',
      });

      const blob = new Blob([response.data], { type: 'application/pdf' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `OMR_Results_Report_${new Date().toISOString().split('T')[0]}.pdf`;
      link.click();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('PDF download failed:', error);
    }
  };

  const resetFilters = () => {
    setFilters({
      examId: '',
      sponsorDS: '',
      course: '',
      passFailStatus: 'all',
    });
    setSearchTerm('');
  };

  const stats = {
    total: filteredResults.length,
    passed: filteredResults.filter(r => r.passFailStatus === 'Pass').length,
    failed: filteredResults.filter(r => r.passFailStatus === 'Fail').length,
    averagePercentage: filteredResults.length > 0 
      ? filteredResults.reduce((sum, r) => sum + r.percentage, 0) / filteredResults.length 
      : 0
  };

  const totalPages = Math.ceil(filteredResults.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const paginatedResults = filteredResults.slice(startIndex, startIndex + itemsPerPage);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Student Results</h1>
        <p className="text-gray-600 mt-2">
          View and filter all student results with comprehensive analytics
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Students</p>
              <p className="text-2xl font-bold text-gray-900">{stats.total}</p>
            </div>
            <div className="bg-blue-50 p-3 rounded-lg">
              <Users className="h-6 w-6 text-blue-600" />
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Passed</p>
              <p className="text-2xl font-bold text-green-600">{stats.passed}</p>
            </div>
            <div className="bg-green-50 p-3 rounded-lg">
              <Award className="h-6 w-6 text-green-600" />
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Failed</p>
              <p className="text-2xl font-bold text-red-600">{stats.failed}</p>
            </div>
            <div className="bg-red-50 p-3 rounded-lg">
              <BarChart3 className="h-6 w-6 text-red-600" />
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Average Score</p>
              <p className="text-2xl font-bold text-gray-900">{stats.averagePercentage.toFixed(1)}%</p>
            </div>
            <div className="bg-orange-50 p-3 rounded-lg">
              <TrendingUp className="h-6 w-6 text-orange-600" />
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 mb-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900">Filters & Search</h2>
          <div className="flex space-x-3">
            <button
              onClick={resetFilters}
              className="flex items-center space-x-2 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
            >
              <RefreshCw className="h-4 w-4" />
              <span>Reset</span>
            </button>
            <button
              onClick={downloadAllResultsPDF}
              disabled={filteredResults.length === 0}
              className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Download className="h-4 w-4" />
              <span>Download PDF</span>
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Search Students
            </label>
            <div className="relative">
              <Search className="h-5 w-5 text-gray-400 absolute left-3 top-1/2 transform -translate-y-1/2" />
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Search by name, ID, locker, or rank"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Exam
            </label>
            <select
              value={filters.examId}
              onChange={(e) => setFilters(prev => ({ ...prev, examId: e.target.value }))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="">All Exams</option>
              {availableFilters.exams.map(exam => (
                <option key={exam.examId} value={exam.examId}>
                  {exam.examName}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Sponsor DS
            </label>
            <select
              value={filters.sponsorDS}
              onChange={(e) => setFilters(prev => ({ ...prev, sponsorDS: e.target.value }))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="">All Sponsor DS</option>
              {availableFilters.sponsorDS.map(sponsor => (
                <option key={sponsor} value={sponsor}>
                  {sponsor}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Course
            </label>
            <select
              value={filters.course}
              onChange={(e) => setFilters(prev => ({ ...prev, course: e.target.value }))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="">All Courses</option>
              {availableFilters.courses.map(course => (
                <option key={course} value={course}>
                  {course}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Result
            </label>
            <select
              value={filters.passFailStatus}
              onChange={(e) => setFilters(prev => ({ ...prev, passFailStatus: e.target.value as any }))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="all">All Results</option>
              <option value="pass">Pass Only</option>
              <option value="fail">Fail Only</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Sort By
            </label>
            <div className="flex space-x-2">
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as any)}
                className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="processedAt">Date</option>
                <option value="studentName">Name</option>
                <option value="percentage">Score</option>
              </select>
              <button
                onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                className="px-3 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
              >
                {sortOrder === 'asc' ? '↑' : '↓'}
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-gray-900">
              Results ({filteredResults.length} of {results.length})
            </h2>
          </div>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          </div>
        ) : filteredResults.length === 0 ? (
          <div className="text-center py-12">
            <BarChart3 className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500">No results found</p>
            <p className="text-sm text-gray-400">Try adjusting your filters</p>
          </div>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50 border-b border-gray-200">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Student Details
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Exam
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Score
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Percentage
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Result
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Locker
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Rank
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Sponsor DS
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Course
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Date
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {paginatedResults.map((result, index) => (
                    <tr key={result._id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div>
                          <div className="text-sm font-medium text-gray-900">
                            {result.studentName || result.studentInfo?.name || 'Unknown'}
                          </div>
                          <div className="text-sm text-gray-500">ID: {result.studentId}</div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-900">{result.examName}</div>
                        <div className="text-sm text-gray-500">{result.wing} - {result.module}</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className="text-sm font-semibold text-gray-900">
                          {result.score}/{result.totalMarks}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`text-sm font-medium ${
                          result.percentage >= (result.passingPercentage || 60)
                            ? 'text-green-600'
                            : 'text-red-600'
                        }`}>
                          {result.percentage.toFixed(1)}%
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                          result.passFailStatus === 'Pass'
                            ? 'bg-green-100 text-green-800'
                            : 'bg-red-100 text-red-800'
                        }`}>
                          {result.passFailStatus}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {result.studentInfo?.lockerNumber || '-'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {result.studentInfo?.rank || '-'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {result.sponsorDS || '-'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        {result.course || '-'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {new Date(result.processedAt).toLocaleDateString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {totalPages > 1 && (
              <div className="px-6 py-4 border-t border-gray-200 flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                    disabled={currentPage === 1}
                    className="px-3 py-1 text-sm border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Previous
                  </button>
                  <span className="text-sm text-gray-600">
                    Page {currentPage} of {totalPages}
                  </span>
                  <button
                    onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
                    disabled={currentPage === totalPages}
                    className="px-3 py-1 text-sm border border-gray-300 rounded hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Next
                  </button>
                </div>
                <div className="text-sm text-gray-600">
                  Showing {startIndex + 1} to {Math.min(startIndex + itemsPerPage, filteredResults.length)} of {filteredResults.length} results
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};