import React, { useState, useEffect } from 'react';
import { 
  Scan, 
  Play, 
  Square, 
  CheckCircle, 
  AlertTriangle,
  FileImage,
  RefreshCw,
  Download,
  Filter
} from 'lucide-react';
import { useApi } from '../context/ApiContext';
import axios from 'axios';

interface Exam {
  examId: string;
  name: string;
  dateTime: string;
  time: string;
  numQuestions: number;
  marksPerMcq: number;
  passingPercentage: number;
  wing: string;
  course: string;
  module: string;
  sponsorDS: string;
  instructions: string;
  studentsUploaded: boolean;
  solutionUploaded: boolean;
  createdAt: string;
  createdBy: string;
  answerKey?: string[];
}

interface SolutionItem {
  question: number;
  answer: string;
}

interface ScanProgress {
  currentSheet: number;
  totalSheets: number;
  processed: number;
  errors: number;
  status: 'idle' | 'processing' | 'completed' | 'error';
}

interface ProcessingResult {
  studentId: string;
  studentName: string;
  rank: string;
  lockerNumber: string;
  score: number;
  totalMarks: number;
  percentage: number;
  accuracy: number;
  confidence: number;
  status: 'success' | 'warning' | 'error';
  passFailStatus: 'Pass' | 'Fail';
  issues?: string[];
  responses: string[];
  correctAnswers: number;
  incorrectAnswers: number;
  blankAnswers: number;
  invalidAnswers: number;
}

export const ScanProcess: React.FC = () => {
  const { api } = useApi();
  
  const [exams, setExams] = useState<Exam[]>([]);
  const [selectedExam, setSelectedExam] = useState<string>('');
  const [isScanning, setIsScanning] = useState(false);
  const [progress, setProgress] = useState<ScanProgress>({
    currentSheet: 0,
    totalSheets: 0,
    processed: 0,
    errors: 0,
    status: 'idle',
  });
  const [results, setResults] = useState<ProcessingResult[]>([]);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [filterStatus, setFilterStatus] = useState<'all' | 'pass' | 'fail'>('all');
  const [error, setError] = useState<string>('');
  const [success, setSuccess] = useState<string>('');
  const [solution, setSolution] = useState<SolutionItem[]>([]);

  useEffect(() => {
    fetchExams();
  }, []);

  const fetchExams = async () => {
    try {
      const response = await api.get('/exams');
      const examsWithSolutions = response.data.filter((exam: Exam) => exam.solutionUploaded);
      setExams(examsWithSolutions);
    } catch (error) {
      console.error('Failed to fetch exams:', error);
      setError('Failed to fetch exams. Please try again.');
    }
  };

  const fetchSolution = async (examId: string) => {
    try {
      const response = await api.get(`/solutions/${examId}`);
      setSolution(response.data.solutions);
      setExams(prev => prev.map(exam => 
        exam.examId === examId ? { ...exam, answerKey: response.data.solutions.map((sol: SolutionItem) => sol.answer) } : exam
      ));
    } catch (error) {
      console.error('Failed to fetch solution:', error);
      setError('Failed to fetch solution. Please try again.');
    }
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    const imageFiles = files.filter(file => file.type.startsWith('image/'));
    setSelectedFiles(imageFiles);
  };

  const startBatchScanning = async () => {
    if (!selectedExam) {
      setError('Please select an exam');
      return;
    }

    if (selectedFiles.length === 0) {
      setError('Please select answer sheet images to process');
      return;
    }

    await fetchSolution(selectedExam);

    setIsScanning(true);
    setError('');
    setSuccess('');
    setResults([]);
    
    const initialProgress = {
      currentSheet: 0,
      totalSheets: selectedFiles.length,
      processed: 0,
      errors: 0,
      status: 'processing' as const,
    };
    setProgress(initialProgress);

    let processedCount = 0;
    let errorCount = 0;
    const processedResults: ProcessingResult[] = [];

    try {
      const selectedExamData = exams.find(exam => exam.examId === selectedExam);
      if (!selectedExamData) {
        throw new Error('Selected exam not found');
      }

      for (let i = 0; i < selectedFiles.length; i++) {
        const file = selectedFiles[i];
        
        setProgress(prev => ({
          ...prev,
          currentSheet: i + 1,
          status: 'processing'
        }));

        try {
          const result = await processAnswerSheet(file, selectedExamData, i + 1);
          processedResults.push(result);
          processedCount++;
          
          setProgress(prev => ({
            ...prev,
            processed: processedCount,
            errors: errorCount
          }));

        } catch (error: any) {
          console.error(`Failed to process sheet ${i + 1}:`, error);
          errorCount++;
          
          setProgress(prev => ({
            ...prev,
            processed: processedCount,
            errors: errorCount
          }));
          
          console.error(`Sheet ${i + 1} processing failed: ${error.message}`);
        }

        await new Promise(resolve => setTimeout(resolve, 500));
      }

      setProgress(prev => ({ 
        ...prev, 
        status: 'completed',
        processed: processedCount,
        errors: errorCount
      }));

      if (processedCount > 0) {
        setSuccess(`Successfully processed ${processedCount} out of ${selectedFiles.length} answer sheets`);
        await publishResults(processedResults, selectedExamData);
      }
      
      if (errorCount > 0) {
        setError(`${errorCount} sheets failed to process. Check the results below.`);
      }

    } catch (error: any) {
      console.error('Batch scanning failed:', error);
      setError(error.message || 'Failed to process answer sheets');
      setProgress(prev => ({ 
        ...prev, 
        status: 'error',
        processed: processedCount,
        errors: errorCount
      }));
    } finally {
      setIsScanning(false);
    }
  };

  const processAnswerSheet = async (file: File, examData: Exam, sheetNumber: number): Promise<ProcessingResult> => {
    const formData = new FormData();
    formData.append('image', file);
    formData.append('examId', selectedExam);
    formData.append('studentId', `STUDENT_${sheetNumber.toString().padStart(3, '0')}`);

    try {
      const response = await api.post('/scan/process', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const responseData = response.data.response;
      const totalMarks = examData.numQuestions * examData.marksPerMcq;
      let correctAnswers = 0;
      let incorrectAnswers = 0;
      let blankAnswers = 0;
      let invalidAnswers = 0;

      const responses = responseData.responses;
      solution.forEach((sol, index) => {
        const response = responses[index] || '';
        if (response === '') {
          blankAnswers++;
        } else if (response === sol.answer) {
          correctAnswers++;
        } else if (response === 'MULTIPLE' || response === 'INVALID') {
          invalidAnswers++;
        } else {
          incorrectAnswers++;
        }
      });

      const score = correctAnswers * examData.marksPerMcq;
      const percentage = totalMarks > 0 ? (score / totalMarks) * 100 : 0;
      const passFailStatus = percentage >= examData.passingPercentage ? 'Pass' : 'Fail';

      const result: ProcessingResult = {
        studentId: responseData.studentId,
        studentName: responseData.studentName || `Student ${sheetNumber}`,
        rank: responseData.rank || 'N/A',
        lockerNumber: responseData.lockerNumber || 'N/A',
        score,
        totalMarks,
        percentage,
        accuracy: responseData.accuracy,
        confidence: responseData.processingMetadata.confidence,
        status: responseData.processingMetadata.confidence > 70 ? 'success' : responseData.processingMetadata.confidence > 50 ? 'warning' : 'error',
        passFailStatus,
        responses,
        correctAnswers,
        incorrectAnswers,
        blankAnswers,
        invalidAnswers,
        issues: responseData.processingMetadata.confidence < 70 ? ['Low confidence detection'] : [],
      };

      setResults(prev => [...prev, result]);
      
      await saveResultToDatabase(result, examData);
      
      return result;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || error.message || 'Processing failed');
    }
  };

  const saveResultToDatabase = async (result: ProcessingResult, examData: Exam) => {
    try {
      const resultData = {
        examId: selectedExam,
        studentId: result.studentId,
        studentName: result.studentName,
        rank: result.rank || null,
        lockerNumber: result.lockerNumber || null,
        examName: examData.name,
        score: Number(result.score),
        totalMarks: Number(result.totalMarks),
        percentage: Number(result.percentage),
        passFailStatus: result.passFailStatus,
        responses: result.responses.map(resp => resp === null ? null : String(resp)),
        correctAnswers: Number(result.correctAnswers),
        incorrectAnswers: Number(result.incorrectAnswers),
        blankAnswers: Number(result.blankAnswers),
        multipleMarks: Number(result.invalidAnswers),
        sponsorDS: examData.sponsorDS || null,
        course: examData.course || null,
        wing: examData.wing || null,
        module: examData.module || null,
      };
      console.log("Saving result to database:", JSON.stringify(resultData, null, 2));
      console.log("Request URL:", `${api.defaults.baseURL || 'http://localhost:3001'}/api/results/save`);
      const response = await api.post('/api/results/save', resultData);
      console.log(`Result saved for student ${result.studentId}:`, response.data);
    } catch (error: any) {
      console.error('Failed to save result to database:', {
        url: `${api.defaults.baseURL || 'http://localhost:3001'}/api/results/save`,
        error: error.response?.data || error.message,
        status: error.response?.status,
      });
      setError(`Failed to save result: ${error.response?.data?.detail || error.message}`);
    }
  };

  const publishResults = async (results: ProcessingResult[], examData: Exam) => {
    try {
      const publishData = {
        examId: selectedExam,
        examName: examData.name,
        results: results.map(result => ({
          studentId: result.studentId,
          studentName: result.studentName,
          rank: result.rank,
          lockerNumber: result.lockerNumber,
          score: result.score,
          totalMarks: result.totalMarks,
          percentage: result.percentage,
          passFailStatus: result.passFailStatus,
          correctAnswers: result.correctAnswers,
          incorrectAnswers: result.incorrectAnswers,
          blankAnswers: result.blankAnswers,
          multipleMarks: result.invalidAnswers,
          responses: result.responses
        }))
      };
      console.log("Publishing results:", JSON.stringify(publishData, null, 2));
      console.log("Request URL:", `${api.defaults.baseURL || 'http://localhost:3001'}/results/publish`);
      await api.post('/results/publish', publishData);
      setSuccess(prev => `${prev} | Results published successfully!`);
    } catch (error: any) {
      console.error('Failed to publish results:', error);
      setError('Failed to publish results. Please try again.');
    }
  };

  const downloadResultsPDF = async () => {
    if (results.length === 0) {
      setError('No results to download');
      return;
    }

    if (selectedFiles.length !== filteredResults.length) {
      setError(`Mismatch: ${selectedFiles.length} files selected but ${filteredResults.length} results available`);
      return;
    }

    try {
      const selectedExamData = exams.find(exam => exam.examId === selectedExam);
      
      const pdfData = {
        examId: selectedExam,
        examName: selectedExamData?.name || 'Exam Results',
        results: filteredResults.map(result => ({
          studentId: result.studentId,
          studentName: result.studentName,
          rank: result.rank || 'N/A',
          lockerNumber: result.lockerNumber || 'N/A',
          score: Number(result.score),
          totalMarks: Number(result.totalMarks),
          percentage: Number(result.percentage),
          passFailStatus: result.passFailStatus,
          correctAnswers: Number(result.correctAnswers),
          incorrectAnswers: Number(result.incorrectAnswers),
          blankAnswers: Number(result.blankAnswers),
          multipleMarks: Number(result.invalidAnswers),
        })),
      };
      console.log("Downloading PDF with data:", JSON.stringify(pdfData, null, 2));
      console.log("Files:", selectedFiles.map(file => file.name));
      console.log("Request URL:", `${api.defaults.baseURL || 'http://localhost:3001'}/api/results/download-batch-pdf`);

      const formData = new FormData();
      formData.append('request_data', JSON.stringify(pdfData));
      selectedFiles.forEach((file, index) => {
        formData.append('images', file);
      });

      const response = await api.post('/api/results/download-batch-pdf', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        responseType: 'blob',
      });

      const blob = new Blob([response.data], { type: 'application/pdf' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${selectedExamData?.name || 'Exam_Results'}_Batch_Results_${new Date().toISOString().split('T')[0]}.pdf`;
      link.click();
      window.URL.revokeObjectURL(url);
      
      setSuccess('PDF downloaded successfully!');
    } catch (error: any) {
      console.error('PDF download failed:', {
        url: `${api.defaults.baseURL || 'http://localhost:3001'}/api/results/download-batch-pdf`,
        error: error.response?.data || error.message,
        status: error.response?.status,
      });
      setError(`Failed to download PDF: ${error.response?.data?.detail || error.message}`);
    }
  };

  const stopScanning = () => {
    setIsScanning(false);
    setProgress(prev => ({ ...prev, status: 'idle' }));
  };

  const resetSession = () => {
    setProgress({
      currentSheet: 0,
      totalSheets: 0,
      processed: 0,
      errors: 0,
      status: 'idle',
    });
    setResults([]);
    setSelectedFiles([]);
    setError('');
    setSuccess('');
    setSolution([]);
    const fileInput = document.getElementById('file-upload') as HTMLInputElement;
    if (fileInput) {
      fileInput.value = '';
    }
  };

  const filteredResults = results.filter(result => {
    if (filterStatus === 'pass') return result.passFailStatus === 'Pass';
    if (filterStatus === 'fail') return result.passFailStatus === 'Fail';
    return true;
  });

  const selectedExamData = exams.find(exam => exam.examId === selectedExam);

  const resultSummary = results.length > 0
    ? `Processed: ${progress.processed}, Errors: ${progress.errors}, Pass: ${results.filter(r => r.passFailStatus === 'Pass').length}, Fail: ${results.filter(r => r.passFailStatus === 'Fail').length}`
    : 'No results yet';

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex flex-col min-h-screen">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Bubble Sheet Scanning</h1>
        <p className="text-gray-600 mt-2">
          Upload and process multiple answer sheets with advanced validation
        </p>
      </div>

      {success && (
        <div className="mb-6 bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center space-x-3">
            <CheckCircle className="h-5 w-5 text-green-600" />
            <span className="text-green-800">{success}</span>
          </div>
        </div>
      )}

      {error && (
        <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center space-x-3">
            <AlertTriangle className="h-5 w-5 text-red-600" />
            <span className="text-red-800">{error}</span>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 flex-grow">
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Exam Configuration</h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Select Exam
                </label>
                <select
                  value={selectedExam}
                  onChange={(e) => {
                    setSelectedExam(e.target.value);
                    if (e.target.value) {
                      fetchSolution(e.target.value);
                    }
                  }}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  disabled={isScanning}
                >
                  <option value="">Choose an exam...</option>
                  {exams.map(exam => (
                    <option key={exam.examId} value={exam.examId}>
                      {exam.name} ({exam.numQuestions} questions)
                    </option>
                  ))}
                </select>
              </div>

              {selectedExamData && (
                <div className="bg-gray-50 rounded-lg p-4">
                  <h3 className="text-sm font-medium text-gray-700 mb-2">Exam Details</h3>
                  <div className="space-y-1 text-sm text-gray-600">
                    <div>Questions: {selectedExamData.numQuestions}</div>
                    <div>Marks per MCQ: {selectedExamData.marksPerMcq}</div>
                    <div>Total Marks: {selectedExamData.numQuestions * selectedExamData.marksPerMcq}</div>
                    <div>Passing %: {selectedExamData.passingPercentage}%</div>
                    <div>Sponsor DS: {selectedExamData.sponsorDS}</div>
                    <div>Course: {selectedExamData.course}</div>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Upload Answer Sheets</h2>
            
            <div className="space-y-4">
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-500 transition-colors">
                <input
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={handleFileSelect}
                  className="hidden"
                  id="file-upload"
                  disabled={isScanning}
                />
                <label htmlFor="file-upload" className={`cursor-pointer ${isScanning ? 'cursor-not-allowed opacity-50' : ''}`}>
                  <FileImage className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                  <p className="text-sm text-gray-600">
                    Click to upload answer sheet images
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    PNG, JPG up to 10MB each (Multiple files supported)
                  </p>
                </label>
              </div>
              
              {selectedFiles.length > 0 && (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                  <div className="flex items-center space-x-3">
                    <FileImage className="h-5 w-5 text-blue-600" />
                    <div>
                      <p className="text-sm font-medium text-blue-900">
                        {selectedFiles.length} file(s) selected
                      </p>
                      <p className="text-xs text-blue-700">
                        Total size: {(selectedFiles.reduce((sum, file) => sum + file.size, 0) / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Validation Rules</h2>
            <div className="space-y-3 text-sm text-gray-600">
              <div className="flex items-start space-x-2">
                <CheckCircle className="h-4 w-4 text-green-600 mt-0.5" />
                <span>Single bubble selection per question</span>
              </div>
              <div className="flex items-start space-x-2">
                <AlertTriangle className="h-4 w-4 text-yellow-600 mt-0.5" />
                <span>Unattempted questions marked as blank</span>
              </div>
              <div className="flex items-start space-x-2">
                <AlertTriangle className="h-4 w-4 text-red-600 mt-0.5" />
                <span>Multiple bubbles marked as invalid</span>
              </div>
              <div className="flex items-start space-x-2">
                <AlertTriangle className="h-4 w-4 text-red-600 mt-0.5" />
                <span>Partial fills treated as invalid</span>
              </div>
              <div className="flex items-start space-x-2">
                <AlertTriangle className="h-4 w-4 text-red-600 mt-0.5" />
                <span>Invalid marks (ticks, crosses) rejected</span>
              </div>
            </div>
          </div>
        </div>

        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold text-gray-900">Processing Controls</h2>
              
              <div className="flex space-x-3">
                {!isScanning ? (
                  <button
                    onClick={startBatchScanning}
                    disabled={!selectedExam || selectedFiles.length === 0}
                    className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                  >
                    <Play className="h-4 w-4" />
                    <span>Start Processing</span>
                  </button>
                ) : (
                  <button
                    onClick={stopScanning}
                    className="flex items-center space-x-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                  >
                    <Square className="h-4 w-4" />
                    <span>Stop</span>
                  </button>
                )}
                
                <button
                  onClick={resetSession}
                  disabled={isScanning}
                  className="flex items-center space-x-2 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  <RefreshCw className="h-4 w-4" />
                  <span>Reset</span>
                </button>

                {results.length > 0 && !isScanning && (
                  <button
                    onClick={downloadResultsPDF}
                    className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    <Download className="h-4 w-4" />
                    <span>Download PDF</span>
                  </button>
                )}
              </div>
            </div>

            {progress.status !== 'idle' && (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-700">
                    {progress.status === 'processing' && `Processing sheet ${progress.currentSheet} of ${progress.totalSheets}...`}
                    {progress.status === 'completed' && 'Processing Completed'}
                    {progress.status === 'error' && 'Processing Error'}
                  </span>
                  <span className="text-sm text-gray-500">
                    {progress.processed} processed, {progress.errors} errors
                  </span>
                </div>
                
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full transition-all duration-300 ${
                      progress.status === 'error' ? 'bg-red-500' : 
                      progress.status === 'completed' ? 'bg-green-500' : 'bg-blue-500'
                    }`}
                    style={{ 
                      width: progress.totalSheets > 0 ? `${((progress.processed + progress.errors) / progress.totalSheets) * 100}%` : '0%'
                    }}
                  ></div>
                </div>
                
                {isScanning && (
                  <div className="flex items-center space-x-3 text-sm text-gray-600">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                    <span>Analyzing answer sheets...</span>
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold text-gray-900">Processing Results</h2>
              
              {results.length > 0 && (
                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-2">
                    <Filter className="h-4 w-4 text-gray-500" />
                    <select
                      value={filterStatus}
                      onChange={(e) => setFilterStatus(e.target.value as any)}
                      className="px-3 py-1 border border-gray-300 rounded text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    >
                      <option value="all">All Results</option>
                      <option value="pass">Pass Only</option>
                      <option value="fail">Fail Only</option>
                    </select>
                  </div>
                </div>
              )}
            </div>
            
            {filteredResults.length === 0 ? (
              <div className="text-center py-8">
                <Scan className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-500">No results yet</p>
                <p className="text-sm text-gray-400">Start processing to see results here</p>
              </div>
            ) : (
              <div className="space-y-4">
                {filteredResults.map((result, index) => (
                  <div
                    key={index}
                    className={`border rounded-lg p-4 ${
                      result.status === 'success' ? 'border-green-200 bg-green-50' :
                      result.status === 'warning' ? 'border-yellow-200 bg-yellow-50' :
                      'border-red-200 bg-red-50'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-3">
                        {result.status === 'success' && <CheckCircle className="h-5 w-5 text-green-600" />}
                        {result.status === 'warning' && <AlertTriangle className="h-5 w-5 text-yellow-600" />}
                        {result.status === 'error' && <AlertTriangle className="h-5 w-5 text-red-600" />}
                        <span className="font-medium text-gray-900">{result.studentName}</span>
                      </div>
                      <div className="flex items-center space-x-4">
                        <span className={`text-sm font-bold ${
                          result.passFailStatus === 'Pass' ? 'text-green-700' : 'text-red-700'
                        }`}>
                          {result.score}/{result.totalMarks} - {result.passFailStatus}
                        </span>
                        <span className={`text-sm font-medium ${
                          result.status === 'success' ? 'text-green-700' :
                          result.status === 'warning' ? 'text-yellow-700' :
                          'text-red-700'
                        }`}>
                          {result.status === 'success' ? 'Success' :
                           result.status === 'warning' ? 'Warning' : 'Error'}
                        </span>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <span className="text-gray-600">Rank:</span>
                        <span className="ml-2 font-medium">{result.rank}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Locker Number:</span>
                        <span className="ml-2 font-medium">{result.lockerNumber}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Percentage:</span>
                        <span className="ml-2 font-medium">{result.percentage.toFixed(1)}%</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Correct:</span>
                        <span className="ml-2 font-medium text-green-600">{result.correctAnswers}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Incorrect:</span>
                        <span className="ml-2 font-medium text-red-600">{result.incorrectAnswers}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Blank:</span>
                        <span className="ml-2 font-medium text-yellow-600">{result.blankAnswers}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Invalid:</span>
                        <span className="ml-2 font-medium text-red-600">{result.invalidAnswers}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Confidence:</span>
                        <span className="ml-2 font-medium">{result.confidence}%</span>
                      </div>
                    </div>
                    
                    {result.issues && result.issues.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-gray-200">
                        <p className="text-sm text-gray-600 mb-1">Issues:</p>
                        <ul className="text-sm text-gray-700 list-disc list-inside">
                          {result.issues.map((issue, idx) => (
                            <li key={idx}>{issue}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      <footer className="mt-8 py-4 text-center">
        <p className="text-gray-600 text-sm font-medium">{resultSummary}</p>
      </footer>
    </div>
  );
};