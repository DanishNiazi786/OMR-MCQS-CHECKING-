import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Save, RefreshCw, FileText, AlertCircle, CheckCircle, Upload, Users } from 'lucide-react';
import { useApi } from '../context/ApiContext';
import { FileUpload } from '../components/FileUpload';
import { StudentPreview } from '../components/StudentPreview';

interface ExamForm {
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
}

interface Student {
  name: string;
  lockerNumber: string;
  rank: string;
  copyNumber: string;
}

export const ExamCreation: React.FC = () => {
  const { api } = useApi();
  const navigate = useNavigate();
  
  const [currentStep, setCurrentStep] = useState(1);
  const [examId, setExamId] = useState<string>('');
  const [form, setForm] = useState<ExamForm>({
    name: '',
    dateTime: '',
    time: '',
    numQuestions: 50,
    marksPerMcq: 1,
    passingPercentage: 60,
    wing: '',
    course: '',
    module: '',
    sponsorDS: '',
    instructions: 'Fill bubbles neatly with a black/blue pen, mark only one option per question—any extra, unclear, or incorrect marking will be considered wrong.',
  });
  
  const [students, setStudents] = useState<Student[]>([]);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [errors, setErrors] = useState<string[]>([]);

  const validateExamForm = () => {
    const newErrors: string[] = [];
    
    if (!form.name.trim()) newErrors.push('Exam name is required');
    if (!form.dateTime.trim()) newErrors.push('Date is required');
    if (!form.time.trim()) newErrors.push('Time is required');
    if (form.numQuestions < 1 || form.numQuestions > 200) {
      newErrors.push('Number of questions must be between 1 and 200');
    }
    if (form.marksPerMcq < 0.5) newErrors.push('Marks per MCQ must be at least 0.5');
    if (form.passingPercentage < 0 || form.passingPercentage > 100) {
      newErrors.push('Passing percentage must be between 0 and 100');
    }
    if (!form.wing.trim()) newErrors.push('Wing is required');
    if (!form.course.trim()) newErrors.push('Course is required');
    if (!form.module.trim()) newErrors.push('Module is required');
    if (!form.sponsorDS.trim()) newErrors.push('Sponsor DS is required');
    
    setErrors(newErrors);
    return newErrors.length === 0;
  };

  const handleCreateExam = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateExamForm()) return;
    
    setLoading(true);
    setErrors([]);
    
    try {
      // Generate a unique examId
      const examId = `EXAM_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      // Prepare form data to match backend schema
      const formData = {
        examId,
        name: form.name.trim(),
        dateTime: form.dateTime.trim(), // Send as string, no format enforcement
        time: form.time.trim(),
        numQuestions: form.numQuestions,
        marksPerMcq: form.marksPerMcq,
        passingPercentage: form.passingPercentage,
        wing: form.wing.trim(),
        course: form.course.trim(),
        module: form.module.trim(),
        sponsorDS: form.sponsorDS.trim(),
        instructions: form.instructions.trim(),
        studentsUploaded: false,
        solutionUploaded: false,
        createdAt: new Date().toISOString(),
        createdBy: 'System', // Adjust based on your auth system if needed
      };
      
      const response = await api.post('/exams', formData);
      setExamId(response.data.examId || examId);
      setCurrentStep(2);
    } catch (error: any) {
      console.error('Exam creation error:', error);
      setErrors([error.response?.data?.error || 'Failed to create exam. Please try again.']);
    } finally {
      setLoading(false);
    }
  };

  const handleStudentUpload = async (file: File) => {
    setLoading(true);
    setErrors([]);
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await api.post(`/students/${examId}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      
      setStudents(response.data.students);
      setCurrentStep(3);
    } catch (error: any) {
      console.error('Student upload error:', error);
      setErrors([error.response?.data?.error || 'Failed to upload students. Please check the file and try again.']);
    } finally {
      setLoading(false);
    }
  };

  const handleFinish = () => {
    setSuccess(true);
    setTimeout(() => {
      navigate('/');
    }, 2000);
  };

  if (success) {
    return (
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-8 text-center">
          <CheckCircle className="h-16 w-16 text-green-500 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Exam Created Successfully!</h2>
          <p className="text-gray-600 mb-4">
            Your exam "{form.name}" has been created with {students.length} students.
          </p>
          <p className="text-sm text-gray-500">Redirecting to dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Create New Exam</h1>
        <p className="text-gray-600 mt-2">
          Set up your exam details and upload student data
        </p>
      </div>

      {/* Progress Steps */}
      <div className="mb-8">
        <div className="flex items-center justify-center space-x-8">
          {[
            { step: 1, title: 'Exam Details', icon: FileText },
            { step: 2, title: 'Upload Students', icon: Upload },
            { step: 3, title: 'Review & Finish', icon: Users },
          ].map(({ step, title, icon: Icon }) => (
            <div key={step} className="flex items-center space-x-2">
              <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                currentStep >= step 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-200 text-gray-600'
              }`}>
                {currentStep > step ? (
                  <CheckCircle className="h-5 w-5" />
                ) : (
                  <Icon className="h-5 w-5" />
                )}
              </div>
              <span className={`text-sm font-medium ${
                currentStep >= step ? 'text-blue-600' : 'text-gray-500'
              }`}>
                {title}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Error Messages */}
      {errors.length > 0 && (
        <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <AlertCircle className="h-5 w-5 text-red-600 mt-0.5" />
            <div>
              <h3 className="text-sm font-medium text-red-800">Please fix the following errors:</h3>
              <ul className="mt-2 text-sm text-red-700 list-disc list-inside">
                {errors.map((error, index) => (
                  <li key={index}>{error}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Step 1: Exam Details */}
      {currentStep === 1 && (
        <form onSubmit={handleCreateExam} className="space-y-8">
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-6">Exam Information</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Exam Name *
                </label>
                <input
                  type="text"
                  value={form.name}
                  onChange={(e) => setForm(prev => ({ ...prev, name: e.target.value }))}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="Enter exam name"
                  required
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Date & Time *
                </label>
                <input
                  type="text"
                  value={form.dateTime}
                  onChange={(e) => setForm(prev => ({ ...prev, dateTime: e.target.value }))}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="Enter date and time"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Time *
                </label>
                <input
                  type="text"
                  value={form.time}
                  onChange={(e) => setForm(prev => ({ ...prev, time: e.target.value }))}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="Enter time"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Number of MCQs *
                </label>
                <input
                  type="number"
                  min="1"
                  max="200"
                  value={form.numQuestions}
                  onChange={(e) => setForm(prev => ({ ...prev, numQuestions: parseInt(e.target.value) || 50 }))}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Marks per MCQ *
                </label>
                <input
                  type="number"
                  min="0.5"
                  step="0.5"
                  value={form.marksPerMcq}
                  onChange={(e) => setForm(prev => ({ ...prev, marksPerMcq: parseFloat(e.target.value) || 1 }))}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Passing Percentage *
                </label>
                <input
                  type="number"
                  min="0"
                  max="100"
                  value={form.passingPercentage}
                  onChange={(e) => setForm(prev => ({ ...prev, passingPercentage: parseInt(e.target.value) || 60 }))}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Wing *
                </label>
                <input
                  type="text"
                  value={form.wing}
                  onChange={(e) => setForm(prev => ({ ...prev, wing: e.target.value }))}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="Enter wing"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Course *
                </label>
                <input
                  type="text"
                  value={form.course}
                  onChange={(e) => setForm(prev => ({ ...prev, course: e.target.value }))}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="Enter course"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Module *
                </label>
                <input
                  type="text"
                  value={form.module}
                  onChange={(e) => setForm(prev => ({ ...prev, module: e.target.value }))}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="Enter module"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Sponsor DS *
                </label>
                <input
                  type="text"
                  value={form.sponsorDS}
                  onChange={(e) => setForm(prev => ({ ...prev, sponsorDS: e.target.value }))}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="Enter sponsor DS"
                  required
                />
              </div>
            </div>

            <div className="mt-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Instructions
              </label>
              <textarea
                value={form.instructions}
                onChange={(e) => setForm(prev => ({ ...prev, instructions: e.target.value }))}
                rows={4}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Enter instructions for students"
              />
            </div>

            {/* Summary */}
            <div className="mt-6 p-4 bg-gray-50 rounded-lg">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Exam Summary</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-gray-600">Total Questions:</span>
                  <span className="ml-2 font-medium">{form.numQuestions}</span>
                </div>
                <div>
                  <span className="text-gray-600">Total Marks:</span>
                  <span className="ml-2 font-medium">{form.numQuestions * form.marksPerMcq}</span>
                </div>
                <div>
                  <span className="text-gray-600">Passing Marks:</span>
                  <span className="ml-2 font-medium">
                    {Math.ceil((form.numQuestions * form.marksPerMcq * form.passingPercentage) / 100)}
                  </span>
                </div>
                <div>
                  <span className="text-gray-600">Passing %:</span>
                  <span className="ml-2 font-medium">{form.passingPercentage}%</span>
                </div>
              </div>
            </div>
          </div>

          <div className="flex justify-end">
            <button
              type="submit"
              disabled={loading}
              className="flex items-center space-x-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? (
                <RefreshCw className="h-5 w-5 animate-spin" />
              ) : (
                <Save className="h-5 w-5" />
              )}
              <span>{loading ? 'Creating...' : 'Create Exam'}</span>
            </button>
          </div>
        </form>
      )}

      {/* Step 2: Upload Students */}
      {currentStep === 2 && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-6">Upload Student Data</h2>
          
          <FileUpload
            onFileUpload={handleStudentUpload}
            acceptedTypes=".xlsx,.xls"
            maxSize={10}
            loading={loading}
            title="Upload Excel File"
            description="Upload an Excel file (.xlsx or .xls) containing student details"
            instructions={[
              'Excel file must contain columns: Name, Locker Number, Rank',
              'Maximum file size: 10MB',
              'Ensure all required fields are filled',
            ]}
          />
        </div>
      )}

      {/* Step 3: Review & Finish */}
      {currentStep === 3 && (
        <div className="space-y-6">
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-6">Review & Finish</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-3">Exam Details</h3>
                <div className="space-y-2 text-sm">
                  <div><span className="text-gray-600">Name:</span> <span className="ml-2">{form.name}</span></div>
                  <div><span className="text-gray-600">Date:</span> <span className="ml-2">{form.dateTime}</span></div>
                  <div><span className="text-gray-600">Time:</span> <span className="ml-2">{form.time}</span></div>
                  <div><span className="text-gray-600">Questions:</span> <span className="ml-2">{form.numQuestions}</span></div>
                  <div><span className="text-gray-600">Marks per MCQ:</span> <span className="ml-2">{form.marksPerMcq}</span></div>
                  <div><span className="text-gray-600">Total Marks:</span> <span className="ml-2">{form.numQuestions * form.marksPerMcq}</span></div>
                  <div><span className="text-gray-600">Passing %:</span> <span className="ml-2">{form.passingPercentage}%</span></div>
                </div>
              </div>
              
              <div>
                <h3 className="text-sm font-medium text-gray-700 mb-3">Course Information</h3>
                <div className="space-y-2 text-sm">
                  <div><span className="text-gray-600">Wing:</span> <span className="ml-2">{form.wing}</span></div>
                  <div><span className="text-gray-600">Course:</span> <span className="ml-2">{form.course}</span></div>
                  <div><span className="text-gray-600">Module:</span> <span className="ml-2">{form.module}</span></div>
                  <div><span className="text-gray-600">Sponsor DS:</span> <span className="ml-2">{form.sponsorDS}</span></div>
                </div>
              </div>
            </div>

            <StudentPreview students={students} />

            <div className="flex justify-end space-x-4 mt-6">
              <button
                onClick={() => setCurrentStep(2)}
                className="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
              >
                Back
              </button>
              <button
                onClick={handleFinish}
                className="flex items-center space-x-2 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
              >
                <CheckCircle className="h-5 w-5" />
                <span>Finish</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};