import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, AlertCircle, RefreshCw } from 'lucide-react';

interface FileUploadProps {
  onFileUpload: (file: File) => void;
  acceptedTypes: string;
  maxSize: number; // in MB
  loading?: boolean;
  title: string;
  description: string;
  instructions?: string[];
}

export const FileUpload: React.FC<FileUploadProps> = ({
  onFileUpload,
  acceptedTypes,
  maxSize,
  loading = false,
  title,
  description,
  instructions = [],
}) => {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      onFileUpload(acceptedFiles[0]);
    }
  }, [onFileUpload]);

  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    onDrop,
    accept: acceptedTypes.split(',').reduce((acc, type) => {
      acc[type.trim()] = [];
      return acc;
    }, {} as Record<string, string[]>),
    maxSize: maxSize * 1024 * 1024,
    multiple: false,
    disabled: loading,
  });

  return (
    <div className="space-y-4">
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer ${
          isDragActive
            ? 'border-blue-500 bg-blue-50'
            : loading
            ? 'border-gray-200 bg-gray-50 cursor-not-allowed'
            : 'border-gray-300 hover:border-blue-500 hover:bg-blue-50'
        }`}
      >
        <input {...getInputProps()} />
        
        <div className="flex flex-col items-center space-y-4">
          {loading ? (
            <RefreshCw className="h-12 w-12 text-blue-500 animate-spin" />
          ) : (
            <Upload className="h-12 w-12 text-gray-400" />
          )}
          
          <div>
            <h3 className="text-lg font-medium text-gray-900">{title}</h3>
            <p className="text-gray-600 mt-1">{description}</p>
          </div>
          
          {!loading && (
            <div className="text-sm text-gray-500">
              {isDragActive ? (
                <p>Drop the file here...</p>
              ) : (
                <p>Drag and drop a file here, or click to select</p>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Instructions */}
      {instructions.length > 0 && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h4 className="text-sm font-medium text-blue-900 mb-2">Instructions:</h4>
          <ul className="text-sm text-blue-800 space-y-1">
            {instructions.map((instruction, index) => (
              <li key={index} className="flex items-start space-x-2">
                <span className="text-blue-600 mt-0.5">•</span>
                <span>{instruction}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* File Rejection Errors */}
      {fileRejections.length > 0 && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-start space-x-3">
            <AlertCircle className="h-5 w-5 text-red-600 mt-0.5" />
            <div>
              <h4 className="text-sm font-medium text-red-800">File Upload Error:</h4>
              <ul className="mt-2 text-sm text-red-700 space-y-1">
                {fileRejections.map(({ file, errors }, index) => (
                  <li key={index}>
                    <strong>{file.name}:</strong>
                    <ul className="ml-4 list-disc">
                      {errors.map((error, errorIndex) => (
                        <li key={errorIndex}>{error.message}</li>
                      ))}
                    </ul>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};