import React from 'react';
import { Users, User } from 'lucide-react';

interface Student {
  name: string;
  lockerNumber: string;
  rank: string;
  copyNumber: string;
}

interface StudentPreviewProps {
  students: Student[];
}

export const StudentPreview: React.FC<StudentPreviewProps> = ({ students }) => {
  if (students.length === 0) {
    return (
      <div className="text-center py-8">
        <Users className="h-12 w-12 text-gray-400 mx-auto mb-4" />
        <p className="text-gray-500">No students uploaded yet</p>
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-gray-700">
          Uploaded Students ({students.length})
        </h3>
      </div>
      
      <div className="bg-gray-50 rounded-lg overflow-hidden">
        <div className="max-h-64 overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-100 sticky top-0">
              <tr>
                <th className="px-4 py-2 text-left font-medium text-gray-700">Copy #</th>
                <th className="px-4 py-2 text-left font-medium text-gray-700">Name</th>
                <th className="px-4 py-2 text-left font-medium text-gray-700">Locker Number</th>
                <th className="px-4 py-2 text-left font-medium text-gray-700">Rank</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {students.map((student, index) => (
                <tr key={index} className="hover:bg-gray-100">
                  <td className="px-4 py-2 font-mono text-gray-900">{student.copyNumber}</td>
                  <td className="px-4 py-2 text-gray-900">{student.name}</td>
                  <td className="px-4 py-2 text-gray-900">{student.lockerNumber}</td>
                  <td className="px-4 py-2 text-gray-900">{student.rank}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      
      <div className="mt-4 flex items-center space-x-4 text-sm text-gray-600">
        <div className="flex items-center space-x-2">
          <User className="h-4 w-4" />
          <span>Total Students: {students.length}</span>
        </div>
        <div className="flex items-center space-x-2">
          <span>Copy Numbers: {students[0]?.copyNumber} - {students[students.length - 1]?.copyNumber}</span>
        </div>
      </div>
    </div>
  );
};