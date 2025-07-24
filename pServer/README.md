# OMR Processing System - Python Backend

This is a Python FastAPI backend for an OMR (Optical Mark Recognition) processing system, converted from the original Node.js/Express implementation.

## Features

- **Exam Management**: Create, update, and manage exams
- **Student Management**: Upload and manage student lists via Excel files
- **Solution Management**: Upload and manage answer keys
- **OMR Processing**: Process scanned answer sheets using OpenCV
- **Results Management**: Generate and export results in various formats
- **Report Generation**: Create detailed Excel and PDF reports
- **OMR Sheet Generation**: Generate printable OMR answer sheets

## Technology Stack

- **FastAPI**: Modern, fast web framework for building APIs
- **Motor**: Async MongoDB driver for Python
- **Pydantic**: Data validation using Python type annotations
- **OpenCV**: Computer vision library for image processing
- **ReportLab**: PDF generation library
- **OpenPyXL**: Excel file processing
- **Pandas**: Data manipulation and analysis

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Set up MongoDB connection (update the connection string in `main.py` if needed)

3. Run the application:
```bash
python main.py
```

The API will be available at `http://localhost:3001`

## API Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:3001/docs`
- ReDoc: `http://localhost:3001/redoc`

## Project Structure

```
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── models/                 # Pydantic models for data validation
│   ├── exam.py
│   ├── student.py
│   ├── solution.py
│   ├── response.py
│   ├── result.py
│   └── report.py
└── routers/               # API route handlers
    ├── exams.py
    ├── students.py
    ├── solutions.py
    ├── scan.py
    ├── results.py
    ├── reports.py
    ├── settings.py
    └── omr.py
```

## Key Differences from Node.js Version

1. **Async/Await**: Uses Python's async/await syntax with Motor for MongoDB operations
2. **Type Safety**: Leverages Pydantic models for request/response validation
3. **Error Handling**: Uses FastAPI's HTTPException for consistent error responses
4. **File Processing**: Uses pandas for Excel processing and OpenCV for image processing
5. **PDF Generation**: Uses ReportLab instead of PDFKit for PDF generation

## Environment Variables

- `PORT`: Server port (default: 3001)
- `MONGODB_URL`: MongoDB connection string

## Development

To run in development mode with auto-reload:
```bash
uvicorn main:app --reload --port 3001
```

## Testing

The API includes comprehensive error handling and validation. Test the endpoints using the interactive documentation or tools like Postman.