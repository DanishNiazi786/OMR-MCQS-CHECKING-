from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from motor.motor_asyncio import AsyncIOMotorClient
import uvicorn
import os
from contextlib import asynccontextmanager
from datetime import datetime
import logging

# Import routers from the correct 'routes' directory
from routers import exams, scan, results, reports, settings, students, omr, solutions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database connection
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb+srv://dani:123@cluster0.zgjz474.mongodb.net/omr_database?retryWrites=true&w=majority")
OCR_ENABLED = os.getenv("OCR_ENABLED", "false").lower() == "true"  # New env var for OCR toggle
client = None
database = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, database
    try:
        client = AsyncIOMotorClient(MONGODB_URL)
        database = client.get_database("omr_database")  # Specify database name
        logger.info(f"MongoDB connected successfully at {datetime.utcnow().isoformat()}Z")
        app.state.database = database
        app.state.ocr_enabled = OCR_ENABLED  # Store OCR toggle in app state
        yield
    finally:
        if client:
            client.close()
            logger.info(f"MongoDB connection closed at {datetime.utcnow().isoformat()}Z")

app = FastAPI(
    title="Enhanced OMR Processing API",
    description="Backend API for Optical Mark Recognition (OMR) processing system with student credential extraction and PDF result generation",
    version="1.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(exams.router, prefix="/api/exams", tags=["exams"])
app.include_router(scan.router, prefix="/api/scan", tags=["scan"])
app.include_router(results.router, prefix="/api/results", tags=["results"])
app.include_router(reports.router, prefix="/api/reports", tags=["reports"])
app.include_router(settings.router, prefix="/api/settings", tags=["settings"])
app.include_router(students.router, prefix="/api/students", tags=["students"])
app.include_router(omr.router, prefix="/api/omr", tags=["omr"])
app.include_router(solutions.router, prefix="/api/solutions", tags=["solutions"])

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {
        "status": "OK",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "ocr_enabled": app.state.ocr_enabled,
        "version": "1.1.0"
    }

# Error handling
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Server error at {datetime.utcnow().isoformat()}Z: {str(exc)}", exc_info=True)
    return HTTPException(
        status_code=500,
        detail={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 3001)),
        reload=True
    )