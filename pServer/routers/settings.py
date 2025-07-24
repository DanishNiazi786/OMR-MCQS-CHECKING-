from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import motor.motor_asyncio
import os

router = APIRouter()

# In-memory settings store
app_settings = {
    "scanner": {
        "resolution": 300,
        "colorMode": "black_and_white",
        "autoFeed": True,
        "duplex": False
    },
    "processing": {
        "confidenceThreshold": 0.7,
        "autoProcessing": True,
        "batchSize": 50
    },
    "exam": {
        "defaultQuestions": 5,
        "passingScore": 60,
        "allowPartialCredit": False
    },
    "database": {
        "connectionString": os.getenv("MONGO_URI", "mongodb+srv://dani:123@cluster0.zgjz474.mongodb.net/"),
        "connected": True
    }
}

class SettingsUpdate(BaseModel):
    scanner: Dict[str, Any] = None
    processing: Dict[str, Any] = None
    exam: Dict[str, Any] = None
    database: Dict[str, Any] = None

@router.get("/")
async def get_settings():
    return app_settings

@router.put("/")
async def update_settings(updates: SettingsUpdate):
    try:
        global app_settings
        
        update_dict = updates.dict(exclude_unset=True)
        
        for key, value in update_dict.items():
            if key in app_settings and isinstance(value, dict):
                app_settings[key].update(value)
            else:
                app_settings[key] = value

        return app_settings
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update settings: {str(e)}")

@router.post("/test-db")
async def test_database_connection():
    try:
        from main import app
        
        if hasattr(app.state, 'database'):
            await app.state.database.command("ping")
            return {
                "connected": True,
                "status": "connected",
                "message": "Database connection successful"
            }
        return {
            "connected": False,
            "status": "disconnected",
            "message": "Database not initialized"
        }
    except Exception as e:
        return {
            "connected": False,
            "status": "error",
            "message": f"Database connection test failed: {str(e)}"
        }

@router.post("/reset")
async def reset_settings():
    global app_settings
    
    app_settings = {
        "scanner": {
            "resolution": 300,
            "colorMode": "black_and_white",
            "autoFeed": True,
            "duplex": False
        },
        "processing": {
            "confidenceThreshold": 0.7,
            "autoProcessing": True,
            "batchSize": 50
        },
        "exam": {
            "defaultQuestions": 5,
            "passingScore": 60,
            "allowPartialCredit": False
        },
        "database": {
            "connectionString": os.getenv("MONGO_URI", "mongodb+srv://dani:123@cluster0.zgjz474.mongodb.net/"),
            "connected": True
        }
    }

    return {
        "message": "Settings reset to defaults",
        "settings": app_settings
    }

@router.get("/scanner-status")
async def get_scanner_status():
    return {
        "available": True,
        "devices": [
            {"id": "scanner1", "name": "Default Scanner", "status": "ready"}
        ],
        "currentDevice": "scanner1"
    }