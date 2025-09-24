#!/usr/bin/env python3
"""
Whisper Transcription API for GitHub Codespaces
Optimized for medium model with persistent environment
"""

import whisper
import tempfile
import os
import json
import uuid
import asyncio
import logging
import psutil
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import time
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
model = None
transcription_semaphore = None

def get_system_info():
    """Get system resource information"""
    try:
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_memory_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "disk_usage_gb": round(psutil.disk_usage('/').free / (1024**3), 2)
        }
    except Exception as e:
        logger.warning(f"Could not get system info: {e}")
        return {"error": str(e)}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global model, transcription_semaphore
    
    # Startup
    system_info = get_system_info()
    logger.info(f"System resources: {system_info}")
    
    logger.info("Loading Whisper MEDIUM model...")
    try:
        # Load medium model - Codespaces should handle this well
        model = whisper.load_model("medium")
        logger.info("✅ Whisper MEDIUM model loaded successfully!")
        
        # Set concurrency based on available resources
        available_memory = system_info.get("available_memory_gb", 4)
        if available_memory >= 8:
            max_concurrent = 2
            logger.info("High memory available - allowing 2 concurrent requests")
        else:
            max_concurrent = 1
            logger.info("Limited memory - restricting to 1 concurrent request")
            
        transcription_semaphore = asyncio.Semaphore(max_concurrent)
        
    except Exception as e:
        logger.error(f"❌ Failed to load Whisper model: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down API...")

app = FastAPI(
    title="Whisper API - Codespaces Medium Model", 
    version="2.0.0",
    description="High-quality audio transcription with Whisper Medium model on GitHub Codespaces",
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

# Configuration
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB - Codespaces can handle this
SUPPORTED_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.mp4', '.avi', '.mov', '.mkv', '.webm', '.flac', '.aac', '.ogg'}
TEMP_DIR = Path(tempfile.gettempdir()) / "whisper_api"
TEMP_DIR.mkdir(exist_ok=True)

def get_file_extension(filename: str, content_type: str) -> str:
    """Determine appropriate file extension"""
    if filename and '.' in filename:
        ext = os.path.splitext(filename)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            return ext

    extension_map = {
        'audio/mpeg': '.mp3', 'audio/wav': '.wav', 'audio/mp4': '.m4a',
        'audio/ogg': '.ogg', 'video/mp4': '.mp4', 'video/avi': '.avi',
        'video/mov': '.mov', 'video/quicktime': '.mov', 'video/webm': '.webm'
    }
    return extension_map.get(content_type, '.mp4')

def is_supported_file(filename: str, content_type: str) -> bool:
    """Check if file format is supported"""
    if filename:
        ext = os.path.splitext(filename)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            return True
    
    if content_type:
        return any(content_type.startswith(t) for t in {'audio/', 'video/'})
    return False

def cleanup_temp_file(file_path: Path):
    """Clean up temporary files"""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"🗑️ Cleaned up: {file_path}")
    except Exception as e:
        logger.warning(f"⚠️ Cleanup failed for {file_path}: {e}")

async def transcribe_audio_file(file_path: Path, options: dict = None) -> dict:
    """Transcribe audio file with comprehensive error handling"""
    if options is None:
        options = {}
    
    # Optimize for medium model
    options.update({
        "verbose": False,
        "fp16": False  # Better CPU compatibility
    })
    
    try:
        loop = asyncio.get_event_loop()
        
        # Generous timeout for medium model quality
        result = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: model.transcribe(str(file_path), **options)),
            timeout=900  # 15 minutes - plenty for medium model
        )
        return result
    except asyncio.TimeoutError:
        logger.error(f"⏰ Transcription timeout for {file_path}")
        raise HTTPException(
            status_code=408, 
            detail="Transcription timeout. File may be too long."
        )
    except Exception as e:
        logger.error(f"💥 Transcription failed for {file_path}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Transcription error: {str(e)}"
        )

@app.post("/transcribe")
async def transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: Optional[str] = None,
    task: Optional[str] = None
):
    """
    Transcribe audio/video with medium model quality
    
    Parameters:
    - file: Audio/video file (max 100MB)
    - language: Optional language code (e.g., 'en', 'es', 'fr')
    - task: 'transcribe' (default) or 'translate'
    """
    
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="🚫 Whisper model not loaded. Please wait and try again."
        )
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="📄 No filename provided")
    
    if not is_supported_file(file.filename, file.content_type):
        raise HTTPException(
            status_code=400, 
            detail=f"🚫 Unsupported format. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # Read and validate file
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"📦 File too large. Maximum: {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="📭 Empty file uploaded")

    # Process transcription with semaphore
    async with transcription_semaphore:
        file_extension = get_file_extension(file.filename, file.content_type)
        unique_id = str(uuid.uuid4())
        temp_path = TEMP_DIR / f"audio_{unique_id}{file_extension}"
        
        try:
            # Save uploaded file
            with open(temp_path, "wb") as f:
                f.write(contents)
            
            logger.info(f"🎵 Processing {file.filename} with MEDIUM model ({len(contents):,} bytes)")
            
            # Prepare transcription options
            options = {"verbose": False, "fp16": False}
            if language:
                options["language"] = language
                logger.info(f"🌍 Language specified: {language}")
            if task:
                options["task"] = task
                logger.info(f"🎯 Task: {task}")
            
            # Perform transcription
            start_time = time.time()
            result = await transcribe_audio_file(temp_path, options)
            processing_time = time.time() - start_time
            
            # Enhance result with metadata
            result["file_info"] = {
                "filename": file.filename,
                "content_type": file.content_type,
                "file_size_bytes": len(contents),
                "processing_time_seconds": round(processing_time, 2),
                "model_used": "whisper-medium",
                "platform": "GitHub Codespaces",
                "language_detected": result.get("language", "unknown")
            }
            
            logger.info(f"✅ MEDIUM model transcription completed in {processing_time:.2f}s")
            logger.info(f"📝 Detected language: {result.get('language', 'unknown')}")
            
            return JSONResponse(content=result)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"💥 Unexpected error processing {file.filename}: {e}")
            raise HTTPException(
                status_code=500, 
                detail="🚨 Internal server error during transcription"
            )
        finally:
            # Schedule cleanup
            background_tasks.add_task(cleanup_temp_file, temp_path)

@app.get("/")
async def root():
    """API root with comprehensive information"""
    system_info = get_system_info()
    return {
        "message": "🎙️ Whisper Transcription API",
        "subtitle": "High-Quality Medium Model on GitHub Codespaces",
        "version": "2.0.0",
        "model": "whisper-medium",
        "platform": "GitHub Codespaces",
        "system_info": system_info,
        "advantages": [
            "✨ Medium model quality",
            "🔒 Persistent 24/7 environment",
            "💾 100MB file support",
            "🌍 120+ languages supported",
            "⚡ Up to 8GB RAM",
            "🎯 Translation capability"
        ],
        "endpoints": {
            "POST /transcribe": "Upload audio/video for transcription",
            "GET /health": "System health and status",
            "GET /supported-formats": "List supported file formats",
            "GET /stats": "Usage statistics"
        },
        "usage_tip": "Send POST request to /transcribe with audio/video file"
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    system_info = get_system_info()
    model_status = "✅ loaded" if model is not None else "❌ not_loaded"
    available_slots = transcription_semaphore._value if transcription_semaphore else 0
    
    return {
        "status": "🟢 healthy" if model is not None else "🟡 degraded",
        "model": "whisper-medium",
        "model_status": model_status,
        "system_info": system_info,
        "processing": {
            "available_slots": available_slots,
            "max_concurrent": 2 if system_info.get("available_memory_gb", 0) >= 8 else 1,
            "timeout_minutes": 15
        },
        "limits": {
            "max_file_size_mb": MAX_FILE_SIZE // (1024*1024),
            "supported_formats": len(SUPPORTED_EXTENSIONS)
        },
        "platform": "GitHub Codespaces",
        "uptime_advantages": [
            "🔄 Persistent environment",
            "⏰ 120 hours/month free",
            "🚀 No session timeouts",
            "💪 Better resource allocation"
        ]
    }

@app.get("/supported-formats")
async def get_supported_formats():
    """Detailed format support information"""
    audio_formats = ['.mp3', '.wav', '.flac', '.aac', '.m4a', '.ogg']
    video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    return {
        "audio_formats": sorted(audio_formats),
        "video_formats": sorted(video_formats),
        "all_supported": sorted(list(SUPPORTED_EXTENSIONS)),
        "total_formats": len(SUPPORTED_EXTENSIONS),
        "max_file_size_mb": MAX_FILE_SIZE // (1024*1024),
        "model_quality": "medium (high accuracy)",
        "platform": "GitHub Codespaces",
        "note": "🎬 Video files: audio track extracted and transcribed"
    }

@app.get("/stats")
async def get_stats():
    """API usage and system statistics"""
    system_info = get_system_info()
    available_slots = transcription_semaphore._value if transcription_semaphore else 0
    max_slots = 2 if system_info.get("available_memory_gb", 0) >= 8 else 1
    
    return {
        "system": system_info,
        "processing": {
            "model": "whisper-medium",
            "max_concurrent_slots": max_slots,
            "active_slots": max_slots - available_slots,
            "available_slots": available_slots,
            "timeout_seconds": 900
        },
        "storage": {
            "temp_directory": str(TEMP_DIR),
            "temp_dir_exists": TEMP_DIR.exists()
        },
        "platform_info": {
            "environment": "GitHub Codespaces",
            "model_loaded": model is not None,
            "advantages": [
                "Persistent 24/7 running",
                "No session disconnections", 
                "Better resource allocation",
                "120 hours/month free tier"
            ]
        }
    }

# Custom error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler with better formatting"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions gracefully"""
    logger.error(f"💥 Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "🚨 Internal server error",
            "status_code": 500,
            "timestamp": time.time()
        }
    )

if __name__ == "__main__":
    # Get system info for startup
    system_info = get_system_info()
    
    print("="*60)
    print("🚀 WHISPER API - GITHUB CODESPACES")
    print("="*60)
    print(f"🖥️  System: {system_info.get('cpu_count', 'unknown')} CPUs")
    print(f"💾 RAM: {system_info.get('memory_gb', 'unknown')}GB total, {system_info.get('available_memory_gb', 'unknown')}GB available")
    print(f"💿 Disk: {system_info.get('disk_usage_gb', 'unknown')}GB free")
    print(f"🧠 Model: Whisper MEDIUM (high quality)")
    print("="*60)
    
    # Determine host - Codespaces needs 0.0.0.0
    host = "0.0.0.0"
    port = 8000
    
    print(f"🌐 Starting server on {host}:{port}")
    print("📋 Once running, check the PORTS tab in VS Code")
    print("🔗 Use the forwarded URL to access your API")
    print("📚 API docs will be at: <forwarded-url>/docs")
    print("💚 Health check at: <forwarded-url>/health")
    print("="*60)
    
    # Run the server
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        log_level="info"
    )