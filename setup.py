#!/usr/bin/env python3
"""
Quick setup script for GitHub Codespaces Whisper API
Run this first to ensure everything is properly installed
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return False

def main():
    print("="*60)
    print("🚀 WHISPER API SETUP FOR GITHUB CODESPACES")
    print("="*60)
    
    # Check Python version
    python_version = sys.version
    print(f"🐍 Python version: {python_version}")
    
    # Create temp directory
    temp_dir = Path("/tmp/whisper_api")
    temp_dir.mkdir(exist_ok=True)
    print(f"📁 Temp directory created: {temp_dir}")
    
    # Install/upgrade packages
    print("\n📦 Installing system dependencies...")
    run_command("sudo apt update", "Updating package lists")
    run_command("sudo apt install -y ffmpeg", "Installing ffmpeg")
    
    packages = [
        "pip install --upgrade pip",
        "pip install openai-whisper",
        "pip install 'fastapi>=0.104.0'",
        "pip install 'uvicorn[standard]'", 
        "pip install python-multipart",
        "pip install psutil",
        "pip install aiofiles"
    ]
    
    print("\n📦 Installing packages...")
    for package in packages:
        run_command(package, f"Installing {package.split()[-1]}")
    
    # Test imports
    print("\n🧪 Testing imports...")
    try:
        import whisper
        import fastapi
        import uvicorn
        import psutil
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Check system resources
    try:
        import psutil
        memory_gb = round(psutil.virtual_memory().total / (1024**3), 2)
        cpu_count = psutil.cpu_count()
        print(f"\n💻 System resources:")
        print(f"   💾 RAM: {memory_gb}GB")
        print(f"   🔄 CPUs: {cpu_count}")
        
        if memory_gb >= 4:
            print("✅ Sufficient RAM for medium model")
        else:
            print("⚠️  Limited RAM - consider using small model")
            
    except Exception as e:
        print(f"⚠️  Could not check system resources: {e}")
    
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("📝 Next steps:")
    print("1. Run: python app.py")
    print("2. Check the PORTS tab in VS Code")
    print("3. Click on the forwarded port to access your API")
    print("4. Add '/docs' to the URL for interactive documentation")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)
    else:
        print("\n✅ Ready to run the Whisper API!")
