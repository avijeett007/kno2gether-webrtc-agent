#!/usr/bin/env python
"""
Download script for NHS agent models

This script downloads the required models for the NHS agent:
- Silero VAD model
- Turn detector model

Run this script before starting the agent to ensure all models are downloaded.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("model_downloader")

def download_models():
    """Download all required models"""
    success = True
    
    # Try to import required modules
    try:
        from livekit.plugins import silero, turn_detector
        logger.info("LiveKit plugins imported successfully")
    except ImportError as e:
        logger.error(f"Failed to import LiveKit plugins: {e}")
        logger.error("Make sure you have installed the requirements: pip install -r requirements.txt")
        return False
    
    # Download Silero VAD model
    try:
        logger.info("Downloading Silero VAD model...")
        vad = silero.VAD.load()
        logger.info("✅ Silero VAD model downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to download Silero VAD model: {e}")
        success = False
    
    # Download turn detector model
    try:
        logger.info("Downloading turn detector model...")
        eou_model = turn_detector.EOUModel()
        logger.info("✅ Turn detector model downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to download turn detector model: {e}")
        success = False
    
    # Check cache directories
    for dir_name, name in [
        ("~/.cache/livekit-plugins-silero", "Silero VAD"),
        ("~/.cache/livekit-plugins-turn-detector", "Turn detector")
    ]:
        expanded_dir = os.path.expanduser(dir_name)
        if os.path.exists(expanded_dir):
            files = os.listdir(expanded_dir)
            logger.info(f"{name} cache directory ({expanded_dir}): {files}")
        else:
            logger.warning(f"{name} cache directory does not exist: {expanded_dir}")
            success = False
    
    return success

if __name__ == "__main__":
    logger.info("Starting model download...")
    
    if download_models():
        logger.info("✅ All models downloaded successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Failed to download some models. Please check the logs.")
        sys.exit(1) 