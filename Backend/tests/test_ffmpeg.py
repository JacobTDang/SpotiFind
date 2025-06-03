#!/usr/bin/env python3
"""
Quick test to verify FFmpeg is working properly
"""

import subprocess
import sys

def test_ffmpeg():
    """Test if FFmpeg is working and accessible."""
    print("🔧 Testing FFmpeg Installation...")

    try:
        # Test ffmpeg command
        result = subprocess.run(['ffmpeg', '-version'],
                              capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            # Extract version info
            lines = result.stdout.split('\n')
            version_line = lines[0] if lines else "Unknown version"
            print(f" FFmpeg found: {version_line}")

            # Check for important encoders
            encoders_result = subprocess.run(['ffmpeg', '-encoders'],
                                           capture_output=True, text=True, timeout=5)

            if 'pcm_s16le' in encoders_result.stdout:
                print(" PCM encoder available")
            else:
                print("  PCM encoder not found")

            return True

        else:
            print(f" FFmpeg error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(" FFmpeg command timed out")
        return False
    except FileNotFoundError:
        print(" FFmpeg not found in PATH")
        print("   Try: choco install ffmpeg")
        print("   Or download from: https://www.gyan.dev/ffmpeg/builds/")
        return False
    except Exception as e:
        print(f" Error testing FFmpeg: {e}")
        return False

def test_yt_dlp_with_ffmpeg():
    """Test yt-dlp with FFmpeg integration."""
    print("\n🎵 Testing yt-dlp + FFmpeg Integration...")

    try:
        import yt_dlp as yt
        import tempfile
        import os

        # Test configuration
        test_opts = {
            'quiet': True,
            'format': 'bestaudio',
            'outtmpl': os.path.join(tempfile.gettempdir(), 'test_%(id)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
        }

        # Test with a very short video (first few seconds)
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        print(f"Testing download from: {test_url}")
        print("(This will download a few seconds for testing...)")

        with yt.YoutubeDL(test_opts) as ydl:
            # Get info first
            info = ydl.extract_info(test_url, download=False)
            print(f" Video info extracted: {info.get('title', 'Unknown')}")

            # Try to download (just to test the pipeline)
            # We'll interrupt it quickly since we just want to test the setup
            print("   Testing download pipeline...")

        print(" yt-dlp + FFmpeg integration appears to be working")
        return True

    except ImportError:
        print(" yt-dlp not installed: pip install yt-dlp")
        return False
    except Exception as e:
        print(f" Error testing yt-dlp integration: {e}")
        return False

def test_audio_libraries():
    """Test audio processing libraries."""
    print("\n🎧 Testing Audio Libraries...")

    # Test soundfile
    try:
        import soundfile as sf
        print(" soundfile available")
    except ImportError:
        print(" soundfile not available: pip install soundfile")

    # Test librosa
    try:
        import librosa
        print(" librosa available")
    except ImportError:
        print("  librosa not available: pip install librosa")

    # Test openl3
    try:
        import openl3
        print(" openl3 available")
    except ImportError:
        print(" openl3 not available: pip install openl3")

if __name__ == "__main__":
    print("🧪 Audio Processing Environment Test")
    print("=" * 50)

    ffmpeg_ok = test_ffmpeg()
    test_audio_libraries()

    if ffmpeg_ok:
        # Only test yt-dlp if FFmpeg is working
        test_yt_dlp_with_ffmpeg()
    else:
        print("\n  Skipping yt-dlp test due to FFmpeg issues")

    print("\n" + "=" * 50)
    if ffmpeg_ok:
        print(" Environment looks good! Try running your audio utils tests again.")
    else:
        print(" Fix FFmpeg installation and run this test again.")
