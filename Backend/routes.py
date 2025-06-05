import os
import shutil
import uuid
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify, send_file
import logging
from typing import List, Dict

# Your imports
from models import Song, SongEmbedding
from database import db
from audio_utils import (
    find_similar_songs,
    load_youtube_track,
    get_embedding_from_file,
    save_song,
    load_youtube_playlist,
    convert_audio_with_ffmpeg
)

# Initialize Blueprint
bp = Blueprint('main', __name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
EMBEDDING_DURATION = 30  # seconds
RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), 'saved_recordings')
TEMP_DIR = os.path.join(os.path.dirname(__file__), 'temp_uploads')

# Create required directories
os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# =============================================================================
# BASIC ROUTES
# =============================================================================

@bp.route('/')
def index():
    """API root endpoint"""
    return jsonify({
        'message': 'Music Similarity API',
        'version': '1.0',
        'endpoints': {
            'upload': 'POST /upload',
            'youtube': 'POST /youtube',
            'youtube_playlist': 'POST /youtube-playlist',
            'search': 'POST /search-audio',
            'playback': 'GET /play-recording/<search_id>',
            'recent': 'GET /recent-recordings'
        }
    })

@bp.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat()
    })

# =============================================================================
# UPLOAD ROUTES
# =============================================================================

@bp.route('/upload', methods=['POST'])
def upload():
    """Upload and process local audio file"""
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No file provided'
            }), 400

        file = request.files['file']
        title = request.form.get('title', '').strip()
        artist = request.form.get('artist', '').strip()

        # Validate inputs
        if not file or file.filename == '':
            return jsonify({
                'success': False,
                'message': 'Please select a file'
            }), 400

        if not title or not artist:
            return jsonify({
                'success': False,
                'message': 'Title and artist are required'
            }), 400

        # Save temporary file
        filename = file.filename
        temp_path = os.path.join(TEMP_DIR, filename)
        file.save(temp_path)

        try:
            # Process audio
            embedding, duration = get_embedding_from_file(temp_path)

            # Create song record
            song = Song(
                title=title,
                artist=artist,
                source='local upload',
                preview_url=None
            )
            save_song(song)

            # Create embedding record
            song_embedding = SongEmbedding(
                songID=song.songID,
                audioStart=0.0,
                audioDuration=min(duration, EMBEDDING_DURATION),
                embedding=embedding.tolist(),
                dimensions=512
            )
            db.session.add(song_embedding)
            db.session.commit()

            logger.info(f"Successfully uploaded: {title} by {artist}")

            return jsonify({
                'success': True,
                'message': 'Upload successful',
                'song': title,
                'songID': song.songID,
                'duration': duration
            }), 200

        except Exception as e:
            db.session.rollback()
            logger.error(f"Processing error: {e}")
            return jsonify({
                'success': False,
                'message': f'Processing error: {str(e)}'
            }), 500

        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({
            'success': False,
            'message': f'Upload failed: {str(e)}'
        }), 500

# =============================================================================
# YOUTUBE ROUTES
# =============================================================================

@bp.route('/youtube', methods=['POST'])
def youtube():
    """Add single YouTube video"""
    try:
        data = request.get_json()

        if not data or 'url' not in data:
            return jsonify({
                'success': False,
                'message': 'YouTube URL is required'
            }), 400

        url = data['url'].strip()

        # Validate URL
        if not url or ('youtube.com' not in url and 'youtu.be' not in url):
            return jsonify({
                'success': False,
                'message': 'Invalid YouTube URL'
            }), 400

        logger.info(f"Processing YouTube URL: {url}")

        # Process video
        result = load_youtube_track(url)

        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400

    except Exception as e:
        logger.error(f"YouTube error: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@bp.route('/youtube-playlist', methods=['POST'])
def youtube_playlist():
    """Process YouTube playlist"""
    try:
        data = request.get_json()

        if not data or 'url' not in data:
            return jsonify({
                'success': False,
                'message': 'Playlist URL is required'
            }), 400

        playlist_url = data['url'].strip()
        max_videos = data.get('max_videos', 50)

        # Validate
        if not playlist_url:
            return jsonify({
                'success': False,
                'message': 'Playlist URL cannot be empty'
            }), 400

        if 'playlist' not in playlist_url:
            return jsonify({
                'success': False,
                'message': 'Invalid playlist URL'
            }), 400

        # Clamp max_videos
        max_videos = max(1, min(100, int(max_videos)))

        logger.info(f"Processing playlist: {playlist_url} (max: {max_videos})")

        # Process playlist
        results = load_youtube_playlist(playlist_url, max_videos)

        # Separate results
        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', False)]

        return jsonify({
            'success': True,
            'message': f'Processed {len(successful)}/{len(results)} videos',
            'total_videos': len(results),
            'successful_count': len(successful),
            'failed_count': len(failed),
            'successful_songs': successful,
            'failed_songs': failed,
            'summary': {
                'playlist_url': playlist_url,
                'max_videos_requested': max_videos,
                'processed_at': datetime.utcnow().isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Playlist error: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

# =============================================================================
# SEARCH ROUTES
# =============================================================================

@bp.route('/search-audio', methods=['POST'])
def search_audio():
    """Search for similar songs using audio recording"""
    converted_path = None
    saved_recording_path = None

    try:
        # Validate request
        if 'audio' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No audio file provided'
            }), 400

        audio_file = request.files['audio']
        if not audio_file or audio_file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No audio file selected'
            }), 400

        # Generate unique ID for this search
        search_id = str(uuid.uuid4())
        timestamp = datetime.now()

        # Save temp file
        temp_path = os.path.join(TEMP_DIR, f"search_{search_id}.webm")
        audio_file.save(temp_path)

        try:
            # Convert to WAV
            converted_path = convert_audio_with_ffmpeg(temp_path)
            if not converted_path:
                raise Exception("Failed to convert audio")

            # Save recording for playback
            saved_filename = f"recording_{search_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.wav"
            saved_recording_path = os.path.join(RECORDINGS_DIR, saved_filename)
            shutil.copy2(converted_path, saved_recording_path)

            # Generate embedding
            embedding, duration = get_embedding_from_file(converted_path)

            # Find similar songs
            similar_songs = find_similar_songs(embedding, limit=10)

            logger.info(f"Found {len(similar_songs)} similar songs")

            # Prepare response
            response_data = {
                'success': True,
                'similar_songs': similar_songs,
                'search_duration': duration,
                'recording_playback': {
                    'search_id': search_id,
                    'playback_url': f'/play-recording/{search_id}',
                    'timestamp': timestamp.isoformat(),
                    'duration': duration
                }
            }

            if not similar_songs:
                response_data['success'] = False
                response_data['message'] = 'No similar songs found'

            return jsonify(response_data)

        except Exception as e:
            logger.error(f"Processing error: {e}")
            return jsonify({
                'success': False,
                'message': f'Processing failed: {str(e)}'
            }), 500

        finally:
            # Cleanup temp files (keep recording)
            for path in [temp_path, converted_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({
            'success': False,
            'message': f'Search failed: {str(e)}'
        }), 500

# =============================================================================
# PLAYBACK ROUTES
# =============================================================================

@bp.route('/play-recording/<search_id>')
def play_recording(search_id):
    """Serve saved recording for playback"""
    try:
        # Find recording file
        recording_files = [
            f for f in os.listdir(RECORDINGS_DIR)
            if f.startswith(f'recording_{search_id}_')
        ]

        if not recording_files:
            return jsonify({
                'success': False,
                'message': 'Recording not found'
            }), 404

        # Get most recent
        recording_file = sorted(recording_files)[-1]
        file_path = os.path.join(RECORDINGS_DIR, recording_file)

        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'message': 'File not found'
            }), 404

        # Serve file
        return send_file(
            file_path,
            mimetype='audio/wav',
            as_attachment=False,
            download_name=f'recording_{search_id}.wav'
        )

    except Exception as e:
        logger.error(f"Playback error: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@bp.route('/recent-recordings')
def recent_recordings():
    """Get list of recent recordings"""
    try:
        recordings = []

        for filename in os.listdir(RECORDINGS_DIR):
            if filename.startswith('recording_') and filename.endswith('.wav'):
                file_path = os.path.join(RECORDINGS_DIR, filename)
                file_stat = os.stat(file_path)

                # Parse search_id
                parts = filename.replace('recording_', '').replace('.wav', '').split('_')
                search_id = parts[0] if parts else 'unknown'

                recordings.append({
                    'search_id': search_id,
                    'filename': filename,
                    'timestamp': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                    'size_bytes': file_stat.st_size,
                    'playback_url': f'/play-recording/{search_id}'
                })

        # Sort by timestamp (newest first)
        recordings.sort(key=lambda x: x['timestamp'], reverse=True)

        return jsonify({
            'success': True,
            'recordings': recordings[:10]  # Last 10
        })

    except Exception as e:
        logger.error(f"Error listing recordings: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@bp.route('/cleanup-recordings', methods=['POST'])
def cleanup_recordings():
    """Clean up old recordings"""
    try:
        data = request.get_json() or {}
        days_old = data.get('days_old', 7)

        cutoff_time = datetime.now() - timedelta(days=days_old)
        deleted_count = 0

        for filename in os.listdir(RECORDINGS_DIR):
            if filename.startswith('recording_') and filename.endswith('.wav'):
                file_path = os.path.join(RECORDINGS_DIR, filename)
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))

                if file_time < cutoff_time:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except OSError:
                        pass

        return jsonify({
            'success': True,
            'message': f'Deleted {deleted_count} old recordings',
            'deleted_count': deleted_count
        })

    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500
