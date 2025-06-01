from flask import Flask, request, jsonify, Blueprint
import os
import tempfile
from audio_utils import get_embedding_from_file, save_song, load_youtube_playlist
from models import Song, SongEmbedding, db
import logging
from datetime import datetime

bp = Blueprint('main', __name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@bp.route('/')
def index():
    # will return JSON, but just setting up template to use in the future
    return jsonify({
        'message': "Hello from Jacob :))))))"
    })

@bp.route('/test')
def testCORS():
  return jsonify({
    'message': 'CORS is working if you see this',
    'status': "success"
  })


@bp.route('/upload', methods=["GET", "POST"])
def query():
    """
    upload song from local device
    """
    try:
        # check if file was in the request
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': 'file not found'
            }), 400

        # get file
        file = request.files['file']
        title = request.form.get('title', '').strip()
        artist = request.form.get('artist', '').strip()

        # make sure all fields are filled in
        if not file or file.filename == "":
            return jsonify({
                'success': False,
                'message': "Please submit a file"
            }), 400

        if not title or not artist:
            return jsonify({
                'success': False,
                'message': 'Title and Artist fields must be filled out'
            }), 400

        # create the file and save it to temp_uploads
        filename = file.filename
        temp_dir = os.path.join(os.path.dirname(__file__), 'temp_uploads')
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)

        try:
            # create embedding and save song + embedding to db
            embedding, duration = get_embedding_from_file(temp_path)

            track = Song(
                title=title,
                artist=artist,
                source='local upload',
                preview_url=None
            )

            save_song(track)  # use "track", not the undefined "song"

            # TODO: Currently don’t have save embedding method, build later
            song_embedding = SongEmbedding(
                songID=track.songID,
                audioStart=0.0,
                audioDuration=duration,
                embedding=embedding.tolist(),
                dimensions=512
            )

            db.session.add(song_embedding)
            db.session.commit()

            return jsonify({
                'success': True,
                'message': 'Processed successfully',
                'song': title,
                'songID': track.songID,
                'duration': duration
            }), 200

        except Exception as processing_error:
            # roll back if there is error
            db.session.rollback()
            return jsonify({
                'success': False,
                'message': f'Error processing audio: {processing_error}'
            }), 500

        finally:
            # delete temp file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    logger.info("Clean up success")
                except Exception:
                    logger.info("Clean up failed")
                    pass

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Upload failed: {e}'
        }), 500


@bp.route('/youtube-playlist', methods=['POST'])
def upload_playlist():
    """
    Process YouTube playlist and add all videos to database.
    Expects JSON: {"url": "playlist_url", "max_videos": 50}
    """
    try:
        # Validate JSON payload
        data = request.get_json()

        if not data or 'url' not in data:
            return jsonify({
                'success': False,
                'message': 'Playlist URL is required'
            }), 400

        playlist_url = data['url'].strip()
        max_videos = data.get('max_videos', 50)  # Default to 50

        # Validate inputs
        if not playlist_url:
            return jsonify({
                'success': False,
                'message': 'Playlist URL cannot be empty'
            }), 400

        # Basic URL validation
        if not any(x in playlist_url for x in ['playlist', 'youtube.com', 'youtu.be']):
            return jsonify({
                'success': False,
                'message': 'Invalid YouTube playlist URL format'
            }), 400

        # Validate max_videos range
        if not isinstance(max_videos, int) or max_videos < 1 or max_videos > 100:
            max_videos = 50  # Reset to default if invalid

        logger.info(f"Processing playlist: {playlist_url} (max: {max_videos})")

        # Process the playlist
        response = load_youtube_playlist(playlist_url, max_videos)

        # Separate successful and failed results
        successful = [r for r in response if r.get('success', False)]
        failed = [r for r in response if not r.get('success', False)]

        total_videos = len(response)
        successful_count = len(successful)
        failed_count = len(failed)

        logger.info(f"Playlist results: {successful_count}/{total_videos} successful")

        return jsonify({
            'success': True,
            'message': f'Processed {successful_count}/{total_videos} videos',
            'total_videos': total_videos,
            'successful_count': successful_count,
            'failed_count': failed_count,
            'successful_songs': successful,
            'failed_songs': failed,
            'summary': {
                'playlist_url': playlist_url,
                'max_videos_requested': max_videos,
                'processed_at': datetime.utcnow().isoformat()
            }
        })

    except KeyError as e:
        logger.error(f"Missing required field: {e}")
        return jsonify({
            'success': False,
            'message': f'Missing required field: {str(e)}'
        }), 400

    except Exception as e:
        logger.error(f"Playlist processing error: {e}")
        return jsonify({
            'success': False,
            'message': f'Playlist processing error: {str(e)}'
        }), 500


