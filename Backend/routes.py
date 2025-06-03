from flask import Flask, request, jsonify, Blueprint
import os
import tempfile
from models import Song, SongEmbedding
from database import db
import logging
import soundfile as sf
from datetime import datetime
from audio_utils import adaptive_similarity_search,find_similar_songs_multi_segment,load_youtube_track, get_embedding_from_file, save_song, load_youtube_playlist,convert_audio_with_ffmpeg,get_multiple_embeddings,preprocess_audio
from database import db

bp = Blueprint('main', __name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@bp.route('/')
def index():
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
def upload():
    """Upload song from local device"""
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

            save_song(track)
            from audio_utils import EMBEDDING_DURATION
            # Save embedding
            song_embedding = SongEmbedding(
                songID=track.songID,
                audioStart=0.0,
                audioDuration=min(duration, EMBEDDING_DURATION),
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
            logger.error(f"Error processing audio: {processing_error}")
            return jsonify({
                'success': False,
                'message': f'Error processing audio: {str(processing_error)}'
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
        logger.error(f"Upload error: {e}")
        return jsonify({
            'success': False,
            'message': f'Upload failed: {str(e)}'
        }), 500

@bp.route('/youtube', methods=['POST'])
def youtube():
    """Add a YouTube track to the database"""
    try:
        data = request.get_json()

        if not data or 'url' not in data:
            return jsonify({
                'success': False,
                'message': 'YouTube URL is required'
            }), 400

        url = data['url'].strip()

        # Basic validation
        if not url or 'youtube.com' not in url and 'youtu.be' not in url:
            return jsonify({
                'success': False,
                'message': 'Invalid YouTube URL'
            }), 400

        logger.info(f"Processing YouTube URL: {url}")

        # Process the YouTube track
        result = load_youtube_track(url)

        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400

    except Exception as e:
        logger.error(f"YouTube processing error: {e}")
        return jsonify({
            'success': False,
            'message': f'Error processing YouTube video: {str(e)}'
        }), 500

@bp.route('/youtube-playlist', methods=['POST'])
def upload_playlist():
    """Process YouTube playlist and add all videos to database."""
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


@bp.route('/search-audio', methods=['POST'])
def search_audio():
    """
    Enhanced audio search with preprocessing and multiple strategies.
    """
    try:
        if 'audio' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No audio file provided'
            }), 400

        audio_file = request.files['audio']

        # Save temporary file
        temp_dir = os.path.join(os.path.dirname(__file__), 'temp_uploads')
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"search_{datetime.now().timestamp()}.webm")
        audio_file.save(temp_path)

        try:
            # Convert to WAV first
            converted_path = convert_audio_with_ffmpeg(temp_path)
            if not converted_path:
                raise Exception("Failed to convert audio file")

            # Load and preprocess
            audio, sr = sf.read(converted_path)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            audio, sr = preprocess_audio(audio, sr)

            # Strategy 1: Single embedding from entire clip
            full_embedding, duration = get_embedding_from_file(converted_path)

            # Strategy 2: Multiple embeddings from segments
            segment_embeddings = get_multiple_embeddings(audio, sr)

            # Combine strategies
            if segment_embeddings:
                # Use multi-segment approach
                similar_songs = find_similar_songs_multi_segment(
                    segment_embeddings + [full_embedding],
                    limit=8
                )
            else:
                # Fall back to adaptive single embedding
                similar_songs = adaptive_similarity_search(full_embedding, limit=8)

            # Clean up temp files
            for path in [temp_path, converted_path]:
                if path and os.path.exists(path):
                    os.remove(path)

            if similar_songs:
                logger.info(f"Found {len(similar_songs)} similar songs")
                return jsonify({
                    'success': True,
                    'similar_songs': similar_songs,
                    'search_duration': duration,
                    'strategy_used': 'multi-segment' if segment_embeddings else 'single-embedding'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'No similar songs found. Try recording in a quieter environment or closer to the speaker.',
                    'similar_songs': []
                })

        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return jsonify({
                'success': False,
                'message': f'Audio processing failed: {str(e)}'
            }), 500

        finally:
            # Cleanup
            for path in [temp_path, converted_path if 'converted_path' in locals() else None]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass

    except Exception as e:
        logger.error(f"Search audio error: {e}")
        return jsonify({
            'success': False,
            'message': f'Search failed: {str(e)}'
        }), 500
