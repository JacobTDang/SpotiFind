import numpy as np
from flask import Flask, request, jsonify, Blueprint
import os
import tempfile
from models import Song, SongEmbedding
from database import db
import logging
import soundfile as sf
from datetime import datetime
from typing import List, Dict
import openl3
from audio_utils import (
    find_similar_songs,
    load_youtube_track,
    get_embedding_from_file,
    save_song,
    load_youtube_playlist,
    convert_audio_with_ffmpeg
)

bp = Blueprint('main', __name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
EMBEDDING_DURATION = 30  # seconds

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
    Basic audio search endpoint - processes uploaded audio and finds similar songs.
    """
    converted_path = None
    try:
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

        logger.info(f"Processing audio search - filename: {audio_file.filename}")

        # Save temporary file
        temp_dir = os.path.join(os.path.dirname(__file__), 'temp_uploads')
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"search_{datetime.now().timestamp()}.webm")
        audio_file.save(temp_path)

        try:
            logger.info("Converting audio file...")
            # Convert to WAV first
            converted_path = convert_audio_with_ffmpeg(temp_path)
            if not converted_path:
                raise Exception("Failed to convert audio file")

            logger.info("Generating embedding...")
            # Generate embedding
            embedding, duration = get_embedding_from_file(converted_path)

            logger.info(f"Searching for similar songs (duration: {duration}s)...")
            # Find similar songs
            similar_songs = find_similar_songs(embedding, limit=10)

            logger.info(f"Found {len(similar_songs)} similar songs")

            if similar_songs:
                return jsonify({
                    'success': True,
                    'similar_songs': similar_songs,
                    'search_duration': duration,
                    'analysis': {
                        'segments_processed': 1,
                        'strategy': 'single-embedding'
                    }
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'No similar songs found. Try recording closer to the music source.',
                    'similar_songs': []
                })

        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return jsonify({
                'success': False,
                'message': f'Audio processing failed: {str(e)}'
            }), 500

        finally:
            # Cleanup temp files
            for path in [temp_path, converted_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        logger.debug(f"Cleaned up temp file: {path}")
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup {path}: {cleanup_error}")

    except Exception as e:
        logger.error(f"Search audio error: {e}")
        return jsonify({
            'success': False,
            'message': f'Search failed: {str(e)}'
        }), 500

@bp.route('/debug-vectors')
def debug_vectors():
    """Debug route to check vector storage and distances."""
    try:
        from models import Song, SongEmbedding
        import numpy as np

        # Get a few songs with embeddings
        songs_with_embeddings = db.session.query(Song, SongEmbedding).join(
            SongEmbedding, Song.songID == SongEmbedding.songID
        ).limit(3).all()

        debug_info = []

        for song, embedding in songs_with_embeddings:
            vector = embedding.embedding

            # Convert to numpy array to analyze
            if isinstance(vector, list):
                np_vector = np.array(vector)
            else:
                np_vector = np.array(vector)

            info = {
                'title': song.title,
                'artist': song.artist,
                'vector_type': str(type(vector)),
                'vector_shape': str(np_vector.shape),
                'vector_length': len(vector) if hasattr(vector, '__len__') else 'Unknown',
                'first_5_values': vector[:5] if hasattr(vector, '__getitem__') else 'Cannot slice',
                'magnitude': float(np.linalg.norm(np_vector)),
                'min_value': float(np.min(np_vector)),
                'max_value': float(np.max(np_vector)),
                'mean_value': float(np.mean(np_vector))
            }
            debug_info.append(info)

        # Test distance calculation manually
        if len(songs_with_embeddings) >= 2:
            vec1 = np.array(songs_with_embeddings[0][1].embedding)
            vec2 = np.array(songs_with_embeddings[1][1].embedding)

            # Calculate different distance metrics
            cosine_dist = 1 - np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            euclidean_dist = np.linalg.norm(vec1 - vec2)

            distance_test = {
                'song1': songs_with_embeddings[0][0].title,
                'song2': songs_with_embeddings[1][0].title,
                'cosine_distance': float(cosine_dist),
                'euclidean_distance': float(euclidean_dist),
                'expected_cosine_range': '0.0 to 2.0',
                'expected_euclidean_range': 'varies widely'
            }
        else:
            distance_test = {'error': 'Need at least 2 songs to test distances'}

        return jsonify({
            'debug_info': debug_info,
            'distance_test': distance_test,
            'recommendation': 'Cosine distances should be 0-2, very high values suggest normalization issues'
        })

    except Exception as e:
        return jsonify({'error': str(e)})

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    Normalize embedding to unit length for better cosine similarity.
    """
    norm = np.linalg.norm(embedding)
    if norm == 0:
        logger.warning("Zero norm embedding detected")
        return embedding
    return embedding / norm

def find_similar_songs_adaptive(embedding: np.ndarray, limit: int = 5) -> List[Dict]:
    """
    Adaptive similarity search that works regardless of distance scale.
    """
    try:
        # Ensure the query embedding is normalized
        embedding = normalize_embedding(embedding)
        embedding_list = embedding.tolist()

        connection = db.engine.raw_connection()
        cursor = connection.cursor()

        # Get all results sorted by distance (no filtering)
        sql = """
        SELECT s."songID", s.title, s.artist, s.source, s.youtube_id,
               (se.embedding <-> %s::vector) as distance
        FROM songs s
        JOIN song_embeddings se ON s."songID" = se."songID"
        ORDER BY se.embedding <-> %s::vector
        LIMIT %s
        """

        cursor.execute(sql, (embedding_list, embedding_list, limit))
        rows = cursor.fetchall()

        similar_songs = []
        for i, row in enumerate(rows):
            distance = float(row[5])

            # Calculate a relative confidence based on ranking
            # Best match gets highest confidence, others scaled down
            if i == 0:
                confidence = 85  # Give best match high confidence
            else:
                # Scale confidence based on distance ratio to best match
                best_distance = float(rows[0][5])
                if best_distance > 0:
                    distance_ratio = distance / best_distance
                    confidence = max(10, 85 / distance_ratio)  # Scale down based on ratio
                else:
                    confidence = 85 - (i * 10)  # Simple linear scaling

            logger.info(f"Rank {i+1}: '{row[1]}' by {row[2]} - distance: {distance:.3f}, confidence: {confidence:.1f}%")

            similar_songs.append({
                'songID': row[0],
                'title': row[1],
                'artist': row[2],
                'source': row[3],
                'youtube_id': row[4],
                'distance': distance,
                'confidence': min(95, confidence)  # Cap at 95%
            })

        cursor.close()
        connection.close()

        return similar_songs

    except Exception as e:
        logger.error(f"Error finding similar songs: {e}")
        return []

@bp.route('/normalize-embeddings')
def normalize_existing_embeddings():
    """One-time fix to normalize all existing embeddings."""
    try:
        from models import SongEmbedding
        import numpy as np

        embeddings = SongEmbedding.query.all()
        updated_count = 0

        for emb in embeddings:
            original_vector = np.array(emb.embedding)
            original_norm = np.linalg.norm(original_vector)

            if original_norm > 0 and abs(original_norm - 1.0) > 0.001:  # Only update if not already normalized
                normalized_vector = original_vector / original_norm
                emb.embedding = normalized_vector.tolist()
                updated_count += 1

                logger.info(f"Normalized embedding {emb.embeddingID}: norm {original_norm:.3f} -> 1.0")

        db.session.commit()

        return jsonify({
            'success': True,
            'message': f'Normalized {updated_count} embeddings',
            'updated_count': updated_count
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)})
@bp.route('/fix-embeddings', methods=['POST'])
def fix_embeddings():
    """
    One-time endpoint to normalize all existing embeddings in the database.
    This fixes the high distance issue.
    """
    try:
        from audio_utils import normalize_embedding
        import numpy as np

        # Get all embeddings
        embeddings = SongEmbedding.query.all()
        updated_count = 0

        logger.info(f"Found {len(embeddings)} embeddings to check")

        for emb in embeddings:
            try:
                # Convert to numpy array
                original_vector = np.array(emb.embedding)
                original_norm = np.linalg.norm(original_vector)

                # Check if normalization is needed
                if abs(original_norm - 1.0) > 0.01:  # Not normalized
                    # Normalize the vector
                    normalized_vector = original_vector / original_norm

                    # Verify normalization
                    new_norm = np.linalg.norm(normalized_vector)

                    # Update in database
                    emb.embedding = normalized_vector.tolist()
                    updated_count += 1

                    logger.info(f"Normalized embedding {emb.embeddingID} for song {emb.songID}: "
                               f"norm {original_norm:.3f} -> {new_norm:.3f}")
                else:
                    logger.debug(f"Embedding {emb.embeddingID} already normalized (norm: {original_norm:.3f})")

            except Exception as e:
                logger.error(f"Error processing embedding {emb.embeddingID}: {e}")
                continue

        # Commit all changes
        if updated_count > 0:
            db.session.commit()
            logger.info(f"Successfully updated {updated_count} embeddings")

        return jsonify({
            'success': True,
            'message': f'Normalized {updated_count} out of {len(embeddings)} embeddings',
            'total_embeddings': len(embeddings),
            'updated_count': updated_count
        })

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error fixing embeddings: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@bp.route('/check-embedding-norms', methods=['GET'])
def check_embedding_norms():
    """
    Debug endpoint to check the norms of embeddings in the database.
    """
    try:
        import numpy as np

        # Get a sample of embeddings
        embeddings = SongEmbedding.query.limit(10).all()

        norm_info = []
        for emb in embeddings:
            vector = np.array(emb.embedding)
            norm = np.linalg.norm(vector)

            # Get song info
            song = Song.query.get(emb.songID)

            norm_info.append({
                'embeddingID': emb.embeddingID,
                'songID': emb.songID,
                'title': song.title if song else 'Unknown',
                'artist': song.artist if song else 'Unknown',
                'norm': float(norm),
                'is_normalized': abs(norm - 1.0) < 0.01
            })

        # Calculate statistics
        norms = [info['norm'] for info in norm_info]
        stats = {
            'min_norm': float(np.min(norms)),
            'max_norm': float(np.max(norms)),
            'mean_norm': float(np.mean(norms)),
            'std_norm': float(np.std(norms))
        }

        return jsonify({
            'success': True,
            'embeddings': norm_info,
            'statistics': stats,
            'recommendation': 'Run /fix-embeddings if norms are not close to 1.0'
        })

    except Exception as e:
        logger.error(f"Error checking norms: {e}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500

@bp.route('/test-similarity/<int:song_id>', methods=['GET'])
def test_similarity(song_id):
    """Test similarity search for a specific song"""
    try:
        # Get the song and its embedding
        song = Song.query.get(song_id)
        if not song:
            return jsonify({'error': 'Song not found'}), 404

        embedding = SongEmbedding.query.filter_by(songID=song_id).first()
        if not embedding:
            return jsonify({'error': 'No embedding found for this song'}), 404

        # Find similar songs (excluding itself)
        similar = find_similar_songs(
            np.array(embedding.embedding),
            limit=10,
            exclude_id=song_id
        )

        return jsonify({
            'query_song': {
                'id': song.songID,
                'title': song.title,
                'artist': song.artist
            },
            'similar_songs': similar,
            'debug_info': {
                'embedding_norm': float(np.linalg.norm(embedding.embedding)),
                'embedding_mean': float(np.mean(embedding.embedding)),
                'embedding_std': float(np.std(embedding.embedding))
            }
        })

    except Exception as e:
        logger.error(f"Error in test similarity: {e}")
        return jsonify({'error': str(e)}), 500
