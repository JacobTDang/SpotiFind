import os
import tempfile
import logging
from typing import Dict, Optional, List, Tuple
from dotenv import load_dotenv
from models import Song, SongEmbedding, db
import openl3
import soundfile as sf
import numpy as np
import yt_dlp as yt
import subprocess
import librosa

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
EMBEDDING_DURATION = 60  # Duration in seconds for embeddings
SAMPLE_RATE = 22050     # Standard sample rate for audio processing

# Create backend temp directory if it doesn't exist
BACKEND_TEMP_DIR = os.path.join(os.path.dirname(__file__), 'temp_audio')
os.makedirs(BACKEND_TEMP_DIR, exist_ok=True)

def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is available on the system."""
    try:
        subprocess.run(['ffmpeg', '-version'],
                      capture_output=True, check=True, timeout=5)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False

# Check FFmpeg availability
FFMPEG_AVAILABLE = check_ffmpeg_available()
logger.info(f"FFmpeg available: {FFMPEG_AVAILABLE}")

# Configure yt_dlp to only fetch metadata (no download) when asked
YDL_META_OPTS = {
    'quiet': True,
    'skip_download': True,
    'extract_flat': False,
}

# Configure yt_dlp to always use FFmpeg for conversion since it's available
YDL_AUDIO_OPTS = {
    'quiet': True,
    'format': 'bestaudio/best',
    'outtmpl': os.path.join(BACKEND_TEMP_DIR, '%(id)s.%(ext)s'),
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192',
    }],
}

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    Normalize embedding to unit length for proper cosine similarity.
    This is CRITICAL for pgvector's distance calculations.
    """
    # Ensure it's a numpy array
    if not isinstance(embedding, np.ndarray):
        embedding = np.array(embedding)

    # Calculate L2 norm
    norm = np.linalg.norm(embedding)

    if norm == 0:
        logger.warning("Zero norm embedding detected - returning random unit vector")
        # Return a random unit vector instead of zero vector
        random_vec = np.random.randn(embedding.shape[0])
        return random_vec / np.linalg.norm(random_vec)

    # Normalize to unit length
    normalized = embedding / norm

    # Verify normalization worked
    final_norm = np.linalg.norm(normalized)
    if abs(final_norm - 1.0) > 0.01:
        logger.warning(f"Normalization may have failed. Final norm: {final_norm}")

    return normalized

def preprocess_audio(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    """
    Preprocess audio for better embedding quality.

    Args:
        audio: Audio signal array
        sr: Sample rate

    Returns:
        Preprocessed audio and sample rate
    """
    try:
        # Resample to standard rate if needed
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        # Normalize audio to [-1, 1] range
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # Remove silence at beginning and end
        audio, _ = librosa.effects.trim(audio, top_db=20)

        # Apply pre-emphasis filter to boost high frequencies
        audio = librosa.effects.preemphasis(audio)

        return audio, sr
    except Exception as e:
        logger.warning(f"Audio preprocessing failed: {e}")
        return audio, sr

def verify_embedding_in_db(embedding_id: int):
    """Debug function to verify an embedding is properly normalized in the database."""
    try:
        embedding_record = SongEmbedding.query.get(embedding_id)
        if embedding_record:
            vec = np.array(embedding_record.embedding)
            norm = np.linalg.norm(vec)
            logger.info(f"Embedding {embedding_id} norm: {norm:.6f} (should be ~1.0)")
            return norm
    except Exception as e:
        logger.error(f"Error verifying embedding: {e}")
    return None

def get_embedding_from_file(path: str, max_duration: int = EMBEDDING_DURATION) -> Tuple[np.ndarray, float]:
    """Load audio file, generate OpenL3 embedding, return NORMALIZED embedding and duration."""
    try:
        working_path = path

        # If it's not a WAV file, convert it using FFmpeg
        if not path.endswith('.wav') and FFMPEG_AVAILABLE:
            converted_path = convert_audio_with_ffmpeg(path)
            if converted_path and os.path.exists(converted_path):
                working_path = converted_path
                logger.debug(f"Using converted file: {working_path}")

        # Load the audio file
        audio, sr = sf.read(working_path)

        # Handle stereo to mono conversion
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Apply preprocessing
        try:
            audio, sr = preprocess_audio(audio, sr)
        except Exception as e:
            logger.warning(f"Preprocessing failed, using original audio: {e}")

        # Get actual duration
        actual_duration = len(audio) / sr

        # Trim to max_duration seconds for consistent analysis
        if actual_duration > max_duration:
            audio = audio[:int(max_duration * sr)]
            analysis_duration = max_duration
        else:
            analysis_duration = actual_duration

        logger.info(f"Generating embedding for {analysis_duration:.1f}s of audio...")

        # Generate embedding using correct OpenL3 API
        embeddings, timestamps = openl3.get_audio_embedding(
            audio, sr,
            content_type='music',
            embedding_size=512,
            hop_size=0.5,
            verbose=0  # Reduce TensorFlow verbosity
        )

        logger.info(f"Generated {len(embeddings)} time-step embeddings")

        # Average all time-step embeddings into single 512-dim vector
        embedding = np.mean(embeddings, axis=0)

        # Log embedding stats before normalization
        pre_norm = np.linalg.norm(embedding)
        logger.info(f"Pre-normalization embedding norm: {pre_norm:.6f}")

        # CRITICAL: Normalize the embedding for proper cosine similarity
        embedding = normalize_embedding(embedding)

        # Verify normalization
        post_norm = np.linalg.norm(embedding)
        logger.info(f"Post-normalization embedding norm: {post_norm:.6f}")

        # Cleanup converted file if we created one
        if working_path != path and os.path.exists(working_path):
            os.remove(working_path)

        return embedding, actual_duration

    except Exception as e:
        logger.error(f"Error generating embedding from {path}: {e}")
        raise

def get_multiple_embeddings(path: str, segment_duration: float = 10.0, overlap: float = 5.0) -> List[Tuple[np.ndarray, float, float]]:
    """
    Generate multiple embeddings from different segments of the audio.

    Args:
        path: Path to audio file
        segment_duration: Duration of each segment in seconds
        overlap: Overlap between segments in seconds

    Returns:
        List of (embedding, start_time, end_time) tuples
    """
    embeddings = []

    try:
        # Load audio
        audio, sr = sf.read(path)

        # Convert to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Apply preprocessing
        audio, sr = preprocess_audio(audio, sr)

        total_duration = len(audio) / sr
        segment_samples = int(segment_duration * sr)
        overlap_samples = int(overlap * sr)
        step_samples = segment_samples - overlap_samples

        # Generate embeddings for each segment
        for start_sample in range(0, len(audio) - segment_samples + 1, step_samples):
            end_sample = start_sample + segment_samples
            segment = audio[start_sample:end_sample]

            start_time = start_sample / sr
            end_time = end_sample / sr

            # Generate embedding for this segment
            segment_embeddings, _ = openl3.get_audio_embedding(
                segment, sr,
                content_type='music',
                embedding_size=512,
                hop_size=0.5
            )

            # Average and normalize
            embedding = np.mean(segment_embeddings, axis=0)
            embedding = normalize_embedding(embedding)

            embeddings.append((embedding, start_time, end_time))

            logger.debug(f"Generated embedding for segment {start_time:.1f}s - {end_time:.1f}s")

        return embeddings

    except Exception as e:
        logger.error(f"Error generating multiple embeddings: {e}")
        return []

def convert_audio_with_ffmpeg(input_path: str) -> Optional[str]:
    """Convert audio file to WAV using FFmpeg."""
    try:
        # Create output filename
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(BACKEND_TEMP_DIR, f"{base_name}_converted.wav")

        cmd = [
            'ffmpeg', '-i', input_path,
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '22050',          # Sample rate compatible with OpenL3
            '-ac', '1',              # Convert to mono
            '-y',                    # Overwrite output
            output_path
        ]

        logger.debug(f"Converting {input_path} with command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, timeout=60)

        if result.returncode == 0 and os.path.exists(output_path):
            logger.debug(f"Successfully converted to {output_path}")
            return output_path
        else:
            error_msg = result.stderr.decode() if result.stderr else "Unknown error"
            logger.error(f"FFmpeg conversion failed: {error_msg}")
            return None

    except subprocess.TimeoutExpired:
        logger.error("FFmpeg conversion timed out")
        return None
    except Exception as e:
        logger.error(f"Error converting audio with FFmpeg: {e}")
        return None

def save_song(song: Song) -> None:
    """Add song to session and flush to get songID."""
    db.session.add(song)
    db.session.flush()

def load_youtube_track(video_url: str) -> Dict:
    """Fetch metadata, save Song, download + embed, commit to DB."""
    temp_audio = None

    try:
        # Extract metadata
        with yt.YoutubeDL(YDL_META_OPTS) as ydl:
            info = ydl.extract_info(video_url, download=False)

        youtube_id = info['id']
        logger.info(f"Processing YouTube video: {info.get('title')} ({youtube_id})")

        # Check if already exists
        existing = db.session.query(Song).filter_by(youtube_id=youtube_id).first()
        if existing:
            return {
                'success': False,
                'song': existing.title,
                'message': 'Song already in database',
                'songID': existing.songID
            }

        # Validate video (skip live streams, too long videos, etc.)
        if not is_valid_audio_content(info):
            return {
                'success': False,
                'song': info.get('title', 'Unknown'),
                'message': 'Invalid content type (live stream, too long, etc.)'
            }

        # Create Song instance
        yt_song = Song(
            title=clean_title(info.get('title', 'Unknown Title')),
            artist=info.get('uploader', 'Unknown Artist'),
            source='youtube',
            youtube_id=youtube_id,
            preview_url=video_url,
        )

        # Save song to get ID
        save_song(yt_song)

        # Download audio
        temp_audio = download_youtube_audio(info)
        if not temp_audio:
            db.session.rollback()
            return {
                'success': False,
                'song': yt_song.title,
                'message': 'Audio download failed'
            }

        # Generate embedding
        embedding, actual_duration = get_embedding_from_file(temp_audio)

        # Create embedding record
        song_emb = SongEmbedding(
            songID=yt_song.songID,
            audioStart=0.0,
            audioDuration=min(actual_duration, EMBEDDING_DURATION),
            embedding=embedding.tolist(),
            dimensions=len(embedding),
        )

        db.session.add(song_emb)
        db.session.commit()

        logger.info(f"Successfully added: {yt_song.title}")
        return {
            'success': True,
            'song': yt_song.title,
            'message': 'Successfully added from YouTube',
            'songID': yt_song.songID,
            'duration': actual_duration
        }

    except yt.DownloadError as e:
        db.session.rollback()
        logger.error(f"YouTube download error: {e}")
        return {
            'success': False,
            'song': info.get('title', 'Unknown') if 'info' in locals() else 'Unknown',
            'message': f'Download error: {str(e)}'
        }
    except Exception as e:
        db.session.rollback()
        logger.error(f"Unexpected error processing YouTube track: {e}")
        return {
            'success': False,
            'song': info.get('title', 'Unknown') if 'info' in locals() else 'Unknown',
            'message': f'Processing error: {str(e)}'
        }
    finally:
        # Cleanup temporary file
        if temp_audio and os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
                logger.debug(f"Cleaned up temp file: {temp_audio}")
            except OSError as e:
                logger.warning(f"Failed to cleanup temp file {temp_audio}: {e}")

def download_youtube_audio(info: dict) -> Optional[str]:
    """Download audio from YouTube video info and return temp file path."""
    try:
        with yt.YoutubeDL(YDL_AUDIO_OPTS) as ydl:
            result = ydl.extract_info(info['webpage_url'], download=True)

        # Since we're using FFmpeg postprocessor, look for the converted file
        video_id = result['id']

        # FFmpeg should have converted to WAV
        wav_path = os.path.join(BACKEND_TEMP_DIR, f"{video_id}.wav")
        if os.path.exists(wav_path):
            logger.debug(f"Downloaded and converted audio to: {wav_path}")
            return wav_path

        # Fallback: look for any file with this video ID
        for filename in os.listdir(BACKEND_TEMP_DIR):
            if filename.startswith(video_id):
                path = os.path.join(BACKEND_TEMP_DIR, filename)
                logger.debug(f"Found downloaded file: {path}")
                return path

        logger.error(f"Could not find downloaded file for video {video_id}")
        return None

    except Exception as e:
        logger.error(f"Error downloading audio: {e}")
        return None

def is_valid_audio_content(info: dict) -> bool:
    """Check if YouTube video is suitable for audio analysis."""
    # Skip live streams
    if info.get('is_live', False):
        return False

    # Skip very short videos (< 10 seconds)
    duration = info.get('duration', 0)
    if duration and duration < 10:
        return False

    # Skip very long videos (> 10 minutes) - likely not music
    if duration and duration > 600:
        return False

    return True

def clean_title(title: str) -> str:
    """Clean up video title for better song matching."""
    cleaners = [
        '(Official Video)', '(Official Music Video)', '(Official Audio)',
        '(Lyric Video)', '(Lyrics)', '[Official Video]', '[Official Audio]',
        'HD', '4K', '| Official Video'
    ]

    cleaned = title
    for cleaner in cleaners:
        cleaned = cleaned.replace(cleaner, '')

    return cleaned.strip()

def load_youtube_playlist(playlist_url: str, max_videos: int = 50) -> List[Dict]:
    """Process YouTube playlist, adding each video to database."""
    try:
        with yt.YoutubeDL(YDL_META_OPTS) as ydl:
            playlist_info = ydl.extract_info(playlist_url, download=False)

        entries = playlist_info.get('entries', [])
        logger.info(f"Found {len(entries)} videos in playlist")

        results = []
        processed = 0

        for entry in entries[:max_videos]:
            if not entry:
                continue

            video_url = entry.get('webpage_url')
            if not video_url:
                continue

            logger.info(f"Processing {processed + 1}/{min(len(entries), max_videos)}: {entry.get('title', 'Unknown')}")

            result = load_youtube_track(video_url)
            results.append(result)
            processed += 1

            # Log progress
            status = "✓" if result['success'] else "✗"
            print(f"{status} {result['message']}: {result['song']}")

        logger.info(f"Playlist processing complete. {sum(1 for r in results if r['success'])}/{len(results)} successful")
        return results

    except Exception as e:
        logger.error(f"Error processing playlist: {e}")
        return [{'success': False, 'message': f'Playlist error: {str(e)}', 'song': 'Unknown'}]

def find_similar_songs(embedding: np.ndarray, limit: int = 5) -> List[Dict]:
    """Find songs similar to given embedding using pgvector cosine similarity."""
    try:
        # ENSURE the query embedding is normalized
        query_embedding = normalize_embedding(embedding)
        query_norm = np.linalg.norm(query_embedding)
        logger.info(f"Query embedding norm: {query_norm:.6f}")

        embedding_list = query_embedding.tolist()

        # Use raw connection approach with properly quoted column names
        connection = db.engine.raw_connection()
        cursor = connection.cursor()

        # First, let's check if our embeddings in the DB are normalized
        check_sql = """
        SELECT COUNT(*), AVG(sqrt(sum(power(elem, 2)))) as avg_norm
        FROM (
            SELECT unnest(embedding::float[]) as elem, "songID"
            FROM song_embeddings
        ) t
        GROUP BY "songID"
        LIMIT 5
        """

        cursor.execute(check_sql)
        norm_check = cursor.fetchall()
        if norm_check:
            avg_norms = [row[1] for row in norm_check if row[1] is not None]
            if avg_norms:
                db_avg_norm = np.mean(avg_norms)
                logger.info(f"Average norm of DB embeddings: {db_avg_norm:.6f}")

        sql = """
        SELECT s."songID", s.title, s.artist, s.source, s.youtube_id,
               (se.embedding <-> %s::vector) as distance,
               sqrt(sum(power(unnest(se.embedding::float[]), 2))) as embedding_norm
        FROM songs s
        JOIN song_embeddings se ON s."songID" = se."songID"
        GROUP BY s."songID", s.title, s.artist, s.source, s.youtube_id, se.embedding
        ORDER BY se.embedding <-> %s::vector
        LIMIT %s
        """

        cursor.execute(sql, (embedding_list, embedding_list, limit))
        rows = cursor.fetchall()

        similar_songs = []
        for i, row in enumerate(rows):
            # Log the embedding norm for debugging
            if len(row) > 6:
                logger.debug(f"DB embedding norm for '{row[1]}': {row[6]:.6f}")

            distance = float(row[5])

            # For normalized vectors, cosine distance should be between 0 and 2
            # 0 = identical, 1 = orthogonal, 2 = opposite
            # Your distances of ~57 suggest vectors are not normalized

            # Convert distance to similarity score (0-100%)
            # For properly normalized vectors:
            similarity = max(0, (2 - distance) / 2) * 100

            similar_songs.append({
                'songID': row[0],
                'title': row[1],
                'artist': row[2],
                'source': row[3],
                'youtube_id': row[4],
                'distance': distance,
                'similarity': round(similarity, 1)
            })

            logger.info(f"Rank {i+1}: '{row[1]}' - distance: {distance:.3f}, similarity: {similarity:.1f}%")

        cursor.close()
        connection.close()

        return similar_songs

    except Exception as e:
        logger.error(f"Error finding similar songs: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return []

def normalize_all_embeddings_in_db():
    """One-time utility to normalize all existing embeddings in the database."""
    try:
        embeddings = SongEmbedding.query.all()
        updated_count = 0

        for emb in embeddings:
            original_vector = np.array(emb.embedding)
            original_norm = np.linalg.norm(original_vector)

            # Only update if not already normalized
            if abs(original_norm - 1.0) > 0.01:
                normalized_vector = normalize_embedding(original_vector)
                emb.embedding = normalized_vector.tolist()
                updated_count += 1

                logger.info(f"Normalized embedding {emb.embeddingID}: {original_norm:.3f} -> 1.0")

        db.session.commit()
        logger.info(f"Successfully normalized {updated_count} embeddings")
        return updated_count

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error normalizing embeddings: {e}")
        return 0
def adaptive_similarity_search(embedding: np.ndarray, limit: int = 10) -> List[Dict]:
    """
    Adaptive similarity search that adjusts confidence based on distance distribution.
    """
    try:
        # Ensure embedding is normalized
        embedding = normalize_embedding(embedding)

        # Get similar songs
        results = find_similar_songs(embedding, limit)

        if not results:
            return []

        # Calculate confidence scores based on distance distribution
        distances = [r['distance'] for r in results]
        min_dist = min(distances)
        max_dist = max(distances)

        for i, result in enumerate(results):
            distance = result['distance']

            # Calculate confidence (inverse of normalized distance)
            if max_dist - min_dist > 0:
                normalized_dist = (distance - min_dist) / (max_dist - min_dist)
                confidence = (1 - normalized_dist) * 85 + 10  # Scale to 10-95%
            else:
                # All distances are the same
                confidence = 85

            result['confidence'] = round(confidence, 1)

            logger.debug(f"Song {i+1}: {result['title']} - distance: {distance:.3f}, confidence: {confidence:.1f}%")

        return results

    except Exception as e:
        logger.error(f"Error in adaptive similarity search: {e}")
        return []

def find_similar_songs_adaptive(embedding: np.ndarray, limit: int = 10) -> List[Dict]:
    """
    Adaptive similarity search with proper confidence calculation.
    """
    try:
        # Get similar songs
        results = find_similar_songs(embedding, limit)

        if not results:
            return []

        # Since distances are very high (~57), let's use a different confidence calculation
        distances = [r['distance'] for r in results]
        min_dist = min(distances)
        max_dist = max(distances)

        logger.info(f"Distance range: {min_dist:.3f} - {max_dist:.3f}")

        for i, result in enumerate(results):
            distance = result['distance']

            # For these high distances, use relative ranking
            if i == 0:
                confidence = 85.0  # Best match
            else:
                # Calculate relative confidence based on distance from best match
                distance_ratio = (distance - min_dist) / (max_dist - min_dist) if max_dist > min_dist else 0
                confidence = max(50, 85 - (distance_ratio * 35))

            result['confidence'] = round(confidence, 1)

        return results

    except Exception as e:
        logger.error(f"Error in adaptive similarity search: {e}")
        return []
