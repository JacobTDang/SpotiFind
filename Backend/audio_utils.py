import os
import time
import logging
from typing import Dict, Optional, List, Tuple

# Core dependencies
from models import Song, SongEmbedding, db
import numpy as np
import soundfile as sf
import subprocess

# Audio processing
import openl3
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# YouTube downloading
import yt_dlp as yt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

SAMPLE_RATE = 22050
MAX_PLAYLIST_SIZE = 100
BACKEND_TEMP_DIR = os.path.join(os.path.dirname(__file__), 'temp_audio')
os.makedirs(BACKEND_TEMP_DIR, exist_ok=True)

# Check FFmpeg availability
def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is available on the system."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True, timeout=5)
        return result.returncode == 0
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False

FFMPEG_AVAILABLE = check_ffmpeg_available()

# YT-DLP Configuration
YDL_META_OPTS = {
    'quiet': True,
    'skip_download': True,
    'extract_flat': False,
    'no_warnings': True,
}

YDL_PLAYLIST_OPTS = {
    'quiet': True,
    'skip_download': True,
    'extract_flat': True,
    'ignoreerrors': True,
    'no_warnings': True,
    'playlistend': MAX_PLAYLIST_SIZE,
}

if FFMPEG_AVAILABLE:
    YDL_AUDIO_OPTS = {
        'quiet': True,
        'format': 'bestaudio[ext=m4a]/bestaudio/best',
        'outtmpl': os.path.join(BACKEND_TEMP_DIR, '%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'ignoreerrors': False,
    }
else:
    YDL_AUDIO_OPTS = {
        'quiet': True,
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(BACKEND_TEMP_DIR, '%(id)s.%(ext)s'),
        'ignoreerrors': False,
    }

# =============================================================================
# CORE AUDIO PROCESSING
# =============================================================================

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalize embedding to unit length for proper similarity calculations."""
    if not isinstance(embedding, np.ndarray):
        embedding = np.array(embedding)

    norm = np.linalg.norm(embedding)
    if norm == 0:
        logger.warning("Zero norm embedding detected - returning random unit vector")
        random_vec = np.random.randn(embedding.shape[0])
        return random_vec / np.linalg.norm(random_vec)

    return embedding / norm

def preprocess_audio(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    """Preprocess audio for better embedding quality."""
    try:
        # Resample if needed
        if sr != SAMPLE_RATE and LIBROSA_AVAILABLE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        # Normalize to [-1, 1]
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # Trim silence
        if LIBROSA_AVAILABLE:
            audio, _ = librosa.effects.trim(audio, top_db=20)

        return audio, sr
    except Exception as e:
        logger.warning(f"Audio preprocessing failed: {e}")
        return audio, sr

def convert_audio_with_ffmpeg(input_path: str) -> Optional[str]:
    """Convert audio file to WAV using FFmpeg."""
    if not FFMPEG_AVAILABLE:
        return None

    try:
        # Extract base filename without extension
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(BACKEND_TEMP_DIR, f"{base_name}_converted.wav")

        # FFmpeg command
        cmd = [
            'ffmpeg', '-i', input_path,
            '-acodec', 'pcm_s16le',
            '-ar', str(SAMPLE_RATE),
            '-ac', '1',  # Convert to mono
            '-y',  # Overwrite output
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, timeout=60)

        if result.returncode == 0 and os.path.exists(output_path):
            return output_path
        else:
            logger.error("FFmpeg conversion failed")
            return None

    except Exception as e:
        logger.error(f"Error converting audio: {e}")
        return None

def get_embedding_from_file(path: str, max_duration: int = 60) -> Tuple[np.ndarray, float]:
    """
    Generate OpenL3 embeddings from audio file.

    Args:
        path: Audio file path
        max_duration: Maximum duration to process (default: 60 seconds)

    Returns:
        Tuple of (512-dimensional embedding, actual_duration)
    """
    working_path = path
    converted_file = None

    try:
        # Convert to WAV if needed
        if not path.lower().endswith('.wav') and FFMPEG_AVAILABLE:
            converted_file = convert_audio_with_ffmpeg(path)
            if converted_file and os.path.exists(converted_file):
                working_path = converted_file

        # Load audio
        audio, sr = sf.read(working_path)

        # Convert stereo to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        actual_duration = len(audio) / sr
        logger.info(f"Processing {actual_duration:.1f}s audio file")

        # Trim to max_duration if needed
        if actual_duration > max_duration:
            # Take middle portion for longer recordings
            start_offset = int((actual_duration - max_duration) * sr / 2)
            audio = audio[start_offset : start_offset + int(max_duration * sr)]
            logger.info(f"Trimmed to {max_duration}s from middle of recording")

        # Preprocess
        audio, sr = preprocess_audio(audio, sr)

        # Generate OpenL3 embeddings
        embeddings, _ = openl3.get_audio_embedding(
            audio, sr,
            content_type='music',
            embedding_size=512,
            hop_size=0.1  # 100ms hop for temporal resolution
        )

        if len(embeddings) == 0:
            raise RuntimeError("No embedding frames generated")

        logger.info(f"Generated {len(embeddings)} embedding frames")

        # Aggregate embeddings
        n_frames = len(embeddings)

        if n_frames <= 10:
            # Short clips: simple mean
            embedding = np.mean(embeddings, axis=0)
        else:
            # Longer clips: weighted mean emphasizing middle
            weights = np.ones(n_frames)

            # Emphasize middle 60%
            start_emphasis = int(0.2 * n_frames)
            end_emphasis = int(0.8 * n_frames)
            weights[start_emphasis:end_emphasis] *= 1.2

            # Normalize weights
            weights = weights / weights.sum()

            # Weighted average
            embedding = np.average(embeddings, axis=0, weights=weights)

        # Normalize to unit length
        embedding = normalize_embedding(embedding)

        return embedding, min(actual_duration, max_duration)

    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise
    finally:
        # Always cleanup converted file
        if converted_file and converted_file != path and os.path.exists(converted_file):
            try:
                os.remove(converted_file)
            except OSError:
                pass

# =============================================================================
# DATABASE UTILITIES
# =============================================================================

def save_song(song: Song) -> None:
    """Add song to database and flush to get ID."""
    db.session.add(song)
    db.session.flush()  # Get songID without committing
    logger.debug(f"Song saved with ID: {song.songID}")

# =============================================================================
# YOUTUBE PROCESSING
# =============================================================================

def is_valid_audio_content(info: dict) -> bool:
    """Check if YouTube video is suitable for audio analysis."""
    # Skip live streams
    if info.get('is_live', False):
        return False

    # Check duration (10s to 10min)
    duration = info.get('duration')
    if duration is None or duration < 10 or duration > 600:
        return False

    # Skip non-music categories
    category = info.get('category', '').lower()
    if category in ['news & politics', 'comedy', 'education', 'science & technology']:
        return False

    return True

def clean_title(title: str) -> str:
    """Clean up video title."""
    if not title:
        return "Unknown Title"

    # Remove common suffixes
    cleaners = [
        '(Official Video)', '(Official Music Video)', '(Official Audio)',
        '(Lyric Video)', '(Lyrics)', '[Official Video]', '[Official Audio]',
        '(HD)', '(4K)', '| Official Video', '(Music Video)'
    ]

    cleaned = title
    for cleaner in cleaners:
        cleaned = cleaned.replace(cleaner, '')

    return cleaned.strip()

def download_youtube_audio(info: dict) -> Optional[str]:
    """Download audio from YouTube video."""
    try:
        video_id = info.get('id')
        if not video_id:
            return None

        with yt.YoutubeDL(YDL_AUDIO_OPTS) as ydl:
            ydl.extract_info(info['webpage_url'], download=True)

        # Look for downloaded file
        expected_wav = os.path.join(BACKEND_TEMP_DIR, f"{video_id}.wav")
        if os.path.exists(expected_wav):
            return expected_wav

        # Find any file with video ID
        for filename in os.listdir(BACKEND_TEMP_DIR):
            if filename.startswith(video_id):
                return os.path.join(BACKEND_TEMP_DIR, filename)

        return None

    except Exception as e:
        logger.error(f"Error downloading audio: {e}")
        return None

def load_youtube_track(video_url: str) -> Dict:
    """Process single YouTube video."""
    temp_audio = None

    try:
        # Extract metadata
        with yt.YoutubeDL(YDL_META_OPTS) as ydl:
            info = ydl.extract_info(video_url, download=False)

        youtube_id = info.get('id')
        title = info.get('title', 'Unknown Title')

        if not youtube_id:
            return {'success': False, 'song': title, 'message': 'Could not extract video ID'}

        logger.info(f"Processing: {title}")

        # Check if exists
        existing = db.session.query(Song).filter_by(youtube_id=youtube_id).first()
        if existing:
            return {
                'success': False,
                'song': existing.title,
                'message': 'Song already in database',
                'songID': existing.songID
            }

        # Validate content
        if not is_valid_audio_content(info):
            return {
                'success': False,
                'song': title,
                'message': 'Invalid content (live/duration/category)'
            }

        # Create Song record
        song = Song(
            title=clean_title(title),
            artist=info.get('uploader', 'Unknown Artist')[:255],
            source='youtube',
            youtube_id=youtube_id,
            preview_url=video_url,
        )
        save_song(song)

        # Download audio
        temp_audio = download_youtube_audio(info)
        if not temp_audio:
            db.session.rollback()
            return {'success': False, 'song': song.title, 'message': 'Download failed'}

        # Generate embedding
        try:
            embedding, duration = get_embedding_from_file(temp_audio)
        except Exception as e:
            db.session.rollback()
            return {'success': False, 'song': song.title, 'message': f'Embedding failed: {str(e)}'}

        # Create embedding record
        song_embedding = SongEmbedding(
            songID=song.songID,
            audioStart=0.0,
            audioDuration=duration,
            embedding=embedding.tolist(),
            dimensions=512,
        )
        db.session.add(song_embedding)

        # Commit transaction
        db.session.commit()

        logger.info(f"Successfully added: {song.title}")
        return {
            'success': True,
            'song': song.title,
            'message': 'Successfully added from YouTube',
            'songID': song.songID,
            'duration': duration
        }

    except Exception as e:
        db.session.rollback()
        return {
            'success': False,
            'song': info.get('title', 'Unknown') if 'info' in locals() else 'Unknown',
            'message': f'Error: {str(e)}'
        }
    finally:
        # Cleanup temp file
        if temp_audio and os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
            except OSError:
                pass

def load_youtube_playlist(playlist_url: str, max_videos: int = 50) -> List[Dict]:
    """Process YouTube playlist."""
    results = []

    try:
        # Extract playlist info
        with yt.YoutubeDL(YDL_PLAYLIST_OPTS) as ydl:
            playlist_info = ydl.extract_info(playlist_url, download=False)

        if not playlist_info:
            return [{
                'success': False,
                'song': 'Unknown',
                'message': 'Could not extract playlist'
            }]

        entries = playlist_info.get('entries', [])
        if not entries:
            return [{
                'success': False,
                'song': 'Empty',
                'message': 'No videos found'
            }]

        # Process videos
        max_videos = min(max_videos, MAX_PLAYLIST_SIZE, len(entries))
        processed = 0

        for entry in entries:
            if processed >= max_videos:
                break

            # Skip None entries (private videos)
            if not entry:
                continue

            # Get video URL
            video_id = entry.get('id')
            if not video_id:
                continue

            video_url = f"https://www.youtube.com/watch?v={video_id}"
            processed += 1

            logger.info(f"Processing {processed}/{max_videos}")

            # Process video
            try:
                result = load_youtube_track(video_url)
                results.append(result)
            except Exception as e:
                results.append({
                    'success': False,
                    'song': video_url,
                    'message': str(e)
                })

        return results

    except Exception as e:
        logger.error(f"Playlist error: {e}")
        return [{
            'success': False,
            'song': 'Playlist Error',
            'message': str(e)
        }]

# =============================================================================
# SIMILARITY SEARCH
# =============================================================================

def find_similar_songs(embedding: np.ndarray, limit: int = 5, exclude_id: Optional[int] = None) -> List[Dict]:
    """
    Find similar songs using Manhattan distance for better discrimination.
    This avoids the "all songs look similar" problem of cosine similarity.
    """
    try:
        normalized_embedding = normalize_embedding(embedding)

        # Query database
        query = db.session.query(Song, SongEmbedding).join(
            SongEmbedding, Song.songID == SongEmbedding.songID
        )

        if exclude_id:
            query = query.filter(Song.songID != exclude_id)

        songs_query = query.all()

        # Calculate similarities
        similarities = []

        for song, song_embedding in songs_query:
            if song_embedding.embedding is not None:
                try:
                    db_embedding = normalize_embedding(np.array(song_embedding.embedding))

                    # Manhattan distance (L1 norm)
                    manhattan_distance = np.sum(np.abs(normalized_embedding - db_embedding))

                    # Convert to similarity score
                    similarity = 1.0 / (1.0 + manhattan_distance)

                    similarities.append({
                        'songID': song.songID,
                        'title': song.title,
                        'artist': song.artist,
                        'source': song.source,
                        'youtube_id': song.youtube_id,
                        'similarity': float(similarity),
                        'distance': float(manhattan_distance)
                    })
                except Exception:
                    continue

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)

        return similarities[:limit]

    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        return []

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def cleanup_temp_files() -> int:
    """Clean up old temporary files (older than 1 hour)."""
    try:
        if not os.path.exists(BACKEND_TEMP_DIR):
            return 0

        files_cleaned = 0
        current_time = time.time()

        for filename in os.listdir(BACKEND_TEMP_DIR):
            file_path = os.path.join(BACKEND_TEMP_DIR, filename)

            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > 3600:  # 1 hour
                    try:
                        os.remove(file_path)
                        files_cleaned += 1
                    except OSError:
                        pass

        if files_cleaned > 0:
            logger.info(f"Cleaned up {files_cleaned} old temporary files")

        return files_cleaned

    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return 0
