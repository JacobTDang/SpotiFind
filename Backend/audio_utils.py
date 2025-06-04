import os
import time
import logging
from typing import Dict, Optional, List, Tuple
from dotenv import load_dotenv

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
    logging.warning("librosa not available - some features disabled")

# YouTube downloading
import yt_dlp as yt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

EMBEDDING_DURATION = 30  # Duration in seconds for embeddings
SAMPLE_RATE = 22050      # Standard sample rate for audio processing
MAX_PLAYLIST_SIZE = 100  # Maximum playlist size to prevent abuse

# Create backend temp directory
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
logger.info(f"FFmpeg available: {FFMPEG_AVAILABLE}, librosa: {LIBROSA_AVAILABLE}")

# =============================================================================
# YT-DLP CONFIGURATION
# =============================================================================

# Metadata extraction for single videos
YDL_META_OPTS = {
    'quiet': True,
    'skip_download': True,
    'extract_flat': False,
    'no_warnings': True,
}

# Playlist extraction - handles private/hidden videos gracefully
YDL_PLAYLIST_OPTS = {
    'quiet': True,
    'skip_download': True,
    'extract_flat': True,     # Only get video IDs, not full metadata
    'ignoreerrors': True,     # Skip unavailable/private videos
    'no_warnings': True,
    'playlistend': MAX_PLAYLIST_SIZE,
}

# Audio download configuration
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
    """Normalize embedding to unit length for proper cosine similarity."""
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
        # Resample to standard rate if needed
        if sr != SAMPLE_RATE and LIBROSA_AVAILABLE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        # Normalize audio to [-1, 1] range
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # Remove silence and apply pre-emphasis if librosa available
        if LIBROSA_AVAILABLE:
            audio, _ = librosa.effects.trim(audio, top_db=20)
            audio = librosa.effects.preemphasis(audio)

        return audio, sr
    except Exception as e:
        logger.warning(f"Audio preprocessing failed: {e}")
        return audio, sr

def convert_audio_with_ffmpeg(input_path: str) -> Optional[str]:
    """Convert audio file to WAV using FFmpeg."""
    if not FFMPEG_AVAILABLE:
        return None

    try:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(BACKEND_TEMP_DIR, f"{base_name}_converted.wav")

        cmd = [
            'ffmpeg', '-i', input_path,
            '-acodec', 'pcm_s16le',
            '-ar', str(SAMPLE_RATE),
            '-ac', '1',
            '-y',
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

def get_embedding_from_file(path: str, max_duration: int = EMBEDDING_DURATION) -> Tuple[np.ndarray, float]:
    """Generate high-quality OpenL3 embeddings from audio file."""
    working_path = path
    converted_file = None

    try:
        # Convert to WAV if needed
        if not path.lower().endswith('.wav') and FFMPEG_AVAILABLE:
            converted_file = convert_audio_with_ffmpeg(path)
            if converted_file and os.path.exists(converted_file):
                working_path = converted_file

        # Load audio
        try:
            audio, sr = sf.read(working_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read audio file {working_path}: {e}")

        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        actual_duration = len(audio) / sr

        # Select most distinctive part for longer tracks
        if actual_duration > max_duration + 10:
            start_offset = int(0.25 * actual_duration * sr)
            audio = audio[start_offset : start_offset + int(max_duration * sr)]
        elif actual_duration > max_duration:
            start_offset = int((actual_duration - max_duration) * sr / 2)
            audio = audio[start_offset : start_offset + int(max_duration * sr)]

        # Preprocess audio
        audio, sr = preprocess_audio(audio, sr)

        # Generate OpenL3 embeddings
        try:
            embeddings, _ = openl3.get_audio_embedding(
                audio, sr,
                content_type='music',
                embedding_size=512,
                hop_size=0.1
            )
        except Exception as e:
            raise RuntimeError(f"OpenL3 embedding generation failed: {e}")

        if len(embeddings) == 0:
            raise RuntimeError("No embedding frames generated by OpenL3")

        # Sophisticated temporal aggregation
        n_frames = len(embeddings)

        # Weighted temporal emphasis
        weights = np.ones(n_frames)
        emphasis_size = max(1, int(0.2 * n_frames))
        weights[:emphasis_size] *= 1.5
        weights[-emphasis_size:] *= 1.5
        weights = weights / weights.sum()

        # Multiple aggregation strategies
        weighted_mean = np.average(embeddings, axis=0, weights=weights)
        max_pool = np.max(embeddings, axis=0)
        std_pool = np.std(embeddings, axis=0)
        percentile_75 = np.percentile(embeddings, 75, axis=0)
        percentile_25 = np.percentile(embeddings, 25, axis=0)

        # Combine aggregations
        embedding = (
            0.4 * weighted_mean +
            0.2 * max_pool +
            0.2 * percentile_75 +
            0.1 * (percentile_75 - percentile_25) +
            0.1 * std_pool
        )

        # Normalize to unit length
        embedding = normalize_embedding(embedding)
        return embedding, actual_duration

    except Exception as e:
        logger.error(f"Error generating embedding from {path}: {e}")
        raise
    finally:
        # Cleanup converted file
        if converted_file and converted_file != path and os.path.exists(converted_file):
            try:
                os.remove(converted_file)
            except OSError as e:
                logger.warning(f"Failed to cleanup converted file: {e}")

# =============================================================================
# DATABASE UTILITIES
# =============================================================================

def save_song(song: Song) -> None:
    """Add song to database session and flush to get songID."""
    try:
        db.session.add(song)
        db.session.flush()
        logger.debug(f"Song saved with ID: {song.songID}")
    except Exception as e:
        logger.error(f"Failed to save song to database: {e}")
        raise

# =============================================================================
# YOUTUBE PROCESSING
# =============================================================================

def is_valid_audio_content(info: dict) -> bool:
    """Check if YouTube video is suitable for audio analysis."""
    # Skip live streams
    if info.get('is_live', False) or info.get('live_status') == 'is_live':
        return False

    # Check duration
    duration = info.get('duration')
    if duration is None or duration < 10 or duration > 600:
        return False

    # Skip non-music categories
    category = info.get('category', '').lower()
    if category in ['news & politics', 'comedy', 'education', 'science & technology']:
        return False

    return True

def clean_title(title: str) -> str:
    """Clean up video title for better song identification."""
    if not title:
        return "Unknown Title"

    cleaners = [
        '(Official Video)', '(Official Music Video)', '(Official Audio)',
        '(Lyric Video)', '(Lyrics)', '[Official Video]', '[Official Audio]',
        '(HD)', '(4K)', '| Official Video', 'Official Video',
        '(Music Video)', '[HD]', '[4K]'
    ]

    cleaned = title
    for cleaner in cleaners:
        cleaned = cleaned.replace(cleaner, '')

    return cleaned.strip()

def download_youtube_audio(info: dict) -> Optional[str]:
    """Download audio from YouTube video and return temp file path."""
    try:
        video_id = info.get('id')
        if not video_id:
            return None

        with yt.YoutubeDL(YDL_AUDIO_OPTS) as ydl:
            ydl.extract_info(info['webpage_url'], download=True)

        # Look for converted WAV file first
        expected_wav = os.path.join(BACKEND_TEMP_DIR, f"{video_id}.wav")
        if os.path.exists(expected_wav):
            return expected_wav

        # Find any file with the video ID
        for filename in os.listdir(BACKEND_TEMP_DIR):
            if filename.startswith(video_id):
                return os.path.join(BACKEND_TEMP_DIR, filename)

        return None

    except Exception as e:
        logger.error(f"Error downloading audio: {e}")
        return None

def load_youtube_track(video_url: str) -> Dict:
    """Process a single YouTube video: fetch metadata, download, embed, and save."""
    temp_audio: Optional[str] = None
    info = {}

    try:
        # Extract metadata
        with yt.YoutubeDL(YDL_META_OPTS) as ydl:
            info = ydl.extract_info(video_url, download=False)

        youtube_id = info.get('id')
        title = info.get('title', 'Unknown Title')

        if not youtube_id:
            return {'success': False, 'song': title, 'message': 'Could not extract video ID'}

        logger.info(f"Processing: {title} ({youtube_id})")

        # Check if already exists
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
                'message': 'Invalid content (live stream, wrong duration, or non-music)'
            }

        # Create Song record
        yt_song = Song(
            title=clean_title(title),
            artist=info.get('uploader', 'Unknown Artist')[:255],
            source='youtube',
            youtube_id=youtube_id,
            preview_url=video_url,
        )

        save_song(yt_song)

        # Download audio
        temp_audio = download_youtube_audio(info)
        if not temp_audio:
            db.session.rollback()
            return {'success': False, 'song': yt_song.title, 'message': 'Audio download failed'}

        # Generate embedding
        try:
            embedding, actual_duration = get_embedding_from_file(temp_audio)
        except Exception as e:
            db.session.rollback()
            return {
                'success': False,
                'song': yt_song.title,
                'message': f'Embedding generation failed: {str(e)}'
            }

        # Create embedding record
        song_embedding = SongEmbedding(
            songID=yt_song.songID,
            audioStart=0.0,
            audioDuration=min(actual_duration, EMBEDDING_DURATION),
            embedding=embedding.tolist(),
            dimensions=len(embedding),
        )

        db.session.add(song_embedding)

        # Commit both song and embedding
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            return {
                'success': False,
                'song': yt_song.title,
                'message': f'Database save failed: {str(e)}'
            }

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
        return {
            'success': False,
            'song': info.get('title', 'Unknown') if info else 'Unknown',
            'message': f'Download error: {str(e)}'
        }
    except Exception as e:
        db.session.rollback()
        return {
            'success': False,
            'song': info.get('title', 'Unknown') if info else 'Unknown',
            'message': f'Processing error: {str(e)}'
        }
    finally:
        # Cleanup temporary file
        if temp_audio and os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
            except OSError as e:
                logger.warning(f"Failed to cleanup temp file: {e}")

def load_youtube_playlist(playlist_url: str, max_videos: int = 50) -> List[Dict]:
    """
    Process YouTube playlist with robust error handling for private/hidden videos.

    Uses yt-dlp's extract_flat=True and ignoreerrors=True to gracefully skip
    private, hidden, or deleted videos without failing.
    """
    results: List[Dict] = []

    try:
        # Extract playlist metadata with flat extraction
        with yt.YoutubeDL(YDL_PLAYLIST_OPTS) as ydl:
            playlist_info = ydl.extract_info(playlist_url, download=False)

        if not playlist_info:
            return [{
                'success': False,
                'song': 'Unknown Playlist',
                'message': 'Could not extract playlist information'
            }]

        entries = playlist_info.get('entries', [])
        total_entries = len(entries)

        if total_entries == 0:
            return [{
                'success': False,
                'song': 'Empty Playlist',
                'message': 'No accessible videos found in playlist'
            }]

        logger.info(f"Found {total_entries} accessible entries in playlist")

        # Limit processing
        max_videos = min(max_videos, MAX_PLAYLIST_SIZE, total_entries)
        processed_count = 0
        skipped_count = 0

        for i, entry in enumerate(entries):
            if processed_count >= max_videos:
                break

            # Skip None entries (private/deleted videos)
            if not entry:
                skipped_count += 1
                continue

            # Extract video ID or URL
            video_id = entry.get('id')
            video_url = entry.get('url')

            if not video_id and not video_url:
                skipped_count += 1
                continue

            # Construct video URL if we only have ID
            if video_id and not video_url:
                video_url = f"https://www.youtube.com/watch?v={video_id}"

            processed_count += 1
            logger.info(f"Processing playlist item {processed_count}/{max_videos}")

            # Process individual video
            try:
                result = load_youtube_track(video_url)
                results.append(result)

                status = "✓" if result.get('success') else "✗"
                logger.info(f"{status} {result.get('song', 'Unknown')}")

            except Exception as e:
                logger.error(f"Unexpected error processing video: {e}")
                results.append({
                    'success': False,
                    'song': video_url,
                    'message': f'Unexpected error: {str(e)}'
                })

        # Summary logging
        successful = sum(1 for r in results if r.get('success', False))
        logger.info(f"Playlist complete: {successful}/{len(results)} successful, {skipped_count} skipped")

        return results

    except Exception as e:
        logger.error(f"Playlist processing error: {e}")
        return [{
            'success': False,
            'song': 'Playlist Error',
            'message': f'Playlist processing error: {str(e)}'
        }]

# =============================================================================
# SIMILARITY SEARCH
# =============================================================================

def find_similar_songs(embedding: np.ndarray, limit: int = 5, exclude_id: Optional[int] = None) -> List[Dict]:
    """Find songs similar to given embedding using pgvector cosine similarity."""
    try:
        # Normalize the query embedding
        normalized_embedding = normalize_embedding(embedding)
        embedding_list = normalized_embedding.tolist()

        # Get raw database connection for pgvector queries
        connection = db.engine.raw_connection()
        cursor = connection.cursor()

        try:
            if exclude_id:
                sql = """
                SELECT s."songID", s.title, s.artist, s.source, s.youtube_id,
                       1 - (se.embedding <=> %s::vector) as similarity,
                       (se.embedding <=> %s::vector) as distance
                FROM songs s
                JOIN song_embeddings se ON s."songID" = se."songID"
                WHERE s."songID" != %s
                ORDER BY se.embedding <=> %s::vector
                LIMIT %s
                """
                cursor.execute(sql, (embedding_list, embedding_list, exclude_id, embedding_list, limit))
            else:
                sql = """
                SELECT s."songID", s.title, s.artist, s.source, s.youtube_id,
                       1 - (se.embedding <=> %s::vector) as similarity,
                       (se.embedding <=> %s::vector) as distance
                FROM songs s
                JOIN song_embeddings se ON s."songID" = se."songID"
                ORDER BY se.embedding <=> %s::vector
                LIMIT %s
                """
                cursor.execute(sql, (embedding_list, embedding_list, embedding_list, limit))

            rows = cursor.fetchall()

            similar_songs = []
            for row in rows:
                similar_songs.append({
                    'songID': row[0],
                    'title': row[1],
                    'artist': row[2],
                    'source': row[3],
                    'youtube_id': row[4],
                    'similarity': float(row[5]),
                    'distance': float(row[6])
                })

            logger.info(f"Found {len(similar_songs)} similar songs")
            return similar_songs

        finally:
            cursor.close()
            connection.close()

    except Exception as e:
        logger.error(f"Error finding similar songs: {e}")
        return []

# =============================================================================
# UTILITIES
# =============================================================================

def cleanup_temp_files() -> int:
    """Clean up old temporary files."""
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
        logger.error(f"Error during temp file cleanup: {e}")
        return 0

def get_database_stats() -> Dict:
    """Get basic database statistics."""
    try:
        total_songs = db.session.query(Song).count()
        total_embeddings = db.session.query(SongEmbedding).count()

        # Source distribution
        source_counts = db.session.query(Song.source, db.func.count(Song.songID)).group_by(Song.source).all()

        return {
            'total_songs': total_songs,
            'total_embeddings': total_embeddings,
            'songs_by_source': dict(source_counts),
            'embedding_coverage': f"{(total_embeddings/total_songs*100):.1f}%" if total_songs > 0 else "0%"
        }

    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return {'error': str(e)}
