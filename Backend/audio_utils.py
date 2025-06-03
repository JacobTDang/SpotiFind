import os
import tempfile
import logging
from typing import Dict, Optional, List
from dotenv import load_dotenv

# Import db from the neutral database module
from models import Song, SongEmbedding,db
import librosa
import openl3
import soundfile as sf
import numpy as np
import yt_dlp as yt
import subprocess
import scipy.signal
from scipy.signal import butter, filtfilt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create backend temp directory if it doesn't exist
BACKEND_TEMP_DIR = os.path.join(os.path.dirname(__file__), 'temp_audio')
os.makedirs(BACKEND_TEMP_DIR, exist_ok=True)

EMBEDDING_DURATION = 60

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
YDL_PLAYLIST_OPTS = {
    'quiet': True,
    'extract_flat': 'in_playlist',  # Don't extract individual video info yet
    'ignoreerrors': True,  # Continue on errors
}

def load_youtube_track(video_url: str) -> Dict:
    """Fetch metadata, save Song, download + embed, commit to DB."""
    temp_audio = None

    try:
        # Extract metadata
        try:
            with yt.YoutubeDL(YDL_META_OPTS) as ydl:
                info = ydl.extract_info(video_url, download=False)
        except yt.DownloadError as e:
            error_msg = str(e)
            logger.error(f"YouTube extraction error: {error_msg}")

            # Provide specific error messages
            if 'Video unavailable' in error_msg:
                return {
                    'success': False,
                    'song': 'Unknown',
                    'message': 'Video is unavailable (may be private, deleted, or region-locked)'
                }
            elif 'Private video' in error_msg:
                return {
                    'success': False,
                    'song': 'Unknown',
                    'message': 'This video is private'
                }
            elif 'age-restricted' in error_msg:
                return {
                    'success': False,
                    'song': 'Unknown',
                    'message': 'This video is age-restricted'
                }
            else:
                return {
                    'success': False,
                    'song': 'Unknown',
                    'message': f'Could not access video: {error_msg[:100]}'
                }

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

        # Download audio with retry logic
        max_retries = 2
        for attempt in range(max_retries):
            try:
                temp_audio = download_youtube_audio(info)
                if temp_audio:
                    break
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    db.session.rollback()
                    return {
                        'success': False,
                        'song': yt_song.title,
                        'message': 'Audio download failed after retries'
                    }
                time.sleep(2)  # Wait before retry

        if not temp_audio:
            db.session.rollback()
            return {
                'success': False,
                'song': yt_song.title,
                'message': 'Audio download failed - no audio file created'
            }

        # Generate embedding
        try:
            embedding, actual_duration = get_embedding_from_file(temp_audio)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            db.session.rollback()
            return {
                'success': False,
                'song': yt_song.title,
                'message': f'Failed to generate audio embedding: {str(e)}'
            }

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

    except Exception as e:
        db.session.rollback()
        logger.error(f"Unexpected error processing YouTube track: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'song': info.get('title', 'Unknown') if 'info' in locals() else 'Unknown',
            'message': f'Processing error: {str(e)[:100]}'
        }
    finally:
        # Cleanup temporary file
        if temp_audio and os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
                logger.debug(f"Cleaned up temp file: {temp_audio}")
            except OSError as e:
                logger.warning(f"Failed to cleanup temp file {temp_audio}: {e}")

def save_song(song) -> None:
    """Add song to session and flush to get songID."""
    db.session.add(song)
    db.session.flush()

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

def get_embedding_from_file(path: str, max_duration: int = 60) -> tuple[np.ndarray, float]:
    """Load audio file, generate OpenL3 embedding, return embedding and duration."""
    try:
        # Since we should have FFmpeg, let's convert problematic formats first
        working_path = path

        # If it's not a WAV file, convert it using FFmpeg
        if not path.endswith('.wav'):
            converted_path = convert_audio_with_ffmpeg(path)
            if converted_path and os.path.exists(converted_path):
                working_path = converted_path
                logger.debug(f"Using converted file: {working_path}")

        # Load the audio file
        audio, sr = sf.read(working_path)

        # Handle stereo to mono conversion
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # preprocess to get clearer audio
        audio, sr = preprocess_audio(audio, sr)

        actual_duration = len(audio) / sr

        # verify audio duration is not longer than max duration
        if actual_duration > max_duration:
            audio = audio[:int(max_duration * sr)]

        # Generate embedding using correct OpenL3 API
        embeddings, timestamps = openl3.get_audio_embedding(
            audio, sr,
            content_type='music',
            embedding_size=512,
            hop_size=0.1
        )
        logger.debug(f'Generated {len(embeddings)} time-step embeddings')

        # use weighted averaging
        # give more weight to louder sections
        weights = np.mean(np.abs(embeddings), axis=1)
        weights = weights / np.sum(weights)

        # average all time-step embeddings into single 512-dim vector using weights
        embedding = np.average(embeddings, axis=0, weights=weights)

        # Cleanup converted file if we created one
        if working_path != path and os.path.exists(working_path):
            os.remove(working_path)

        return embedding, actual_duration

    except Exception as e:
        logger.error(f"Error generating embedding from {path}: {e}")
        raise

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
    """Process YouTube playlist with better error handling."""
    results = []

    try:
        # First get playlist entries without extracting full info
        with yt.YoutubeDL(YDL_PLAYLIST_OPTS) as ydl:
            playlist_info = ydl.extract_info(playlist_url, download=False)

        if not playlist_info:
            return [{
                'success': False,
                'message': 'Could not extract playlist information',
                'song': 'Unknown'
            }]

        entries = playlist_info.get('entries', [])
        playlist_title = playlist_info.get('title', 'Unknown Playlist')

        logger.info(f"Processing playlist: {playlist_title}")
        logger.info(f"Found {len(entries)} entries in playlist")

        # Process each video individually
        for idx, entry in enumerate(entries[:max_videos]):
            if not entry:
                continue

            # Extract video ID and title from flat extraction
            video_id = entry.get('id') or entry.get('url', '').split('v=')[-1].split('&')[0]
            video_title = entry.get('title', f'Video {idx + 1}')

            if not video_id or len(video_id) != 11:  # YouTube IDs are 11 chars
                logger.warning(f"Invalid video ID for entry {idx}: {video_id}")
                results.append({
                    'success': False,
                    'message': 'Invalid video ID',
                    'song': video_title
                })
                continue

            video_url = f"https://www.youtube.com/watch?v={video_id}"
            logger.info(f"Processing video {idx + 1}/{min(len(entries), max_videos)}: {video_title}")

            try:
                # Process individual video
                result = load_youtube_track(video_url)
                results.append(result)

                # Small delay to avoid rate limiting
                import time
                time.sleep(1)

            except Exception as e:
                error_msg = str(e)

                # Check for specific error types
                if 'Video unavailable' in error_msg:
                    message = 'Video is unavailable (private, deleted, or region-locked)'
                elif 'Private video' in error_msg:
                    message = 'Video is private'
                elif 'age-restricted' in error_msg:
                    message = 'Video is age-restricted'
                else:
                    message = f'Error: {error_msg[:100]}'

                logger.warning(f"Failed to process {video_title}: {message}")
                results.append({
                    'success': False,
                    'message': message,
                    'song': video_title
                })

        # Summary
        successful = sum(1 for r in results if r.get('success', False))
        logger.info(f"Playlist processing complete: {successful}/{len(results)} successful")

        if not results:
            results.append({
                'success': False,
                'message': 'No videos could be processed from this playlist',
                'song': 'None'
            })

        return results

    except Exception as e:
        logger.error(f"Critical error processing playlist: {e}")
        return [{
            'success': False,
            'message': f'Playlist processing failed: {str(e)}',
            'song': 'Unknown'
        }]

def find_similar_songs(embedding: np.ndarray,limit: int = 5, return_closest_only: bool = False,distance_threshold: float = 0.3) -> List[Dict]:
    """Find songs similar to given embedding using pgvector cosine similarity."""
    try:
        embedding_list = embedding.tolist()

        # Use raw connection
        connection = db.engine.raw_connection()
        cursor = connection.cursor()

        # Get more results initially to ensure we have good matches
        query_limit = 1 if return_closest_only else limit

        sql = """
        SELECT s."songID", s.title, s.artist, s.source, s.youtube_id,
              (se.embedding <-> %s::vector) as distance
        FROM songs s
        JOIN song_embeddings se ON s."songID" = se."songID"
        ORDER BY se.embedding <-> %s::vector
        LIMIT %s
        """
        cursor.execute(sql, (embedding_list, embedding_list, query_limit))
        rows = cursor.fetchall()

        similar_songs = []
        for row in rows:
            distance = float(row[5])

            # Skip if distance exceeds threshold (not similar enough)
            if distance > distance_threshold:
                logger.info(
                    f"Skipping song '{row[1]}' - distance {distance:.3f} exceeds threshold {distance_threshold}"
                )
                continue

            song_data = {
                'songID': row[0],
                'title': row[1],
                'artist': row[2],
                'source': row[3],
                'youtube_id': row[4],
                'distance': distance,
                'similarity': 1 - distance  # Convert distance to similarity score
            }
            similar_songs.append(song_data)

        cursor.close()
        connection.close()

        # If we're looking for closest only and found nothing within threshold
        if return_closest_only and not similar_songs:
            logger.warning(
                f"No songs found within distance threshold {distance_threshold}"
            )

        return similar_songs

    except Exception as e:
        logger.error(f"Error finding similar songs: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return []

def find_closest_song(embedding: np.ndarray):
  '''
  Finding one song that is close to matching the givin embedding
  '''
  results = find_similar_songs(embedding, limit=1, return_closest_only=True, distance_threshold=distance_threshold)
  return results[0] if results else None

def preprocess_audio(audio, sr, target_sr=22050):
  """
  used to preprocess audio to accuracy and quality before embedding
  """

  # resample if necessary
  try:
    if sr != target_sr:
      audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
      sr = target_sr

    # normalize amplitude
    audio = librosa.util.normalize(audio)

    # create bandpass filter using nyquist frequency.
    # nyquist_freq = half my sample rate
    nyquist_freq = sr / 2
    high_freq = 80 / nyquist_freq
    low_freq = 8000 / nyquist_freq

    # apply butterworth filter
    if high_freq < 1.0:
      b, a = butter(4,[low_freq, high_freq], btype = 'band')
      audio = filtfilt(b,a,audio)

    # apply spectral gating to reduce noise that is quieter than 20 decibels
    audio = librosa.effects.trim(audio,top_db=20)[0]

    # reduce volume differences while making the audio more consistent without harsh distortion
    # multiply by 1.5 increase amplitude, pushing louder audio into nonlinear part of tan func
    # tan will compress audio valuse smoothly and dividing by 1.5 scales the output back
    audio = np.tanh(audio * 1.5)/1.5

    return audio, sr

  except Exception as e:
      logger.warning(f"Audio preprocessing failed: {e}")
      return audio, sr  # Return original if preprocessing fails

def get_multiple_embeddings(audio, sr, segment_duration =5.0, overlap = 2.0):
  """
  generate multple embeddings from overlapping segments to help capture differet
  parts of the song
  """

  embeddings = []
  segment_samples = int(segment_duration * sr)
  overlap_samples = int(overlap * sr)

  start = 0

  while (start + segment_samples) <= len(audio):
    #
    segment = audio[start:start + segment_samples]

    # compute average amplitude
    if np.mean(np.abs(segment)) > 0.01:
      emb, _ = openl3.get_audio_embedding(
        segment, sr, content_type='music',embedding_size=512,hop_size=0.1
      )
      # if openl3 returned at least one embedding
      if len(emb) > 0:
        embeddings.append(np.mean(emb,axis=0))
    # move stard forward
    start += segment_samples - overlap_samples

  return embeddings


def find_similar_songs_multi_segment(embeddings_list: List[np.ndarray], limit: int =10) -> List[Dict]:
  """
  find similar songs using multiple different embeddings
  """
  final_results = []

  # unpacks and adds similar queried embeddings to final_results
  for embed in embeddings_list:
    result = find_similar_songs(embed, limit=limit*2)
    final_results.extend(result)

  # combin results by song ID and taking which every one has the shortest distance

  song_distance_list = {}
  song_info = {}

  for result in final_results:
    song_id = result['songID']
    distance = result['distance']

    if song_id not in song_distance_list or distance < song_distance_list['song_id']:
      song_distance_list['song_id'] = distance
      song_info[song_id] = result

  sorted_songs = sorted(song_info.values(), key=lambda x: x['distance'])
  return sorted_songs[:limit]

def adaptive_similarity_search(embedding: np.ndarray, limit: int = 5) -> List[Dict]:
  """
  adaptive search that will adjust threshold based on result quality distribution
  """

  # get results intially
  initial_results = find_similar_songs(embedding, limit=20)

  # return empty string if initial results is empty
  if not initial_results:
    return[]

  # gets array of inital distances
  distances = [r['distance'] for r in initial_results]

  mean_distance = np.mean(distances)
  std_distance = np.std(distances)

  # if the results are spread out, it most likely maens we have bad
  # quality, so be more lenient with threshold

  if std_distance > 0.3:
    threshold = mean_distance + std_distance
    logger.info(f"Using lenient threshold due to high variance: {threshold:.3f}")
  else:
    threshold = mean_distance + 0.5 * std_distance

  filtered_results = [r for r in initial_results if r['distance'] <= threshold]

   # Always return at least top 3 if available
  if len(filtered_results) < 3 and len(initial_results) >= 3:
      filtered_results = initial_results[:3]
  return filtered_results[:limit]
