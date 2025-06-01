from flask import Flask, request, jsonify, Blueprint
import os
import tempfile
from audio_utils import get_embedding_from_file, save_song
from models import Song, SongEmbedding, db
import logging

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



