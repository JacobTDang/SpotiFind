import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv
from models import Song, SongEmbedding, db
import datetime
import requests
import tempfile
import openl3
import soundfile as sf
from typing import List
import numpy as np

# 1.) parse spotify track data
# 2.) set data from song model
# 3.) create embedding based on Spotify audio data and songEmbedding model
# 4.) add to local postgress database

# load spotify api credentials
load_dotenv()
credentials = SpotifyClientCredentials(
    client_id=os.getenv('SPOTIFY_CLIENT_ID'),
    client_secret=os.getenv('SPOTIFY_CLIENT_SECRET')
)
sp = spotipy.Spotify(client_credentials_manager=credentials)

def load_spotify_track(url: str):
    # load track by url
    track = sp.track(url)
    
    # load data into the song
    spotify_song = Song(
        title = track['name'],
        artist = track['artists'][0]['name'],
        source = 'SPOTIFY',
        spotify_id = track['id'],
        preview_url = track['preview_url']
    )
    
    # Send a query to check if song is in the database
    # check if song exists in database by track_id
    existing_song = db.session.query(Song).filter_by(spotify_id=track['id']).first()
    try:
        if not existing_song:
            save_song(spotify_song)
    
            # handle preview url being null
            if spotify_song.preview_url is not None:
                song_embedding = get_embedding(spotify_song)
                
                # create song and load song
                embedding = SongEmbedding(
                audioStart = 0.0,
                audioDuration = 30.0,
                songID = spotify_song.songID,
                embedding = song_embedding,
                dimensions = 512
                )

                # load song into database
                db.session.add(embedding)
                db.session.commit()
                
                return {
                'success': True,
                'song': spotify_song.title,
                'message': 'Song added successfully'
                }
                
            else:
                # rollback the sessoin
                db.session.rollback()
                return {
                'success': False,
                'song': spotify_song.title,
                'message': 'No spotify preview url'
                }
        else:
            return {
                'success': False,
                'song': spotify_song.title,
                'message': 'Song already in database'
                }
    except Exception as e:
        db.session.rollback()
        return{
            'success': False,
            'song': spotify_song.title if spotify_song else 'Unknown',
            'message': f'Error: {str(e)}'
        }
        
        
# save song to the database    
def save_song(spotify_song: Song):
    
    # create session and add song to database
    db.session.add(spotify_song)
    
    # use flush bc what if song doesn't embedding doesn't save
    db.session.flush()

# creates the song embedding based on the 30 second preview from spotify and returns np array of the embdedding
def get_embedding(spotify_song: Song) -> np.ndarray:
    
    temp_path = None
    
    try:
        # first need to downlaod the mp3
        temp_path = download_preview(spotify_song)
        
        # generate the embedding with openL3
        # Load audio
        audio, sr = sf.read(temp_path)

        # generate embedding
        # - content_type='music' for music
        # - embedding_size=512 for 512-dimensional vectors
        # - hop_size determines time resolution
        embedding, timestamps = openl3.get_embedding(
            audio, 
            sr,
            content_type='music',
            embedding_size=512,
            # generates embedding every 0.5 seconds
            hop_size=0.5  
        )
        
        # since it is multiple embeddings, average using numpy, since pg vector also takes in np arrays
        average_embedding = np.mean(embedding, axis=0)
        
        return average_embedding
    
    # cleans up the temp file so they dont accumulate over time
    finally:
        # This runs even if there's an error
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            
# downloads preview in a temp file
def download_preview(spotify_song: Song):
    
    response = requests.get(spotify_song.preview_url)
    
    # creates a temp file to store the mp3 file
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
        tmp_file.write(response.content)
        temp_path = tmp_file.name
        
    return temp_path

# function to load playlist
def load_playlist(playlist_url: str):
    # playlist_url is the playlist url
    # limit is the max amount of tracks I get from playlist
     # This method is specifically for getting tracks
    tracks = sp.playlist_tracks(playlist_url, limit=100)
    tracks_loaded = []
    for item in tracks['items']:
        track = item['track']
        if track:
            track_url = track['external_urls']['spotify']
            loaded_track = load_spotify_track(track_url)
            tracks_loaded.append(loaded_track)
            print(f"{result['message']}: {result['song']}")
    return tracks_loaded