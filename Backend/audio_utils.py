import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import openl3
import os
from dotenv import load_dotenv
from models, import Song, SongEmbedding, db
import datetime
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

# load track by url
def load_spotify_track(String: url):
    track = sp.track(url)
    
    # load data into the song
    spotify_song = Song(
        title = track['name']
        artist = track['artists'][0]['name']
        source = 'SPOTIFY'
        spotify_id = track['id']
        created_at = datetime.datetime.now()
    )
    
    save_song(spotify_song)
    
    
# save song to the database    
def save_song(Song: spotify_song):
    
    # create session and add song to database
    db.session.add(spotify_song)
    db.session.commit()