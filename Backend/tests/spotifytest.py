import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
# dotev used to load environment variables
from dotenv import load_dotenv
import os

# load env variables
load_dotenv()

# use spotify client creditals to store api key

credentials = SpotifyClientCredentials(
    client_id=os.getenv('SPOTIFY_CLIENT_ID'),
    client_secret=os.getenv('SPOTIFY_CLIENT_SECRET')
)

# create the spotify object
sp = spotipy.Spotify(client_credentials_manager=credentials)

# - q='artist:radiohead'  -> Search query with field filter
# - type='track'          -> What type of result we want  
# - limit=1              -> Maximum number of results to return

# The 'q' parameter uses Spotify's search syntax:
# 'artist:radiohead'     -> Only search for tracks by the artist "radiohead"
# You could also use:
# 'track:creep'          -> Search for tracks named "creep"
# 'album:ok computer'    -> Search within a specific album
# 'year:1997'           -> Search for tracks from 1997
# 'radiohead creep'     -> General search (no field specified)
results = sp.search(q='artist:radiohead', type='track', limit=1)


if results['tracks']['items']:
    track = results['tracks']['items'][0]
    print(f"Success! Found: {track['name']} by {track['artists'][0]['name']}")
else:
    print("No results found - check your credentials")