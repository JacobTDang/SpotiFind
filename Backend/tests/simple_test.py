import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Create Spotify client
credentials = SpotifyClientCredentials(
    client_id=os.getenv('SPOTIFY_CLIENT_ID'),
    client_secret=os.getenv('SPOTIFY_CLIENT_SECRET')
)
sp = spotipy.Spotify(client_credentials_manager=credentials)

print("\n=== Finding Tracks with Preview URLs in Your Region ===\n")

# Method 1: Search for popular tracks
print("1. Searching for popular tracks with previews...\n")

search_queries = [
    "year:2023",
    "year:2022", 
    "year:2021",
    "genre:pop",
    "genre:rock",
    "genre:hip-hop",
    "taylor swift",
    "ed sheeran",
    "drake",
    "post malone",
    "billie eilish",
    "the weeknd"
]

tracks_with_previews = []

for query in search_queries:
    try:
        results = sp.search(q=query, type='track', limit=10)
        
        for track in results['tracks']['items']:
            if track['preview_url'] and track['id'] not in [t['id'] for t in tracks_with_previews]:
                tracks_with_previews.append({
                    'id': track['id'],
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'url': track['external_urls']['spotify'],
                    'preview_url': track['preview_url']
                })
                print(f"✓ Found: {track['name']} by {track['artists'][0]['name']}")
                
                if len(tracks_with_previews) >= 10:
                    break
    except Exception as e:
        print(f"Error searching for '{query}': {e}")
    
    if len(tracks_with_previews) >= 10:
        break

print(f"\nFound {len(tracks_with_previews)} tracks with preview URLs")

# Save the track URLs for testing
if tracks_with_previews:
    print("\n2. Track URLs you can use for testing:\n")
    for i, track in enumerate(tracks_with_previews[:5]):
        print(f"Track {i+1}: {track['name']} by {track['artist']}")
        print(f"URL: {track['url']}")
        print(f"Preview: {track['preview_url'][:50]}...\n")

# Method 2: Try specific markets
print("\n3. Checking your market/region...\n")
try:
    # Get user's market
    me = sp.current_user()
    print(f"Note: Using client credentials (no user market info)")
except:
    print("Using default market")

# Method 3: Search in US market explicitly
print("\n4. Searching with explicit market (US)...\n")
try:
    results = sp.search(q="top hits 2024", type='track', limit=5, market='US')
    us_tracks = 0
    for track in results['tracks']['items']:
        if track['preview_url']:
            us_tracks += 1
            print(f"✓ US Market: {track['name']} - HAS preview")
        else:
            print(f"✗ US Market: {track['name']} - NO preview")
    print(f"\nUS market: {us_tracks}/5 tracks have previews")
except Exception as e:
    print(f"Error with US market search: {e}")

# Create a test file with working tracks
if tracks_with_previews:
    print("\n5. Creating test file with working tracks...")
    
    with open('working_tracks.txt', 'w') as f:
        f.write("# Tracks with preview URLs in your region\n\n")
        for track in tracks_with_previews[:5]:
            f.write(f"# {track['name']} by {track['artist']}\n")
            f.write(f"{track['url']}\n\n")
    
    print("✓ Created 'working_tracks.txt' with track URLs that have previews")

# Test loading one track
if tracks_with_previews:
    print("\n6. Testing audio utils with a working track...")
    
    from app import create_app, db
    from audio_utils import load_spotify_track
    
    app = create_app()
    
    with app.app_context():
        test_track = tracks_with_previews[0]
        print(f"\nLoading: {test_track['name']} by {test_track['artist']}")
        result = load_spotify_track(test_track['url'])
        print(f"Result: {result}")

print("\n=== Search Complete ===")

if not tracks_with_previews:
    print("\nNo tracks with preview URLs found!")
    print("This might be due to:")
    print("1. Regional restrictions on preview URLs")
    print("2. Spotify API limitations in your country")
    print("3. Account type restrictions")
    print("\nYou might need to:")
    print("- Use a VPN to access previews")
    print("- Use local audio files instead")
    print("- Check Spotify API documentation for your region")