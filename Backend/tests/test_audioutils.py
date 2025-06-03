import os
import sys
import tempfile
import numpy as np
from flask import Flask
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from app import create_app, db
from models import Song, SongEmbedding
from audio_utils import (
    load_youtube_track,
    load_youtube_playlist,
    find_similar_songs,
    is_valid_audio_content,
    clean_title,
    get_embedding_from_file
)

class AudioUtilsTestSuite:
    def __init__(self):
        self.app = create_app()
        self.app_context = self.app.app_context()
        self.app_context.push()

        # Create tables if they don't exist
        db.create_all()

        print(" Audio Utils Test Suite")
        print("=" * 50)

    def cleanup(self):
        """Clean up test context"""
        self.app_context.pop()

    def test_title_cleaning(self):
        """Test the clean_title function"""
        print("\n Testing Title Cleaning...")

        test_cases = [
            ("Song Name (Official Video)", "Song Name"),
            ("Artist - Track [Official Audio]", "Artist - Track"),
            ("Music Video HD 4K (Official Music Video)", "Music Video"),
            ("Simple Title", "Simple Title"),
            ("Song | Official Video", "Song"),
        ]

        for original, expected in test_cases:
            result = clean_title(original)
            status = "" if result == expected else ""
            print(f"{status} '{original}' → '{result}' (expected: '{expected}')")

    def test_content_validation(self):
        """Test the is_valid_audio_content function"""
        print("\n Testing Content Validation...")

        test_cases = [
            ({'is_live': True, 'duration': 300}, False, "Live stream"),
            ({'is_live': False, 'duration': 5}, False, "Too short"),
            ({'is_live': False, 'duration': 800}, False, "Too long"),
            ({'is_live': False, 'duration': 180}, True, "Valid duration"),
            ({'duration': None}, True, "No duration info"),
        ]

        for info, expected, description in test_cases:
            result = is_valid_audio_content(info)
            status = "" if result == expected else ""
            print(f"{status} {description}: {result} (expected: {expected})")

    def test_single_youtube_video(self):
        """Test loading a single YouTube video"""
        print("\n Testing Single YouTube Video...")

        # Use a short, popular music video that should be available
        test_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Roll (short, always available)
            "https://youtu.be/dQw4w9WgXcQ",  # Short URL format
        ]

        for url in test_urls:
            print(f"\nTesting URL: {url}")
            try:
                result = load_youtube_track(url)

                if result['success']:
                    print(f" Success: {result['message']}")
                    print(f"   Song: {result['song']}")
                    if 'songID' in result:
                        print(f"   Song ID: {result['songID']}")
                else:
                    print(f"  Failed: {result['message']}")
                    print(f"   Song: {result['song']}")

            except Exception as e:
                print(f" Exception: {str(e)}")

            # Test duplicate detection
            print("\nTesting duplicate detection...")
            try:
                result2 = load_youtube_track(url)
                if not result2['success'] and 'already in database' in result2['message']:
                    print(" Duplicate detection working")
                else:
                    print(f"  Unexpected result: {result2['message']}")
            except Exception as e:
                print(f" Duplicate test failed: {str(e)}")

    def test_playlist_processing(self):
        """Test loading a YouTube playlist (with limit)"""
        print("\n Testing YouTube Playlist...")

        # Use a small public playlist
        test_playlist = "https://www.youtube.com/playlist?list=PLOHoVaTp8R7d8e5zz3vKql-5rVGKLKW1q"  # Example small playlist

        print(f"Testing playlist: {test_playlist}")
        try:
            # Limit to 3 videos for testing
            results = load_youtube_playlist(test_playlist, max_videos=3)

            print(f"Processed {len(results)} videos:")
            successful = 0
            for i, result in enumerate(results, 1):
                status = "" if result['success'] else ""
                print(f"{status} Video {i}: {result['message']} - {result['song']}")
                if result['success']:
                    successful += 1

            print(f"\nSummary: {successful}/{len(results)} successful")

        except Exception as e:
            print(f" Playlist test failed: {str(e)}")

    def test_similarity_search(self):
        """Test finding similar songs"""
        print("\n Testing Similarity Search...")

        # First, check if we have any songs in the database
        song_count = db.session.query(Song).count()
        embedding_count = db.session.query(SongEmbedding).count()

        print(f"Database status: {song_count} songs, {embedding_count} embeddings")

        if embedding_count == 0:
            print("  No embeddings in database - run some YouTube tests first")
            return

        # Get a random embedding from the database
        sample_embedding = db.session.query(SongEmbedding).first()
        if sample_embedding:
            print(f"Using embedding from song ID {sample_embedding.songID}")

            # Convert back to numpy array
            embedding_vector = np.array(sample_embedding.embedding)

            # Test similarity search
            similar_songs = find_similar_songs(embedding_vector, limit=3)

            print(f"Found {len(similar_songs)} similar songs:")
            for i, song in enumerate(similar_songs, 1):
                print(f"  {i}. {song['title']} by {song['artist']} (distance: {song['distance']:.4f})")

        # Test with random vector
        print("\nTesting with random vector...")
        random_vector = np.random.rand(512).astype(np.float32)
        similar_songs = find_similar_songs(random_vector, limit=3)
        print(f"Random vector found {len(similar_songs)} songs (this should work even with random data)")

    def test_database_operations(self):
        """Test basic database operations"""
        print("\n Testing Database Operations...")

        # Count existing records
        song_count = db.session.query(Song).count()
        embedding_count = db.session.query(SongEmbedding).count()

        print(f"Current database state:")
        print(f"  Songs: {song_count}")
        print(f"  Embeddings: {embedding_count}")

        # Test querying by source
        youtube_songs = db.session.query(Song).filter_by(source='youtube').all()
        upload_songs = db.session.query(Song).filter_by(source='upload').all()

        print(f"  YouTube songs: {len(youtube_songs)}")
        print(f"  Upload songs: {len(upload_songs)}")

        # Show some sample songs
        if youtube_songs:
            print("\nSample YouTube songs:")
            for song in youtube_songs[:3]:
                print(f"  - {song.title} by {song.artist} (ID: {song.youtube_id})")

    def test_audio_file_processing(self):
        """Test processing a local audio file (if available)"""
        print("\n Testing Audio File Processing...")

        # Look for test audio files in common locations
        test_files = [
            "test.mp3",
            "test.wav",
            "sample.m4a",
            os.path.expanduser("~/Music/test.mp3"),
        ]

        test_file = None
        for file_path in test_files:
            if os.path.exists(file_path):
                test_file = file_path
                break

        if not test_file:
            print("  No test audio file found. Place a test file as 'test.mp3' to test this.")
            print("   Tested locations:", test_files)
            return

        print(f"Testing with: {test_file}")

        try:
            embedding, duration = get_embedding_from_file(test_file)
            print(f" Generated embedding: shape {embedding.shape}, duration {duration:.2f}s")

            # Test that embedding is valid
            if len(embedding) == 512:
                print(" Embedding has correct dimensions (512)")
            else:
                print(f" Wrong embedding dimensions: {len(embedding)}")

            if np.all(np.isfinite(embedding)):
                print(" Embedding contains valid numbers")
            else:
                print(" Embedding contains invalid values")

        except Exception as e:
            print(f" Audio processing failed: {str(e)}")

    def run_all_tests(self):
        """Run all test suites"""
        try:
            self.test_title_cleaning()
            self.test_content_validation()
            self.test_database_operations()
            self.test_audio_file_processing()
            self.test_single_youtube_video()
            # Uncomment when you want to test playlists (downloads multiple videos)
            # self.test_playlist_processing()
            self.test_similarity_search()

            print("\n" + "=" * 50)
            print(" Test suite completed!")
            print("\nNext steps:")
            print("1. Check that YouTube videos were processed correctly")
            print("2. Test the Flask routes with curl or Postman")
            print("3. Try the similarity search with different vectors")

        except KeyboardInterrupt:
            print("\n  Tests interrupted by user")
        except Exception as e:
            print(f"\n Test suite failed: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

def interactive_mode():
    """Interactive testing mode"""
    test_suite = AudioUtilsTestSuite()

    try:
        while True:
            print("\n" + "=" * 50)
            print(" Interactive Test Mode")
            print("1. Test single YouTube video")
            print("2. Test YouTube playlist (3 videos)")
            print("3. Test similarity search")
            print("4. Test title cleaning")
            print("5. Show database status")
            print("6. Run all tests")
            print("0. Exit")

            choice = input("\nEnter your choice (0-6): ").strip()

            if choice == '0':
                break
            elif choice == '1':
                url = input("Enter YouTube URL: ").strip()
                if url:
                    print(f"\nTesting: {url}")
                    result = load_youtube_track(url)
                    print(f"Result: {result}")
            elif choice == '2':
                url = input("Enter YouTube playlist URL: ").strip()
                if url:
                    print(f"\nTesting playlist: {url}")
                    results = load_youtube_playlist(url, max_videos=3)
                    for result in results:
                        status = "" if result['success'] else ""
                        print(f"{status} {result['message']}: {result['song']}")
            elif choice == '3':
                test_suite.test_similarity_search()
            elif choice == '4':
                test_suite.test_title_cleaning()
            elif choice == '5':
                test_suite.test_database_operations()
            elif choice == '6':
                test_suite.run_all_tests()
                break
            else:
                print("Invalid choice. Please try again.")

    except KeyboardInterrupt:
        print("\n Goodbye!")
    finally:
        test_suite.cleanup()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        # Run all tests
        test_suite = AudioUtilsTestSuite()
        test_suite.run_all_tests()
