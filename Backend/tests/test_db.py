"""
Database connection and model testing script.
Tests the Flask app factory pattern and database connectivity.
"""
from app import create_app
from database import db
from models import Song, SongEmbedding
from sqlalchemy import text
import numpy as np

def test_database_connection():
    """Test database connection and basic operations."""
    print("Testing database setup...")

    # Create app with application context
    app = create_app()

    with app.app_context():
        try:
            # Test 1: Basic database connection
            result = db.session.execute(text('SELECT 1 as test_value'))
            row = result.fetchone()
            print(f"Database connection: OK (result: {row.test_value})")

            # Test 2: Check if tables exist
            inspector = db.inspect(db.engine)
            tables = inspector.get_table_names()
            print(f"Tables in database: {tables}")

            # Test 3: Test Song model creation
            test_song = Song(
                title="Test Song",
                artist="Test Artist",
                source="test",
                youtube_id="test123"
            )
            print(f"Song model creation: {test_song}")
            print(f"  Song details: {test_song.to_dict()}")

            # Test 4: Test database write operations (optional)
            test_write_operations = input("Test write operations? (y/n): ").lower() == 'y'

            if test_write_operations:
                # Add and commit a test song
                db.session.add(test_song)
                db.session.flush()  # Get the songID without committing

                print(f"Song saved with ID: {test_song.songID}")

                # Test embedding creation
                test_embedding = SongEmbedding(
                    songID=test_song.songID,
                    audioStart=0.0,
                    audioDuration=30.0,
                    embedding=np.random.rand(512).tolist(),  # Random 512-dim vector
                    dimensions=512
                )

                db.session.add(test_embedding)
                db.session.commit()

                print(f"Embedding saved with ID: {test_embedding.embeddingID}")

                # Test query
                saved_song = db.session.query(Song).filter_by(songID=test_song.songID).first()
                print(f"Query test: Found song '{saved_song.title}' by {saved_song.artist}")

                # Test relationship
                print(f"Relationship test: Song has {len(saved_song.embeddings)} embedding(s)")

                # Clean up test data
                db.session.delete(test_embedding)
                db.session.delete(test_song)
                db.session.commit()
                print("Test data cleaned up")

            print("\nAll tests passed! Database setup is working correctly.")

        except Exception as e:
            print(f"Database test failed: {e}")
            import traceback
            print(f"Full traceback:\n{traceback.format_exc()}")

            # Rollback any pending transactions
            db.session.rollback()

def test_embedding_similarity_pipeline():
    """Test the complete embedding similarity pipeline."""
    print("\nTesting embedding similarity pipeline...")

    app = create_app()

    with app.app_context():
        try:
            # Create two test songs with different embeddings
            song1 = Song(
                title="Test Song 1",
                artist="Test Artist 1",
                source="test",
                youtube_id="test_001"
            )

            song2 = Song(
                title="Test Song 2",
                artist="Test Artist 2",
                source="test",
                youtube_id="test_002"
            )

            db.session.add_all([song1, song2])
            db.session.flush()

            # Create similar embeddings (small difference)
            base_embedding = np.random.rand(512)
            similar_embedding = base_embedding + np.random.rand(512) * 0.1  # Add small noise

            emb1 = SongEmbedding(
                songID=song1.songID,
                audioStart=0.0,
                audioDuration=30.0,
                embedding=base_embedding.tolist(),
                dimensions=512
            )

            emb2 = SongEmbedding(
                songID=song2.songID,
                audioStart=0.0,
                audioDuration=30.0,
                embedding=similar_embedding.tolist(),
                dimensions=512
            )

            db.session.add_all([emb1, emb2])
            db.session.commit()

            print(f"Created test songs with IDs: {song1.songID}, {song2.songID}")

            # Test similarity search using your actual function
            from audio_utils import find_similar_songs

            similar_songs = find_similar_songs(base_embedding, limit=5)

            if similar_songs:
                print(f"Similarity search returned {len(similar_songs)} songs")
                for i, song in enumerate(similar_songs):
                    print(f"  {i+1}. '{song['title']}' by {song['artist']} (distance: {song['distance']:.6f})")
            else:
                print("Similarity search returned no results")

            # Clean up test data
            db.session.delete(emb1)
            db.session.delete(emb2)
            db.session.delete(song1)
            db.session.delete(song2)
            db.session.commit()
            print("Pipeline test data cleaned up")

        except Exception as e:
            print(f"Pipeline test failed: {e}")
            import traceback
            print(f"Full traceback:\n{traceback.format_exc()}")
            db.session.rollback()

def test_pgvector_functionality():
    """Test pgvector extension and similarity operations."""
    print("\nTesting pgvector functionality...")

    app = create_app()

    with app.app_context():
        try:
            # Test if pgvector extension is available
            result = db.session.execute(text("SELECT * FROM pg_extension WHERE extname = 'vector'"))
            extension = result.fetchone()

            if extension:
                print("pgvector extension is installed")

                # Test 1: Simple vector operations without table joins
                print("Testing basic vector operations...")
                sql_basic = text("""
                    SELECT
                        '[1,2,3]'::vector <-> '[1,2,3]'::vector as same_distance,
                        '[1,2,3]'::vector <-> '[4,5,6]'::vector as diff_distance
                """)

                result = db.session.execute(sql_basic)
                row = result.fetchone()
                print("Basic vector operations work:")
                print(f"  Same vector distance: {row[0]} (should be 0)")
                print(f"  Different vector distance: {row[1]:.6f}")

                # Test 2: Check if we have embeddings
                count_query = text("SELECT COUNT(*) FROM song_embeddings")
                result = db.session.execute(count_query)
                count = result.fetchone()[0]
                print(f"Found {count} embeddings in database")

                if count > 0:
                    # Test 3: Use your working find_similar_songs function
                    print("Testing with your actual similarity function...")

                    # Get a sample embedding from database
                    sample_query = text("SELECT embedding FROM song_embeddings LIMIT 1")
                    result = db.session.execute(sample_query)
                    sample_embedding_list = result.fetchone()[0]

                    # Convert to numpy array
                    import numpy as np
                    sample_embedding = np.array(sample_embedding_list)

                    # Use your actual function (which we know works!)
                    from audio_utils import find_similar_songs
                    similar_songs = find_similar_songs(sample_embedding, limit=3)

                    if similar_songs:
                        print("Similarity search via your function works:")
                        for i, song in enumerate(similar_songs, 1):
                            print(f"  {i}. '{song['title']}' by {song['artist']} (distance: {song['distance']:.6f})")
                    else:
                        print("Your similarity function returned no results")

                print("pgvector functionality confirmed working!")

            else:
                print("pgvector extension not found")

        except Exception as e:
            print(f"pgvector test failed: {e}")
            import traceback
            print(f"Full traceback:\n{traceback.format_exc()}")

def test_database_schema():
    """Check the actual database schema and column names."""
    print("\nTesting database schema...")

    app = create_app()

    with app.app_context():
        try:
            # Method 1: Use SQLAlchemy inspector to see actual schema
            from sqlalchemy import inspect

            inspector = inspect(db.engine)

            # Check songs table
            songs_columns = inspector.get_columns('songs')
            print("songs table columns:")
            for col in songs_columns:
                print(f"  - {col['name']} ({col['type']})")

            # Check song_embeddings table
            embeddings_columns = inspector.get_columns('song_embeddings')
            print("song_embeddings table columns:")
            for col in embeddings_columns:
                print(f"  - {col['name']} ({col['type']})")

            # Method 2: Query PostgreSQL system tables directly
            schema_query = text("""
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_name IN ('songs', 'song_embeddings')
                ORDER BY table_name, ordinal_position
            """)

            result = db.session.execute(schema_query)
            print("\nPostgreSQL system catalog view:")
            current_table = None
            for row in result:
                if row[0] != current_table:
                    current_table = row[0]
                    print(f"  {current_table}:")
                print(f"    - {row[1]} ({row[2]})")

        except Exception as e:
            print(f"Schema test failed: {e}")
            import traceback
            print(f"Full traceback:\n{traceback.format_exc()}")

def test_audio_utils_import():
    """Test if audio_utils can be imported without circular import issues."""
    print("\nTesting audio_utils import...")

    try:
        from audio_utils import (
            check_ffmpeg_available,
            get_embedding_from_file,
            find_similar_songs
        )
        print("audio_utils imported successfully")
        print(f"FFmpeg available: {check_ffmpeg_available()}")

        # Test that functions exist and are callable
        assert callable(get_embedding_from_file), "get_embedding_from_file should be callable"
        assert callable(find_similar_songs), "find_similar_songs should be callable"
        print("All audio utility functions are accessible")

    except ImportError as e:
        print(f"audio_utils import failed: {e}")
    except Exception as e:
        print(f"audio_utils test failed: {e}")

if __name__ == '__main__':
    print("SpotiFind Database Test Suite")
    print("=" * 40)

    test_database_connection()
    test_database_schema()
    test_pgvector_functionality()
    test_audio_utils_import()
    test_embedding_similarity_pipeline()

    print("\n" + "=" * 40)
    print("Test suite completed!")
