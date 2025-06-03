#!/usr/bin/env python3
"""
Simple test for similarity search using different approaches
"""

import sys
import os
import numpy as np
from flask import Flask

# Add the current directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from models import Song, SongEmbedding

def test_similarity_methods():
    """Test different approaches to similarity search."""
    app = create_app()

    with app.app_context():
        print("🔍 Testing Similarity Search Methods...")

        # Get a sample embedding
        sample_embedding = db.session.query(SongEmbedding).first()
        if not sample_embedding:
            print(" No embeddings in database")
            return

        print(f"Using embedding from song ID {sample_embedding.songID}")
        embedding_vector = np.array(sample_embedding.embedding)

        # Method 1: Direct PostgreSQL with psycopg2 style
        print("\n1️ Testing direct SQL approach...")
        try:
            sql_direct = """
            SELECT s.songID, s.title, s.artist, s.source, s.youtube_id,
                   (se.embedding <-> %s::vector) as distance
            FROM songs s
            JOIN song_embeddings se ON s.songID = se.songID
            ORDER BY se.embedding <-> %s::vector
            LIMIT %s
            """

            embedding_list = embedding_vector.tolist()
            result = db.session.execute(sql_direct, (embedding_list, embedding_list, 3))

            songs = []
            for row in result:
                songs.append({
                    'songID': row[0],
                    'title': row[1],
                    'artist': row[2],
                    'distance': float(row[5])
                })

            print(f" Direct SQL: Found {len(songs)} songs")
            for i, song in enumerate(songs, 1):
                print(f"   {i}. {song['title']} by {song['artist']} (distance: {song['distance']:.4f})")

        except Exception as e:
            print(f" Direct SQL failed: {e}")

        # Method 2: Using SQLAlchemy text()
        print("\n2️ Testing SQLAlchemy text() approach...")
        try:
            from sqlalchemy import text

            sql_text = text("""
            SELECT s.songID, s.title, s.artist, s.source, s.youtube_id,
                   (se.embedding <-> :vec::vector) as distance
            FROM songs s
            JOIN song_embeddings se ON s.songID = se.songID
            ORDER BY se.embedding <-> :vec::vector
            LIMIT :lim
            """)

            result = db.session.execute(sql_text, {
                'vec': str(embedding_list),
                'lim': 3
            })

            songs = []
            for row in result:
                songs.append({
                    'songID': row[0],
                    'title': row[1],
                    'artist': row[2],
                    'distance': float(row[5])
                })

            print(f" SQLAlchemy text(): Found {len(songs)} songs")
            for i, song in enumerate(songs, 1):
                print(f"   {i}. {song['title']} by {song['artist']} (distance: {song['distance']:.4f})")

        except Exception as e:
            print(f" SQLAlchemy text() failed: {e}")

        # Method 3: Using raw connection
        print("\n 3️ Testing raw connection approach...")
        try:
            connection = db.engine.raw_connection()
            cursor = connection.cursor()

            sql_raw = """
            SELECT s.songID, s.title, s.artist, s.source, s.youtube_id,
                   (se.embedding <-> %s::vector) as distance
            FROM songs s
            JOIN song_embeddings se ON s.songID = se.songID
            ORDER BY se.embedding <-> %s::vector
            LIMIT %s
            """

            cursor.execute(sql_raw, (embedding_list, embedding_list, 3))
            rows = cursor.fetchall()

            print(f" Raw connection: Found {len(rows)} songs")
            for i, row in enumerate(rows, 1):
                print(f"   {i}. {row[1]} by {row[2]} (distance: {row[5]:.4f})")

            cursor.close()
            connection.close()

        except Exception as e:
            print(f" Raw connection failed: {e}")

        print("\n" + "=" * 50)
        print(" Use the working method in your audioutils.py!")

if __name__ == "__main__":
    test_similarity_methods()
