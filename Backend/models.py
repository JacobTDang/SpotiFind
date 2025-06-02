from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Float, Boolean, Text
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from database import db

class Song(db.Model):
    __tablename__ = 'songs'

    songID = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    artist = Column(String(255), nullable=False)
    source = Column(String(50), default='local')
    youtube_id  = Column(String(255), nullable=True)
    preview_url = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship with cascade delete
    embeddings = relationship('SongEmbedding', back_populates='song', cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'songID': self.songID,
            'title': self.title,
            'artist': self.artist,
            'source': self.source,
            'youtube_id': self.youtube_id ,
            'preview_url': self.preview_url,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class SongEmbedding(db.Model):
    __tablename__ = 'song_embeddings'

    embeddingID = Column(Integer, primary_key=True)
    songID = Column(Integer, ForeignKey('songs.songID'), nullable=False)

    # Audio metadata
    dimensions = Column(Integer, nullable=False, default=512)
    audioStart = Column(Float, default=0.0)
    audioDuration = Column(Float, nullable=True)

    # Vector column for 512-dimensional embeddings
    embedding = Column(Vector(512), nullable=False)

    # Relationship
    song = relationship('Song', back_populates='embeddings')

    def to_dict(self):
        return {
            'embeddingID': self.embeddingID,
            'songID': self.songID,
            'dimensions': self.dimensions,
            'audioStart': self.audioStart,
            'audioDuration': self.audioDuration
        }
