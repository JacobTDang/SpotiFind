from app import db
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Float, Boolean, Text
from sqlalchemy.orm import relationship
import sqlalchemy as sa

class song(db.Model):
    __tablename__ = 'songs'
    songID = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    artist = Column(String(255), nullable=False)
    source = Column(String(50), , default = 'local')
    
    
class songEmbedding(db.Model):
    __tablename__ = 'songEmbedding'
    
    # Set ID's
    embeddingID = Column(Integer, primary_key=True)
    songID = Column(Integer, ForeignKey('song.songID'), nullable=False)
    
    # Spotify MetaData
    dimensions = Column(Integer, nullable=False)
    audioStart = Column(Float, default = 0.0)
    audioDuration = Column(Float)