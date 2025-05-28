from app import db

class song(db.Model):
    __table__ = 'songs'
    
    def __repr__(self):
        return ""