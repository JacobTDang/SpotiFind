from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate()

# to init database in app.py
def init_db(app):
  db.init_app(app)
  migrate.init_app(app, db)

  from models import Song, SongEmbedding

  return db
