from flask import Flask
from flask_cors import CORS
from database import db
from flask_migrate import Migrate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize extensions
migrate = Migrate()

def create_app():
    app = Flask(__name__)

    # Configuration
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
    app.config['UPLOAD_FOLDER'] = 'temp_uploads'

    # Initialize extensions with app
    db.init_app(app)
    migrate.init_app(app, db)

    # Register routes after app and db are initialized
    CORS(app,
      origins=[os.getenv('FRONTEND_DEV_SERVER_URL')],
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'])

    from routes import bp
    app.register_blueprint(bp)

    return app

# Create the application instance
app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
