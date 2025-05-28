from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from routes import register_routes
import os
# iniitalize the app and db
db = SQLAlchemy()
app = Flask(__name__)

#TODO: set up POSTGRESS database for song vectors...
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
db.init_app(app)
    
# register the routes
register_routes(app, db)

# registering migrate command
migrate = Migrate(app, db)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
