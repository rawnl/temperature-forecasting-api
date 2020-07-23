from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin
from flask_mail import Mail
#from app.config import Config

db = SQLAlchemy()

bcrypt = Bcrypt()
login_manager = LoginManager()
login_manager.login_view = 'users.login'
login_manager.login_message_category = 'info'

mail = Mail()

app = Flask(__name__, template_folder='template')

app.config.from_pyfile('config.py')

db.init_app(app)
bcrypt.init_app(app)
login_manager.init_app(app)
mail.init_app(app)

from app.main.routes import main
from app.users.routes import users
from app.forecasts.routes import forecasts
from app.errors.handlers import errors

app.register_blueprint(main)
app.register_blueprint(users)
app.register_blueprint(forecasts)
app.register_blueprint(errors)

with app.app_context():
    db.create_all()
