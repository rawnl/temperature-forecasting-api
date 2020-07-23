from datetime import datetime
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer
from app import db, login_manager
from flask import current_app, jsonify
from flask_login import UserMixin


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(120), unique=True, nullable=False)
    admin = db.Column(db.Boolean, default=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    image_file = db.Column(db.String(20), nullable=False, default='default.jpg')
    password = db.Column(db.String(60), unique=True, nullable=False)
    registration_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    token = db.Column(db.String(120), unique=True, nullable=False) # to fix

    def get_reset_token(self, expires_sec=1800):
        s = Serializer(current_app.config['SECRET_KEY'], expires_sec)
        return s.dumps({'user_id': self.id}).decode('utf-8')

    @staticmethod
    def verify_reset_token(token):
        s = Serializer(current_app.config['SECRET_KEY'])
        try:
            user_id = s.loads(token)['user_id']
        except:
            return None
        return User.query.get(user_id)

    def __repr__(self):
        # 'User({}, {}, {}, {})'.format(self.username, self.email, self.image_file, self.token)
        return jsonify({'id': self.id,
                        'public_id': self.public_id,
                        'admin': self.admin,
                        'username': self.username,
                        'email': self.email,
                        'registration_date': self.registration_date
                        })


class HourlyForecasting(db.Model):
    date_hour = db.Column(db.DateTime, primary_key=True, default=datetime.utcnow)
    temperature = db.Column(db.String(20), nullable=False)

    def __repr__(self):
        return 'HourlyForecasting({}, {})'.format(self.date_hour, self.temperature)


class DailyForecasting(db.Model):
    date = db.Column(db.DateTime, primary_key=True, default=datetime.utcnow)
    forecast = db.Column(db.String(300), nullable=False)  # jsonfile

    def __repr__(self):
        return 'DailyForecasting({}, {})'.format(self.date, self.forecast)

