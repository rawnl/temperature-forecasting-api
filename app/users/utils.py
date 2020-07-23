import os
import secrets
from PIL import Image
from flask import url_for, current_app
from app import mail
from flask_mail import Message

def save_picture(form_picture):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(current_app.root_path, 'static/profile_pics', picture_fn)
    form_picture.save(picture_path)

    output_size = (125, 125)
    img = Image.open(form_picture)
    img.thumbnail(output_size)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(picture_path)

    return picture_fn

def send_reset_email(user):
    token = user.get_reset_token()
    msg = Message('Requette de changement de mot de passe ', sender='noreply@demo.com', recipients=[user.email])
    msg.body = '''Cher(e) {}.
Veuillez suivre le lien suivant pour changer votre mot de passe:
{}
Si vous n'avez pas fait cette demande, ignorez cet email et aucune modification sera apportée.
'''.format(str(user.username),url_for('users.reset_token', token=token, _external=True))
    mail.send(msg)

