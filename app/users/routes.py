from flask import Blueprint
from flask import render_template, url_for, flash, redirect, request, jsonify
from app import db, bcrypt
import config
from app.users.forms import RegistrationForm, LoginForm, UpdateAccountForm, RequestResetForm, ResetPasswordForm, AddForm
from app.users.utils import save_picture, send_reset_email
from db_models import User
from flask_login import login_user, current_user, logout_user, login_required
import uuid
from functools import wraps
import jwt

users = Blueprint('users', __name__)

@users.route('/register', methods=['GET','POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        public_id = str(uuid.uuid4())
        #token = bcrypt.generate_password_hash('{}{}{}'.format(public_id, str(form.username.data), str(datetime.utcnow))).decode('utf-8')
        #using jwt with expiration date :
        #token = jwt.encode({'user': form.username.data, 'exp': datetime.datetime.utc() + datetime.timedelta(minutes=30), app.config=['SECRET_KEY']})
        token = jwt.encode({'public_id': public_id}, key=config.SECRET_KEY).decode('UTF-8')
        user = User(username=form.username.data,public_id=public_id,admin=True, email=form.email.data, password=hashed_password, token=token)
        db.session.add(user)
        db.session.commit()
        flash('Compte crée avec succes! Maintenant vous pouvez connecter', 'success') #.format(str(form.username.data))
        return redirect(url_for('users.login'))
    return render_template('register.html',title='Register', form=form)

@users.route('/login', methods=['GET','POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            flash("Vous etes maintenant connecté", 'success')
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            if user.admin:
                return redirect('/dashboard')
            else:
                return redirect('/account')
        else:
            flash("Echec d'authentification. Vérifiez votre adresse e-mail ou votre mot de passe!", "danger")
    return render_template('login.html',title='Login', form=form)

@users.route('/logout', methods=['GET','POST'])
def logout():
    logout_user()
    return redirect(url_for('main.home'))

@users.route('/account', methods=['GET','POST'])
@login_required
def account():
    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash('Votre compte a été mis a jours avec succes', 'success')
        return redirect(url_for('users.account'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
    image_file = url_for('static', filename='profile_pics/{}'.format(current_user.image_file))
    print(image_file)
    return render_template('account.html',title='Account', image_file=image_file, form=form)

@users.route('/reset_password', methods=['GET', 'POST'])
def reset_request():
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))
    form = RequestResetForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        send_reset_email(user)
        flash('Vérifiez votre boite e-mail pour voir les instructions a suivre pour modifier votre mot de passe', 'info')
        return redirect(url_for('users.login'))
    return render_template('reset_request.html', title='Reset Password', form=form)

@users.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_token(token):
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))
    user = User.verify_reset_token(token)
    if user is None:
        flash('That is an invalid or expired token', 'warning')
        return redirect(url_for('users.reset_request'))
    form = ResetPasswordForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user.password = hashed_password
        db.session.commit()
        flash('Votre mot de passe a été mis a jour ! Vous pouvez maintenant se connecter ', 'success') #.format(str(form.username.data))
        return redirect(url_for('users.login'))
    return render_template('reset_token.html', title='Reset Password', form=form)

def admin_required(f):
    def decorated():
        #if current_user.admin == False:
            #print(current_user.admin)
            #return redirect(url_for('main.home'))
        #print(current_user.user.admin)
        #return f()
        return f() if current_user.admin else redirect(url_for('main.home'))
    return decorated

@users.route('/dashboard', methods=['GET']) #,'POST'
@login_required
@admin_required
def dashboard(): #current_user
    return render_template('dashboard.html',title='Admin Management')

@users.route('/database', methods=['GET'])
@login_required
def db_management():
    page = request.args.get('page', 1, type=int)
    users = User.query.paginate(page=page, per_page=5)
    page_total = len(users.items)
    addForm = AddForm()
    return render_template('database.html',title='Database Management',
                           users = users, addForm= addForm,
                           per_page=5, page_total=page_total)

@users.route('/development', methods=['GET'])
@login_required
def development(): #current_user
    return render_template('development.html',title='Development section')

@users.route('/admin_account', methods=['GET','POST'])
@login_required
def admin_account():
    form = UpdateAccountForm()
    if form.validate_on_submit():
        if form.picture.data:
            picture_file = save_picture(form.picture.data)
            current_user.image_file = picture_file
        current_user.username = form.username.data
        current_user.email = form.email.data
        db.session.commit()
        flash('Votre compte a été mis à jours !', 'success')
        return redirect(url_for('users.admin_account'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.email.data = current_user.email
    image_file = url_for('static', filename='profile_pics/{}'.format(current_user.image_file))
    print(image_file)
    return render_template('admin_account.html', title='Admin Account', image_file=image_file, form=form)


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'token' in request.headers:
            token = request.headers['token']
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        try:
            data = jwt.decode(token, config.SECRET_KEY)
            token_user = User.query.filter_by(public_id=data['public_id']).first()
        except:
            return jsonify({'message': 'API Token is invalid '}), 401
        return f(token_user, *args, **kwargs)
    return decorated

@users.route('/add', methods=['POST']) #
@login_required
def add():
    form = AddForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        public_id = str(uuid.uuid4())
        token = jwt.encode({'public_id': public_id}, key=config.SECRET_KEY).decode('UTF-8')
        admin = False
        if request.form.get('admin'):
            admin = True
        user = User(username=form.username.data, public_id=public_id, admin=admin, email=form.email.data,
                    password=hashed_password, token=token)
        db.session.add(user)
        db.session.commit()
        flash('Utilisateur ajouté','success')  # .format(str(form.username.data))
        return redirect(url_for('users.db_management'))
    flash("Une erreur s'est produite lors de l'ajout de cet utilisateur", "danger")
    return redirect(url_for('users.db_management'))


@users.route('/update/<id>', methods=['POST'])
@login_required
def update(id):
    user = User.query.filter_by(id=int(id)).first()
    existed_user = User.query.filter_by(username=request.form['username']).first()
    if existed_user and existed_user!= user:
        flash("Le nom d'utilisateur saisi est déja pris. Veuillez choisir un autre.", "danger")
    else:
        user.username = request.form['username']
        user.email = request.form['email']
        db.session.commit()
        flash("Informations d'utilisateur mis à jours ", "success")
    return redirect(url_for('users.db_management'))

@users.route('/delete/<id>', methods=['POST'])
@login_required
def delete(id):
    user = User.query.filter_by(id=int(id)).first()
    db.session.delete(user)
    db.session.commit()
    flash("Utilisateur supprimé", "success")
    return redirect(url_for('users.db_management'))

@users.route('/predict', methods=['GET'])
def predict():
    return jsonify({'prediction':5})
    #token = request.args['token'] # 400 error if None
    '''data = request.get_json() #optional
    token = data['token']
    username = data['auth']['username']
    password = data['auth']['password']
    dict_data_1 = data['dict'][0]
    dict_data_2 = data['dict'][1]

    return jsonify({'prediction': 5 , 'username':username, 'password':password,
                    'token':token, 'dict1': dict_data_1, 'dict2': dict_data_2})'''

'''
@users.route('/users', methods=['GET']) #'GET',
@login_required
def users():
    users = User.query.all()
    return jsonify(users)
'''

'''
# working fine
@users.route('/predict/<string:token>') #, methods=['GET']
#@token_required
def get_prediction(token): #token_user
    return jsonify({'prediction': 5 }) #to fix

#working fine :
@users.route('/predict')
def predict():
    token = request.args.get('token') #optional
    return jsonify({'prediction': 5 , 'token': token})
'''