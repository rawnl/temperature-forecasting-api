from flask import Blueprint, render_template

main = Blueprint('main', __name__)

@main.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@main.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')


@main.route('/how-to', methods=['GET'])
def how_to():
    return render_template('how-to.html')
