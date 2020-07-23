from app import create_app, db

app = create_app()

#app.app_context().push()
#with app.app_context():
#db.create_all()

# run server
if __name__ == '__main__':
    app.run(debug=True)

