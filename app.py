import os
import sys
from flask import Flask, render_template, request, redirect, flash
from flask_bootstrap import Bootstrap
from predict import predict_number

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config.from_object(__name__)

app.config['UPLOAD_FOLDER'] = 'static/input'
app.config['SECRET_KEY'] = 'secret key' #change this!

bootstrap = Bootstrap(app)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CATEGORIES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

@app.errorhandler(404)
def page_not_found(e=None):
	return render_template('404.html', title='404'), 404

@app.errorhandler(500)
def internal_server_error(e=None):
	return render_template('500.html', title='500'), 500

@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            flash('No file provided')
            return redirect(request.url)
        extension = file.filename.split('.')[1]
        name = file.filename.split('.')[0]
        if (extension in ALLOWED_EXTENSIONS):
            filename = name + "." + extension
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            prediction = predict_number(f"static/input/{filename}")
            return redirect(f'/?filename={filename}&prediction="{prediction}"')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.args.get("filename"):
        prediction = request.args.get("prediction").split("_")[0].replace('"', "")
        filename = request.args.get('filename')
        return render_template("index.html", image=f"input/{filename}", number=prediction)
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)