import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import json
from argparse import ArgumentParser

#***************************************************To process IMAGE******************************************************************#
from inference import Infer


#***************************************************WEB APP******************************************************************#

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
app = Flask(__name__)

@app.route("/")
@app.route("/upload")
def upload():
	return render_template('upload.html', models = CrowdCount.getAvailableModels())

@app.route('/ajax/index')
def ajax_index():
	
	count, path = CrowdCount.infer(pathToFile, algo)
	
	return render_template('preview.html', count = int(count), original = 'uploads/'+str(filename),newimg = "images/"+path)

@app.route("/result",methods=['POST','GET'])
def upload_file():
	if request.method == 'POST':
		# check if the post request has the file part
		if 'photos' not in request.files:
			flash('No file part')
			return render_template(upload.html,error = "Please upload a file")
		file = request.files['photos']
		# if user does not select file, browser also
		# submit an empty part without filename
		if file.filename == '':
			flash('No selected file')
			return render_template(upload.html,error = "Error File not found, please try again.")
		if file and allowed_file(file.filename):
			global filename
			global pathToFile
			global algo
			
			filename = secure_filename(file.filename)
			pathToFile = os.getcwd() + '/static/uploads/'+str(filename)
			file.save(pathToFile)

			algo = request.form['algo']			
			return render_template('index.html')
		else:
			return render_template(upload.html, model = CrowdCount.getAvailableModels(), error = "Error while storing file, Please try again")

if __name__ == "__main__":
	app.secret_key = os.urandom(24)
	parser = ArgumentParser()
	parser.add_argument("--use_gpu", help="use gpu or not", nargs="?", default=False, const=True, type = bool)
	args = parser.parse_args()
	CrowdCount = Infer(args.use_gpu)
	app.run(host = "0.0.0.0", port=8000, debug =True)
