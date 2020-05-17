import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import json
from argparse import ArgumentParser

#***************************************************To process IMAGE******************************************************************#
from inference import Infer


#***************************************************WEB APP******************************************************************#

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
AVAILABLE_MODELS, AVAILABLE_DEVICES = Infer().getAvailableModelsAndDevices()


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
app = Flask(__name__)

@app.route("/")
@app.route("/upload")
def upload():
	return render_template('upload.html', models = AVAILABLE_MODELS, devices = AVAILABLE_DEVICES)

@app.route('/ajax/index')
def ajax_index():
	
	count, path = Infer().infer(pathToFile, algo, use_gpu)
	
	return render_template('preview.html', count = int(count), original = 'uploads/'+str(filename),newimg = "images/"+path)

@app.route("/result",methods=['POST','GET'])
def upload_file():
	if request.method == 'POST':
		# check if the post request has the file part
		if 'photos' not in request.files:
			flash('No file part')
			return render_template(upload.html,error = "Please upload a file", models = AVAILABLE_MODELS, devices = AVAILABLE_DEVICES)
		file = request.files['photos']
		# if user does not select file, browser also
		# submit an empty part without filename
		if file.filename == '':
			flash('No selected file')
			return render_template(upload.html,error = "Error File not found, please try again.", models = AVAILABLE_MODELS, devices = AVAILABLE_DEVICES)
		if file and allowed_file(file.filename):
			global filename
			global pathToFile
			global algo
			global use_gpu
			
			filename = secure_filename(file.filename)
			pathToFile = os.getcwd() + '/static/uploads/'+str(filename)
			use_gpu = request.form['device'] == 'gpu'
			file.save(pathToFile)

			algo = request.form['algo']			
			return render_template('index.html')
		else:
			return render_template(upload.html, models = AVAILABLE_MODELS, error = "Error while storing file, Please try again", devices = AVAILABLE_DEVICES)

if __name__ == "__main__":
	app.secret_key = os.urandom(24)
	# parser = ArgumentParser()
	# parser.add_argument("--use_gpu", help="use gpu or not", nargs="?", default=False, const=True, type = bool)
	# args = parser.parse_args()
	# CrowdCount = Infer(args.use_gpu)
	app.run(port=8000, debug =True)
