import os
#import magic
import urllib.request
from app import app
import flask
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import pickle
import os
import librosa
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns
from graph import build_graph


ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'wav', 'mp3','m4a'])
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
	if request.method == 'POST':
        # check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No file selected for uploading')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			flash('File successfully uploaded')
			return redirect('/')
		else:
			flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
			return redirect(request.url)


def df_preprocess(df):
	n_fft = 512
	hop_length = 128

	stripped = []
	signal_extract = []
	mfcc = []
	rmsed = []
	pitch_list = []
	pitch_mag_list = []
	pitch_stat_from_rms = []

	for i in range(len(df['filename'])):
		# Empty holder for data for efficiency memory
		stripped_holder, ffted_holder = [], []

		# Load file as signal & strip the silences at the beginning and at the end
		signal, sr = librosa.load(df['filename'].iloc[i])#, res_type='kaiser_fast')
		signal = strip(signal)
		stripped.append(signal)

		# Extract From Signal
		signal_extract.append(compress_data_and_extract_features(signal, len(signal)))

		# Get pitches & Magnitudes
		pitch, magnitude = librosa.core.piptrack(y=signal, sr=22050, n_fft=n_fft, hop_length=hop_length, fmin=50,
												 fmax=380)
		pitches, magnitudes = extract_max(pitch, magnitude, pitch.shape)
		pitches = [x for x in pitches if x > 0]
		pitch_list.append(compress_data_and_extract_features(np.array(pitches), len(pitches)))
		pitch_mag_list.append(compress_data_and_extract_features(np.array(magnitudes), len(magnitudes)))

		# 20 Mfcc features
		mfcc.append(np.mean(librosa.feature.mfcc(signal, sr=sr), axis=1))

		# Get rms
		rms = librosa.feature.rms(y=signal, frame_length=n_fft, hop_length=hop_length)[0]
		rmsed.append(compress_data_and_extract_features(np.array(rms), len(rms)))
	df['stripped'] = stripped
	df['stripped_stat'] = signal_extract
	df['mfcc'] = mfcc
	df['rmsed'] = rmsed
	df['pitch_list'] = pitch_list
	df['pitch_mag_list'] = pitch_mag_list
	# df['pitch_stat_from_rms'] = pitch_stat_from_rms
	return df
# Strip silence at the beginning and end of a signal
def strip(signal, frame_length=512, hop_length=256):
	# Compute RMSE.
	rmse = librosa.feature.rms(signal, frame_length=frame_length, hop_length=hop_length, center=True)

	# Identify the first frame index where RMSE exceeds a threshold.
	thresh = 0.001
	frame_index = 0
	while rmse[0][frame_index] < thresh:
		frame_index += 1

	# Convert units of frames to samples.
	start_sample_index = librosa.frames_to_samples(frame_index, hop_length=hop_length)

	signal = signal[start_sample_index:]
	signal = np.array(list(signal)[::-1])

	# Compute RMSE.
	rmse = librosa.feature.rms(signal, frame_length=frame_length, hop_length=hop_length, center=True)

	# Identify the first frame index where RMSE exceeds a threshold.
	thresh = 0.001
	frame_index = 0
	while rmse[0][frame_index] < thresh:
		frame_index += 1

	# Convert units of frames to samples.
	start_sample_index = librosa.frames_to_samples(frame_index, hop_length=hop_length)

	signal = np.array(signal[start_sample_index:])

	# Return the trimmed signal.
	return np.array(list(signal)[::-1])

# Bucket_size while not losing information by extracting features : std, mean, max, min
def compress_data_and_extract_features(data, bucket_size):
	data_bucket_std = []
	data_bucket_mean = []
	data_bucket_percentile_0 = []
	data_bucket_percentile_1 = []
	data_bucket_percentile_25 = []
	data_bucket_percentile_50 = []
	data_bucket_percentile_75 = []
	data_bucket_percentile_99 = []
	data_bucket_percentile_100 = []

	for j in range(0, data.shape[0], bucket_size):
		data_bucket_std.append(abs(data[j:(j + bucket_size)]).std())
		data_bucket_mean.append(abs(data[j:(j + bucket_size)]).mean())
		holder_percentile = np.percentile(abs(data[j:(j + bucket_size)]), [0, 1, 25, 50, 75, 99, 100])
		data_bucket_percentile_0.append(holder_percentile[0])
		data_bucket_percentile_1.append(holder_percentile[1])
		data_bucket_percentile_25.append(holder_percentile[2])
		data_bucket_percentile_50.append(holder_percentile[3])
		data_bucket_percentile_75.append(holder_percentile[4])
		data_bucket_percentile_99.append(holder_percentile[5])
		data_bucket_percentile_100.append(holder_percentile[6])

	return np.array(
		data_bucket_std + data_bucket_mean + data_bucket_percentile_0 + data_bucket_percentile_1 + data_bucket_percentile_25 + data_bucket_percentile_50 + data_bucket_percentile_75 + data_bucket_percentile_99 + data_bucket_percentile_100)

# return np.array(data_bucket_std+data_bucket_mean+data_bucket_percentile_90+data_bucket_percentile_99+data_bucket_percentile_100)
def extract_max(pitches, magnitudes, shape):
	new_pitches = []
	new_magnitudes = []
	for i in range(0, shape[1]):
		new_pitches.append(np.max(pitches[:, i]))
		new_magnitudes.append(np.max(magnitudes[:, i]))
	return (new_pitches, new_magnitudes)


def process_test_data():
	file_name = os.listdir('upload/')[0]
	full_path_to_file = os.getcwd()+'/upload/' + file_name
	test_data_folder_list = [full_path_to_file]

	# Create dataframe for test data
	df_test = pd.DataFrame({'filename': test_data_folder_list})

	# Process df_test and create ffted,mfcc,and etc...
	df_test_processed = df_preprocess(df_test)

	X_test_strip_stat = pd.DataFrame(list(df_test_processed['stripped_stat']))
	X_test_strip_stat.columns = X_strip_stat_column_names

	X_test_pitch_extract = pd.DataFrame(list(df_test_processed['pitch_list']))
	X_test_pitch_extract.columns = X_pitch_extract_column_names

	X_test_pitch_mag_extract = pd.DataFrame(list(df_test_processed['pitch_mag_list']))
	X_test_pitch_mag_extract.columns = X_pitch_mag_extract_column_names

	X_test_rms_extract = pd.DataFrame(list(df_test_processed['rmsed']))
	X_test_rms_extract.columns = X_rms_extract_column_names

	X_test_mfcc = pd.DataFrame(list(df_test_processed['mfcc']))
	X_test_mfcc.columns = X_mfcc_column_names

	df_test_multiple_features = pd.concat(
		[X_test_strip_stat, X_test_pitch_extract, X_test_pitch_mag_extract, X_test_rms_extract, X_test_mfcc], axis=1)

	# For X_real_test, choose one of the top
	test_gender = df_test_multiple_features[X_strip_stat_column_names]
	test_class = df_test_multiple_features[
		X_strip_stat_column_names + X_pitch_extract_column_names + X_rms_extract_column_names + X_mfcc_column_names[:5]]
	return test_gender, test_class, df_test_processed

X_strip_stat_column_names = ['strip_std', 'strip_mean', 'strip_0th', 'strip_1st', 'strip_25th', 'strip_50th', 'strip_75th', 'strip_99th', 'strip_100th']
X_pitch_extract_column_names = ['pitch_std', 'pitch_mean', 'pitch_0th', 'pitch_1st', 'pitch_25th', 'pitch_50th', 'pitch_75th', 'pitch_99th', 'pitch_100th']
X_pitch_mag_extract_column_names =  ['pitch_mag_std', 'pitch_mag_mean', 'pitch_mag_0th', 'pitch_mag_1st', 'pitch_mag_25th', 'pitch_mag_50th', 'pitch_mag_75th', 'pitch_mag_99th', 'pitch_mag_100th']
X_rms_extract_column_names = ['rms_std', 'rms_mean', 'rms_0th', 'rms_1st', 'rms_25th', 'rms_50th', 'rms_75th', 'rms_99th', 'rms_100th']
X_mfcc_column_names = ['mfcc_1' , 'mfcc_2' , 'mfcc_3' , 'mfcc_4' , 'mfcc_5' , 'mfcc_6' , 'mfcc_7' , 'mfcc_8' , 'mfcc_9' , 'mfcc_10' , 'mfcc_11' , 'mfcc_12' , 'mfcc_13' , 'mfcc_14' , 'mfcc_15' , 'mfcc_16' , 'mfcc_17' , 'mfcc_18' , 'mfcc_19' , 'mfcc_20']


with open("lgbm_grid_emotion.pkl", "rb") as f:
	lgbm_model_class = pickle.load(f)
with open("lgbm_grid_gender.pkl", "rb") as f:
	lgbm_model_gender = pickle.load(f)


@app.route("/classify", methods=["POST", "GET"])
def predict():
	test_gender, test_class, df_test_processed = process_test_data()
	pred_class = lgbm_model_class.predict_proba(test_class)

	return flask.render_template('classify_test.html',pred_class=pred_class)


@app.route('/graphs')
def graphs():
    test_gender, test_class, df_test_processed = process_test_data()
    df_bar = pd.DataFrame({'Emotion': lgbm_model_class.classes_,'Percentage': [np.round(x * 100, 1) for x in lgbm_model_class.predict_proba(test_class)[0]]})
    df_bar = df_bar.sort_values(by='Percentage', ascending=False)
    graph1_url = build_graph(df_bar['Emotion'].values,df_bar['Percentage'].values)
    return render_template('graphs.html',
                           graph1=graph1_url)


if __name__ == "__main__":
	app.run()
