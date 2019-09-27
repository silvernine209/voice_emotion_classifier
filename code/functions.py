import matplotlib.pyplot as plt
def rfft(signal):
    return abs(scipy.fftpack.rfft(abs(signal)))

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


# Fourier Transformation. Link : https://musicinformationretrieval.com/fourier_transform.html
# It transforms our time-domain signal into the frequency domain. Whereas the time domain expresses our signal as a sequence of samples, the frequency domain expresses our signal as a superposition of sinusoids of varying magnitudes, frequencies, and phase offsets.

def rfft(signal):
    return abs(scipy.fftpack.rfft(abs(signal)))

def rfft_clip(signal,min_freq,max_freq,sr):
    freq = np.fft.rfftfreq(len(signal), d=1./sr)
    return abs(np.fft.rfft((signal)))[(freq>min_freq) & (freq<max_freq)]


def fft(signal):
    return scipy.fft(signal)

# I want to normalize in hope of taking gender out of voice
def normalize(signal):
    return librosa.util.normalize(signal)

def min_max_scale(signal):
    scaler = MinMaxScaler(feature_range=(-1,1))
    return scaler.fit_transform(np.array(signal).reshape(-1,1)).reshape(-1)

def spectral_centroid(signal,sr):
    return librosa.feature.spectral_centroid(signal,sr=sr)[0]

# Get length of voice
def seconds(signal):
    return librosa.get_duration(signal)

# Returns total number of times signal crossed zero. It will return 0 if I removed signal below threshold
def zero_crossings(signal):
    return sum(librosa.zero_crossings(signal))

def freq_from_zero_crossings(signal):
    return zero_crossings(signal)/seconds(signal)

# Bucket_size while not losing information by extracting features : std, mean, max, min
def compress_data_and_extract_features(data,bucket_size):
    data_bucket_std = []
    data_bucket_mean = []
    data_bucket_percentile_0 = []
    data_bucket_percentile_1 = []
    data_bucket_percentile_25 = []
    data_bucket_percentile_50 = []
    data_bucket_percentile_75 = []
    data_bucket_percentile_99 = []
    data_bucket_percentile_100 = []
        
    for j in range(0,data.shape[0],bucket_size):
        data_bucket_std.append(abs(data[j:(j+bucket_size)]).std())
        data_bucket_mean.append(abs(data[j:(j+bucket_size)]).mean())
        holder_percentile=np.percentile(abs(data[j:(j+bucket_size)]),[0, 1,25,50,75, 99, 100])
        data_bucket_percentile_0.append(holder_percentile[0])
        data_bucket_percentile_1.append(holder_percentile[1])
        data_bucket_percentile_25.append(holder_percentile[2])
        data_bucket_percentile_50.append(holder_percentile[3])
        data_bucket_percentile_75.append(holder_percentile[4])
        data_bucket_percentile_99.append(holder_percentile[5])
        data_bucket_percentile_100.append(holder_percentile[6])

    
    return np.array(data_bucket_std+data_bucket_mean+data_bucket_percentile_0+data_bucket_percentile_1+data_bucket_percentile_25+data_bucket_percentile_50+data_bucket_percentile_75+data_bucket_percentile_99+data_bucket_percentile_100)
    #return np.array(data_bucket_std+data_bucket_mean+data_bucket_percentile_90+data_bucket_percentile_99+data_bucket_percentile_100)
def extract_max(pitches,magnitudes, shape):
    new_pitches = []
    new_magnitudes = []
    for i in range(0, shape[1]):
        new_pitches.append(np.max(pitches[:,i]))
        new_magnitudes.append(np.max(magnitudes[:,i]))
    return (new_pitches,new_magnitudes)

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
    
    
    for i in trange(len(df['filename'])):
        # Empty holder for data for efficiency memory
        stripped_holder,ffted_holder=[],[]               
        
        
        # Load file as signal & strip the silences at the beginning and at the end
        signal,sr = librosa.load(df['filename'].iloc[i],res_type='kaiser_fast') 
        signal = strip(signal)
        stripped.append(signal)
        
        # Extract From Signal
        signal_extract.append(compress_data_and_extract_features(signal,len(signal)))
        
        
        # Get pitches & Magnitudes
        pitch, magnitude = librosa.core.piptrack(y=signal, sr=22050, n_fft=n_fft, hop_length=hop_length, fmin=50, fmax=380, threshold=0.0001, win_length=None, window='hann', center=True, pad_mode='reflect', ref=None)
        pitches,magnitudes = extract_max(pitch,magnitude,pitch.shape)
        pitches = [x for x in pitches if x>0]
        pitch_list.append(compress_data_and_extract_features(np.array(pitches),len(pitches)))
        pitch_mag_list.append(compress_data_and_extract_features(np.array(magnitudes),len(magnitudes)))
        
        # 20 Mfcc features
        mfcc.append(np.mean(librosa.feature.mfcc(signal, sr=sr),axis=1))
        # Get rms
        rms = librosa.feature.rms(y=signal,frame_length=n_fft,hop_length=hop_length)[0]
        rmsed.append(compress_data_and_extract_features(np.array(rms),len(rms)))
        
        # Get top strongest RMS segments of audio file.
        top_idx = np.argsort(rms)[-30:]
        mean_freq_list = []
        # Iterate through each strong rms segments to extract pitches
        for ix,index in enumerate(top_idx):
            max_rms_frame_start = int(index/len(rms)*len(signal)-n_fft*2)
            max_rms_frame_stop = int(index/len(rms)*len(signal)+n_fft*2)
            audio_strongest_rms = signal[max_rms_frame_start:max_rms_frame_stop]
            # Auto correlate the audio in hope of removing noise and getting fundamental frequencyu
            signal_autocorr = librosa.autocorrelate(audio_strongest_rms)
            # Get pitch and magnitude of each segment
            if list(signal_autocorr):
                pitch_rms, _ = librosa.core.piptrack(y=signal_autocorr, sr=22050, n_fft=n_fft, hop_length=hop_length, fmin=50, fmax=380, threshold=0.0001, win_length=None, window='hann', center=True, pad_mode='reflect', ref=None)
                freq_list_holder = []
                for i in range(len(pitch_rms)):            
                    freq_list_holder+=list(pitch_rms[i])
                mean_freq_list.append(np.mean([x for x in freq_list_holder if x>0])) 
            mean_freq_list = [x for x in mean_freq_list if x>0]
        pitch_stat_from_rms_holder = []
        pitch_stat_from_rms_holder.append(np.mean(mean_freq_list))
        pitch_stat_from_rms_holder.append(np.std(mean_freq_list))
        pitch_stat_from_rms_holder.append(np.min(mean_freq_list))
        pitch_stat_from_rms_holder.append(np.max(mean_freq_list))
        pitch_stat_from_rms.append(np.array(pitch_stat_from_rms_holder))
          
    df['stripped']= stripped
    df['stripped_stat']=signal_extract
    df['mfcc']= mfcc
    df['rmsed']= rmsed
    df['pitch_list']= pitch_list
    df['pitch_mag_list']=pitch_mag_list
    df['pitch_stat_from_rms'] = pitch_stat_from_rms
    return df

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    SMALL_SIZE = 8
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 20

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
 
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def various_scores_string_binary(model,X,y,data_type):
    pred = model.predict(X)
    binarizer = MultiLabelBinarizer()
    binarizer.fit(y)

    print(data_type," Data")
    print("Accuracy : {:.2f} %".format(metrics.accuracy_score(binarizer.transform(y),binarizer.transform(pred))*100))
    print("Precision : {:.2f}".format(metrics.precision_score(binarizer.transform(y),binarizer.transform(pred),average='macro')))
    print("Recall : {:.2f}".format(metrics.recall_score(binarizer.transform(y),binarizer.transform(pred),average='macro')))
    print("F1 : {:.2f}".format(metrics.f1_score(binarizer.transform(y),binarizer.transform(pred),average='macro')))
    
def various_scores(model,X,y,data_type):
    pred = model.predict(X)
    print(data_type," Data")
    print('######################################################')
    print("Accuracy : {:.2f} %".format(metrics.accuracy_score(y, pred)*100))
    print('______________________________________________________')
    print(metrics.classification_report(y,pred,digits=2))
    print('______________________________________________________')   
    