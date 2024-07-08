import tkinter as tk
from tkinter import Label, Button, messagebox
import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def GenderModel():
    mydata = pd.read_csv(r"C:\Users\LENEVO\OneDrive\Desktop\Internship\Task dump\voice.csv\voice.csv")

    mydata.loc[:,'label'][mydata['label']=="male"] = 0
    mydata.loc[:,'label'][mydata['label']=="female"] = 1
    mydata_train, mydata_test = train_test_split(mydata, random_state=0, test_size=.2)
    scaler = StandardScaler()
    scaler.fit(mydata_train.iloc[:,0:20])
    X_train = scaler.transform(mydata_train.iloc[:,0:20])
    X_test = scaler.transform(mydata_test.iloc[:,0:20])
    y_train = list(mydata_train['label'].values)
    y_test = list(mydata_test['label'].values)

    classifiers = {
        "Decision Tree": DecisionTreeClassifier(random_state=0),
        "Random Forest": RandomForestClassifier(n_estimators=5, random_state=0),
        "Gradient Boosting": GradientBoostingClassifier(random_state=0),
        "SVM": SVC(),
        "MLP": MLPClassifier(random_state=0)
    }

    
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        print(f"{name} trained.")

    return classifiers

def load_emotion_model(model_path, weights_path):
    with open(model_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights(weights_path)
    return model

root = tk.Tk()
root.title("Voice Emotion Detection")
root.geometry('800x600')
root.configure(background='#CDCDCD')


classifiers = GenderModel()

emotion_model = load_emotion_model(r'C:\Users\LENEVO\OneDrive\Desktop\Assignment1\model_a.json'
                                   ,r'C:\Users\LENEVO\OneDrive\Desktop\Assignment1\model_wieghts.wieght.h5')
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def preprocess_audio(audio_path):
    
    audio, sr = librosa.load(audio_path, sr=None)
    
    
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    
    
    
    return mfccs_mean
def predict_emotion(audio_path):
    try:
        processed_audio = preprocess_audio(audio_path)
        emotion_prediction = emotion_model.predict(np.expand_dims(processed_audio, axis=0))
        pred_emotion = EMOTIONS_LIST[np.argmax(emotion_prediction)]
        return pred_emotion
    except Exception as e:
        print(f"Error predicting emotion: {str(e)}")
        return None
    

def is_female_voice(audio_path):
    try:
        processed_audio = preprocess_audio(audio_path)
        votes = []
        for name, clf in classifiers.items():
            prediction = clf.predict(processed_audio.reshape(1, -1))
            votes.append(prediction[0])

        female_count = sum(votes)
        return female_count > len(votes) / 2

    except Exception as e:
        print(f"Error classifying gender: {str(e)}")
        return False

def record_and_detect_emotion():
    duration = 5  
    fs = 44100 
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
    sd.wait()  
    filename = "temp.wav"
    sf.write(filename, recording, fs)
    
    
    if not is_female_voice(filename):
        messagebox.showerror("Error", "Please speak in a female voice.")
        return
    
    
    predicted_emotion = predict_emotion(filename)
    
    if predicted_emotion:
        emotion_label.config(text=f"Detected Emotion: {predicted_emotion}")
    else:
        messagebox.showerror("Error", "Unable to detect emotion.")


header_label = Label(root, text="Voice Emotion Detection", font=("Helvetica", 24), background='#CDCDCD')
header_label.pack(pady=20)

record_button = Button(root, text="Record and Detect Emotion", font=("Helvetica", 16), command=record_and_detect_emotion)
record_button.pack(pady=50)

emotion_label = Label(root, text="Detected Emotion: ", font=("Helvetica", 18), background='#CDCDCD')
emotion_label.pack(pady=20)


root.mainloop()