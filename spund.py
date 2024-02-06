import sounddevice as sd
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense

def build_sound_model(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def recognize_sound_intensity(model):
    audio_data = sd.rec(int(5 * 44100), samplerate=44100, channels=1, dtype=np.int16)
    sd.wait()

    processed_audio_data = preprocess_audio_data(audio_data)

    sound_intensity = model.predict(processed_audio_data.reshape(1, -1))[0][0]

    return sound_intensity

def preprocess_audio_data(audio_data):
    return audio_data.flatten()

def main():
    sound_model = build_sound_model((44100 * 5,)) 

    while True:
        sound_intensity = recognize_sound_intensity(sound_model)

        threshold = 0.5  
        if sound_intensity > threshold:  
            notify = True  
            if notify:
                print("Zbyt głośny dźwięk")
        else:
            notify = False

if __name__ == "__main__":
    main()