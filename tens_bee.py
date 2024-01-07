import cv2
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import StratifiedKFold
import random

# Wczytaj obraz pszczół
image = cv2.imread(r'C:\Users\kondz\Downloads\Pszczola_skrzydelka.jpg')

# Konwertuj obraz z BGR do HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Zdefiniuj zakres kolorów żółtych w przestrzeni HSV
lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
upper_yellow = np.array([30, 255, 255], dtype=np.uint8)

# Stwórz maskę dla kolorów żółtych
yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Zdefiniuj zakres kolorów czarnych w przestrzeni HSV
lower_black = np.array([0, 0, 0], dtype=np.uint8)
upper_black = np.array([180, 255, 30], dtype=np.uint8)

# Stwórz maskę dla kolorów czarnych
black_mask = cv2.inRange(hsv, lower_black, upper_black)

# Wykryj kontury na masce
contours_yellow, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Wykryj kontury na masce
contours_black, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Przygotuj dane treningowe
data = []
labels = []

# Sprawdź, czy istnieje obszar, który jest zarówno żółty, jak i czarny
for contour_yellow in contours_yellow:
    for contour_black in contours_black:
        x_yellow, y_yellow, w_yellow, h_yellow = cv2.boundingRect(contour_yellow)
        x_black, y_black, w_black, h_black = cv2.boundingRect(contour_black)

        # Sprawdź nakładanie się obszarów
        if x_yellow < x_black + w_black and x_yellow + w_yellow > x_black and y_yellow < y_black + h_black and y_yellow + h_yellow > y_black:
            # Obszar zawiera zarówno barwy żółte, jak i czarne

            # Dodaj obraz pszczół jako dane treningowe
            bee_image = image[y_yellow:y_yellow+h_yellow, x_yellow:x_yellow+w_yellow]
            resized_bee_image = cv2.resize(bee_image, (32, 32))
            data.append(resized_bee_image)
            labels.append(1)  # 1 oznacza pszczółkę

# Konwertuj dane do numpy arrays
data = np.array(data)
labels = np.array(labels)

# Normalizuj dane obrazów
data = data.astype('float32') / 255.0

# Przekształć etykiety na one-hot encoding
labels = to_categorical(labels, 2)

# Zdefiniuj model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))  # 2 neurony, bo mamy dwie klasy (pszczółki i nie pszczółki)

# Kompiluj model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Przygotuj kroswalidację
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Trenuj model
# model.fit(data, labels, epochs=10, batch_size=64, validation_split=0.2)

# Trenuj model używając kroswalidacji
for train_indices, test_indices in kfold.split(data, labels.argmax(axis=1)):

    random.shuffle(train_indices)
    random.shuffle(test_indices)

    train_data, test_data = data[train_indices], data[test_indices]
    train_labels, test_labels = labels[train_indices], labels[test_indices]

    model.fit(train_data, train_labels, epochs=10, batch_size=64, validation_data=(test_data, test_labels))

    # Wczytaj obrazy testowe
    test_image_1 = cv2.imread(r'C:\Users\kondz\Downloads\Pszczola_skrzydelka.jpg')
    test_image_2 = cv2.imread(r'C:\Users\kondz\OneDrive\Pulpit\nlp\kartka_1.jpg')

    # Konwertuj obrazy testowe do wymaganego formatu
    test_data = [
        cv2.resize(test_image_1, (32, 32)),
        cv2.resize(test_image_2, (32, 32))
    ]
    test_data = np.array(test_data)
    test_data = test_data.astype('float32') / 255.0

    # Przeprowadź predykcję na danych testowych
    predictions = model.predict(test_data)

    # Wyświetl wyniki predykcji
    for i, prediction in enumerate(predictions):
        print(f'Obraz testowy {i + 1}:')
        print(f'Prawdopodobieństwo, że to pszczółka: {prediction[1]}')
        print(f'Prawdopodobieństwo, że to nie pszczółka: {prediction[0]}\n')

        # Check if the predicted probability for class "pszczółka" is greater than a threshold
        if prediction[0] > 0.0:
            print("Są pszczoły")
        else:
            print("Brak pszczołek")