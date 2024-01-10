import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

def detect_tooth_color(image_path):
    # Wczytaj obraz
    img = cv2.imread(image_path)

    # Sprawdź, czy obraz został wczytany poprawnie
    if img is None:
        print(f"Błąd: Nie udało się wczytać obrazu z {image_path}")
        return None, None

    # Konwersja BGR na HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Zakres barw żółtych
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Maska dla koloru żółtego
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Wyciągnij tylko obszary żółte z obrazu
    yellow_teeth = cv2.bitwise_and(img, img, mask=yellow_mask)

    # Sprawdź, czy większość zębów jest żółta
    yellow_pixel_count = cv2.countNonZero(yellow_mask)
    total_pixel_count = img.shape[0] * img.shape[1]

    yellow_percentage = (yellow_pixel_count / total_pixel_count) * 100

    if yellow_percentage > 10:  # Prog żółtości zębów
        print("Zęby są żółte.")
        tooth_label = 1  # 1 oznacza żółte zęby
    else:
        print("Zęby są białe.")
        tooth_label = 0  # 0 oznacza białe zęby

    return yellow_teeth, tooth_label

image_path_none = r'C:\Users\kondz\OneDrive\Pulpit\nlp\Zeby\zeby_biale.jpg'
tooth_img, tooth_label = detect_tooth_color(image_path_none)

# Przygotuj dane treningowe
data = []
labels = []

# Dodaj obraz zębów jako dane treningowe
if tooth_img is not None and tooth_label is not None:
    resized_tooth_image = cv2.resize(tooth_img, (32, 32))
    data.append(resized_tooth_image)
    labels.append(tooth_label)

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
model.add(Dense(2, activation='softmax'))  # 2 neurony, bo mamy dwie klasy (żółte i białe zęby)

# Kompiluj model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Powtarzaj trening przez 10 epok
for epoch in range(20):
    print(f"Epoch {epoch + 1}/{30}")
    # Trenuj model
    model.fit(data, labels, epochs=10, batch_size=1)

    # Możesz teraz użyć modelu do predykcji na innych obrazach zębów.
    # Wczytaj obrazy testowe
    test_image_1 = cv2.imread(image_path_none)  # Replace with your dental image path

    # Konwertuj obrazy testowe do wymaganego formatu
    test_data = [
        cv2.resize(test_image_1, (32, 32))
    ]
    test_data = np.array(test_data)
    test_data = test_data.astype('float32') / 255.0

    # Przeprowadź predykcję na danych testowych
    predictions = model.predict(test_data)

    # Wyświetl wyniki predykcji
    for i, prediction in enumerate(predictions):
        print(f'Obraz testowy {i + 1}:')
        print(f'Prawdopodobieństwo, że to żółte zęby: {prediction[1]}')

        # Sprawdź, czy przewidywane prawdopodobieństwo dla klasy "żółte zęby" przekracza próg
        if prediction[1] > 0.5:
            print("Zęby są żółte")
        else:
            print("Zęby są białe")