import cv2
import numpy as np

def detect_tooth_color(image_path):
    # Wczytaj obraz
    img = cv2.imread(image_path)

    # Konwersja z BGR do HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Zakres barw żółtych
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Maska dla koloru żółtego
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Wyciągnij tylko obszary żółte z obrazu
    yellow_teeth = cv2.bitwise_and(img, img, mask=yellow_mask)

    # Wyświetl obrazy
    cv2.imshow('Original Image', img)
    cv2.imshow('Yellow Teeth Only', yellow_teeth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Sprawdź, czy większość zębów jest żółta
    yellow_pixel_count = cv2.countNonZero(yellow_mask)
    total_pixel_count = img.shape[0] * img.shape[1]

    yellow_percentage = (yellow_pixel_count / total_pixel_count) * 100

    if yellow_percentage > 10:  # Prog żółtości zębów
        print("Zęby są żółte.")
    else:
        print("Zęby są białe.")

# Przykładowe użycie
image_path = r'C:\Users\kondz\OneDrive\Pulpit\nlp\zeby_zolte.jpg'
detect_tooth_color(image_path)