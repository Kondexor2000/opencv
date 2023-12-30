import cv2
import numpy as np

# Wczytaj obraz
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

# Sprawdź, czy wykryto jakieś kontury (pszczół)
if len(contours_yellow) > 0 and len(contours_black) > 0:
    print("Znaleziono pszczoły!")
    
    # Przejdź przez kontury i narysuj prostokąty wokół obszarów żółtych
    for contour_black in contours_black:
        x, y, w, h = cv2.boundingRect(contour_black)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Przejdź przez kontury i narysuj prostokąty wokół obszarów żółtych
    for contour_yellow in contours_yellow:
        x, y, w, h = cv2.boundingRect(contour_yellow)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Wyświetl obraz z naniesionymi prostokątami
    cv2.imshow('Detekcja Pszczoł', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Nie znaleziono pszczoł.")