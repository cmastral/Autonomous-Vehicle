import cv2
import numpy as np
import pytesseract
import re
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Directory
directory = 'C:/Users/cmastral/Desktop/diplw/Digit_tesseract/'
count = 0
# Loop through images and annotations
for i in range(1, 101): 
    txt_f = str(i) +".txt"
    
    filename = str(i)+".png"
    img = cv2.imread(filename)

    # Convert image to black and white (using adaptive threshold)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 11)
    config = "--psm 9"

    # Text in circle
    text = pytesseract.image_to_string(adaptive_threshold, config=config)

    # Digits 0-9
    text = re.sub('[^0-9]', '', text)

    with open(txt_f) as f:
        lines = f.readlines()

    if np.array(text) == lines:
        # print('ok')
        count = count+1
    else:
        print(i, 'Something went wrong')

    
print(count/100)
