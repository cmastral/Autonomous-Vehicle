import cv2
from matplotlib import lines
import numpy as np
import math


directory = 'C:/Users/cmastral/Desktop/diplwmatiki/Perception/intersections/int/'
count = 0
for i in range(1, 101): 

    filename = str(i)+".jpg"
    print(filename)

    # read image
    image = cv2.imread(filename)

    lane_image = np.copy(image)

    # Grayscale 
    def grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Smoothing (Gaussian Filter) with a 5x5 Kernel to reduce noise
    def gaussian_blur(image):
        return cv2.GaussianBlur(image, (5,5), 0)
    # Canny: Traces the edge with large changing in intesity
    #        by finding the gradient in an outline of white
    #        pixels
    # Canny Function
    def canny(image):
        gray = grayscale(image)
        blur = gaussian_blur(gray)
        canny = cv2.Canny(blur, 50, 100) 
        return canny

    # # Region Of Interest Function: Specify the region of interest 
    # # NOTE: Why? We want a mask (a spacific portion of the image) and everything else 0 so it will only show the region of interest
    def region_of_interest(image, vertices):
        mask = np.zeros_like(image)
        match_mask_color = (255,)
        cv2.fillPoly(mask, vertices, match_mask_color)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image


    # Process Image Function
    flag = 0 
    def process(image):
        canny_img = canny(image)   
        height = image.shape[0] 
        width = image.shape[1]

        roi_vertices = [
            (0, height),
            (width/5, height/2),
            (4*width/5, height/2),
            (width, height)
            ]
        
        masked = region_of_interest(canny_img, np.array([roi_vertices], np.int32),)
        lines = cv2.HoughLinesP(masked, 1, math.pi/2, 2, None, 30, 1)
        print(lines)
        if lines is not None:
            print("Junction")
            for line in lines[0]:
                pt1 = (line[0],line[1])
                pt2 = (line[2],line[3])
                cv2.line(image, pt1, pt2, (0,0,255), 3)
            cv2.putText(img=image, text='Junction ahead', org=(20, 60), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 0),thickness=1)    
            flag = 1
                
        else:
            print("Keep going")
            flag = 0

        return image, flag

    # Images

    combo, flag = process(lane_image)
    print(flag)
    # if flag == 1:
    #     cv2.imshow('Result', combo)
    #     # cv2.imwrite('18-det.jpg', combo)
    #     cv2.waitKey(0)



