import cv2
import numpy as np
import skimage.exposure as exposure
import os

directory = 'C:/Users/cmastral/Desktop/diplw/traffic_light_recognition/'
count = 0
for i in range(1, 101): 
    txt_f = str(i) +".txt"
    
    filename = str(i)+".png"
    print(filename)

    # read image
    img = cv2.imread(filename)

    # 2D histograms for pairs of channels: GR
    GR_hist = cv2.calcHist([img], [1, 2], None, [256, 256], [0, 256, 0, 256])

    # histogram is float and counts need to be scale to range 0 to 255
    scaled_histogramm = exposure.rescale_intensity(GR_hist, in_range=(0,1), out_range=(0,255)).clip(0,255).astype(np.uint8)

    # Masks
    ww = 256
    hh = 256
    ww13 = ww // 3
    ww23 = 2 * ww13
    hh13 = hh // 3
    hh23 = 2 * hh13
    black = np.zeros_like(scaled_histogramm, dtype=np.uint8)

    # specify points in OpenCV x,y format
    ptsUR = np.array( [[[ww13,0],[ww-1,hh23],[ww-1,0]]], dtype=np.int32 )
    redMask = black.copy()
    cv2.fillPoly(redMask, ptsUR, (255,255,255))
    ptsBL = np.array( [[[0,hh13],[ww23,hh-1],[0,hh-1]]], dtype=np.int32 )
    greenMask = black.copy()
    cv2.fillPoly(greenMask, ptsBL, (255,255,255))

    #Test histogram against masks
    region = cv2.bitwise_and(scaled_histogramm,scaled_histogramm,mask=redMask)
    redCount = np.count_nonzero(region)
    region = cv2.bitwise_and(scaled_histogramm,scaled_histogramm,mask=greenMask)
    greenCount = np.count_nonzero(region)
    print('redCount:',redCount)
    print('greenCount:',greenCount)

    # Color recognition
    threshCount = 100
    if redCount > greenCount and redCount > threshCount:
        color = "red"
    elif greenCount > redCount and greenCount > threshCount:
        color = "green"
    elif redCount < threshCount and greenCount < threshCount:
        color = "yellow"
    else:
        color = "other"
    print("color: ", color)  

    with open(txt_f) as f:
        lines = f.readlines()
    print(lines)

    if np.array(color) == lines:
        print('ok')
        count = count+1
    else:
        print(i, 'nope')

print(count/100)    
    # # save result
# cv2.imshow("hist", histScaled)
    # cv2.imwrite('redMask.jpg', redMask)
    # cv2.imwrite('greenMask.jpg', greenMask)

    # # view results
# cv2.imshow("hist", histGR)
# cv2.imshow("redMask", redMask)
# cv2.imshow("greenMask", greenMask)
# cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print('')