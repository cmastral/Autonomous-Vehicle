import cv2
import numpy as np
import math

# # Read image
image = cv2.imread('lane/14.png')
# Copy image
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
    canny = cv2.Canny(blur, 50, 60) 
    return canny

# Region Of Interest Function: Specify the region of interest 
# NOTE: Why? We want a mask (a spacific portion of the image) and everything else 0 so it will only show the region of interest
def region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    match_mask_color = (255,)
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Draw the lines Function
# After Hough Transform we have a set of lines detected that we want to draw on the image 
def draw_lines(image, lines): 
    lines_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(lines_image, (x1,y1), (x2,y2), (255, 0 , 0), 10)
    return lines_image    

# Coordinates Function
# Returns an array with coordinates based on the line parameters
def coordinates(image,line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int((y1*3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1, x2,y2])

# Define the average Functions
# Returns the coordinates of the left and right line 
# NOTE: At first we prefer to find the average among the lines with slope<-0.5 for the left line and slope>0.5 for the right line to prevent 
# calculations that include the "outliers", like lines that don't belong in the main road we're interested in or lines from noise. In case we can't find 
# lines that fit this threshold for the slope, we find the average among all the lines for this (left/right) side and we make a less accurate prediction. 

def def_avg(image,lines):
    left_fit = [] 
    right_fit = []
    for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            parameters = np.polyfit((x1,x2), (y1,y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope<=-0.5:
                left_fit.append((slope,intercept))
            elif slope>0.5:
                right_fit.append((slope,intercept))
    
    left_fit_average = np.average(left_fit, axis = 0 )    
    right_fit_average = np.average(right_fit, axis = 0 ) 
    # print(f"Right fit Avg ", right_fit_average)
    # print(f"Left fit Avg ", left_fit_average)

    if np.isnan(np.sum(left_fit_average)):  
        print(f"Left lane was not there")
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            parameters = np.polyfit((x1,x2), (y1,y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope<=-0.3:
                left_fit.append((slope,intercept))
    elif np.isnan(np.sum(right_fit_average)): 
        print(f"Right lane was not there")
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            parameters = np.polyfit((x1,x2), (y1,y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope>0.3:
                right_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit, axis = 0 )    
    right_fit_average = np.average(right_fit, axis = 0 )   
    # print(f"Right fit Avg ", right_fit_average)
    # print(f"Left fit Avg ", left_fit_average)
    #-------------------------------------------------
    if np.isnan(np.sum(left_fit_average)):  
        print(f"Left lane was not there AGAIN")
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            parameters = np.polyfit((x1,x2), (y1,y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope>0.5:
                left_fit.append((-slope,intercept+330))
    elif np.isnan(np.sum(right_fit_average)): 
        print(f"Right lane was not there AGAIN")
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            parameters = np.polyfit((x1,x2), (y1,y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope<=-0.5:
                right_fit.append((-slope,intercept-330))


    left_fit_average = np.average(left_fit, axis = 0 )    
    right_fit_average = np.average(right_fit, axis = 0 )   
    # print(f"Final Right fit Avg ", right_fit_average)
    # print(f"Final Left fit Avg ", left_fit_average)

    # left_fit_average = [-0.5, 408]
    # right_fit_average = [0.6, 42]
    left_line = coordinates(image,left_fit_average)
    right_line = coordinates(image,right_fit_average)

    # print(f"Right line ", right_line)
    # print(f"Left line ", left_line)
    
    return np.array([left_line, right_line])
       
                                

# Process Image Function
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
    lines = cv2.HoughLinesP(masked, 1, np.pi/180, 70, np.array([]), 40, 5 )
    averaged_lines = def_avg(image,lines)
    _,_,left_x2,_ = averaged_lines[0]
    _,_,right_x2,_ = averaged_lines[1]
    mid = width/2
    x_offset = (left_x2 + right_x2)/2 -mid
    y_offset = height/2
    # print(left_x2,right_x2,mid,x_offset,y_offset)

    angle_to_mid_radian = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical line
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angle (in degrees) to center vertical line
    st = angle_to_mid_deg + 90 

    st = st / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(st)) 
    y2 = int(height / 2)
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 5)
    steering_angle = angle_to_mid_deg / 70

    # Clamp the steering command to valid bounds
    steer = np.fmax(np.fmin(steering_angle, 1.0), -1.0)
    print("steering angle", steer)

    lines_image = draw_lines(image, averaged_lines)
    combo = cv2.addWeighted(image, 0.8, lines_image, 1, 1)
    return combo

# Images
combo = process(lane_image)
cv2.imshow('Result', combo)
cv2.imwrite('heading.jpg', combo)

cv2.waitKey(0)

# # Video  
# cap = cv2.VideoCapture('solidWhiteRight.mp4')

# while cap.isOpened():
#     ret, frame = cap.read()
#     frame = process(frame)
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

# cap.release()
# cv2.destroyAllWindows()