from __future__ import print_function

import glob
import os
import sys
import math 
import numpy as np

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

import random
import time
import numpy as np
import cv2
actor_list = []


fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (640,480))

# ==============================================================================
# -- Image Processing Functions ------------------------------------------------
# ==============================================================================
IM_WIDTH = 640
IM_HEIGHT = 480
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
def draw_lines(image, lines, headingLine): 
    lines_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(lines_image, (x1,y1), (x2,y2), (255, 0 , 0), 10)
    x1,y1,x2,y2 = headingLine
    cv2.line(lines_image, (x1,y1), (x2,y2), (0, 0 , 255), 5)
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
            if slope<=-0.5 and slope>-0.65:
                left_fit.append((slope,intercept))
            elif (slope>0.5 and slope<0.65):
                right_fit.append((slope,intercept))
    
    left_fit_average = np.average(left_fit, axis = 0 )    
    right_fit_average = np.average(right_fit, axis = 0 ) 
    
    if np.isnan(np.sum(left_fit_average)):  
        print(f"Left lane was not there")
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            parameters = np.polyfit((x1,x2), (y1,y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope<=-0.4 and slope>-0.65:
                left_fit.append((slope,intercept))
    elif np.isnan(np.sum(right_fit_average)): 
        print(f"Right lane was not there")
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            parameters = np.polyfit((x1,x2), (y1,y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if (slope>0.4 and slope<0.65):
                right_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit, axis = 0 )    
    right_fit_average = np.average(right_fit, axis = 0 )   

    #-------------------------------------------------
    if np.isnan(np.sum(left_fit_average)):  
        print(f"Left lane was not there AGAIN")
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            parameters = np.polyfit((x1,x2), (y1,y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if (slope>0.5 and slope<0.65):
                left_fit.append((-slope,intercept+330))
    elif np.isnan(np.sum(right_fit_average)): 
        print(f"Right lane was not there AGAIN")
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            parameters = np.polyfit((x1,x2), (y1,y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope<=-0.5 and slope>-0.65:
                right_fit.append((-slope,intercept-330))


    left_fit_average = np.average(left_fit, axis = 0 )    
    right_fit_average = np.average(right_fit, axis = 0 )   
    left_line = coordinates(image,left_fit_average)
    right_line = coordinates(image,right_fit_average)
    
    print(left_fit_average, right_fit_average )
    return np.array([left_line, right_line])


def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    canny_img = canny(i3) 
    

    roi_vertices = [
        (0,IM_HEIGHT),
        (IM_WIDTH/5,IM_HEIGHT/2),
        (4*IM_WIDTH/5,IM_HEIGHT/2),
        (IM_WIDTH,IM_HEIGHT)
        ]
    
    masked = region_of_interest(canny_img, np.array([roi_vertices], np.int32),)
    lines = cv2.HoughLinesP(masked, 1, np.pi/180, 25, np.array([]), minLineLength=10, maxLineGap=20)
    averaged_lines = def_avg(i3,lines)


    _,_,left_x2,_ = averaged_lines[0]
    _,_,right_x2,_ = averaged_lines[1]
    mid = IM_WIDTH/2
    x_offset = (left_x2 + right_x2)/2 -mid
    y_offset = IM_HEIGHT/2
    # print(left_x2,right_x2,mid,x_offset,y_offset)

    
    angle_to_mid_radian = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical line
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angle (in degrees) to center vertical line
    st = angle_to_mid_deg + 90 

    st = st / 180.0 * math.pi
    x1 = int(IM_WIDTH / 2)
    y1 = IM_HEIGHT
    x2 = int(x1 - IM_HEIGHT / 2 / math.tan(st)) 
    y2 = int(IM_HEIGHT / 2)
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 5)
    steering_angle = angle_to_mid_deg / 70

    # Clamp the steering command to valid bounds
    steer = np.fmax(np.fmin(steering_angle, 1.0), -1.0)
    print("steering angle", steer)
    print("-----------------------------")

    lines_image = draw_lines(i3, averaged_lines, np.array([x1, y1, x2, y2]))
    combo = cv2.addWeighted(i3, 0.8, lines_image, 1, 1)
    cv2.imshow("", combo)
    cv2.waitKey(1)
    out.write(combo) 
    return combo



def main():

    try:
        print('started')
        client = carla.Client('localhost',2000)
        client.set_timeout(10.0)
        world = client.load_world('Town03')
        # world = client.get_world()
        map  =  world.get_map()
        print('townmap')
        blueprint_library = world.get_blueprint_library()
        
        bp = blueprint_library.filter('a2')[0]
        print(bp)
        list = world.get_map().get_spawn_points()
        spawn_point = list[2]
        print(spawn_point)
        vehicle = world.spawn_actor(bp, spawn_point)
        vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=0.0))

        # vehicle.set_autopilot(True)  # if you just wanted some NPCs to drive.

        actor_list.append(vehicle)
        # get the blueprint for this sensor
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        # change the dimensions of the image
        camera_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
        camera_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
        camera_bp.set_attribute('fov', '110')

        # Adjust sensor relative to vehicle
        camera_tranform = carla.Transform(carla.Location(x=3, z=1))

        
        # spawn the sensor and attach to vehicle.
        sensor = world.spawn_actor(camera_bp, camera_tranform, attach_to=vehicle)

        # add sensor to list of actors
        actor_list.append(sensor)

        # do something with this sensor
        sensor.listen(lambda data: process_img(data))
        # sensor.listen(lambda data: print(data))


        time.sleep(80)
    finally:

        print('destroying actors')
        for actor in actor_list:
            actor.destroy()
        print('done.')    

if __name__ == '__main__':
    main()