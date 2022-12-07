from __future__ import print_function

import glob
import os
import sys

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

IM_WIDTH = 1600
IM_HEIGHT = 1200

# def process_img(image):
#     i = np.array(image.raw_data)
#     i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
#     i3 = i2[:, :, :3]
#     cv2.imshow("", i3)
#     cv2.waitKey(1)
#     return i3/255.0


actor_list = []
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10)

    world = client.get_world()

    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter('model3')[0]
    print(bp)

    spawn_point = random.choice(world.get_map().get_spawn_points())
    weather = carla.WeatherParameters(1)
    # world.apply_settings(carla.WorldSettings(no_rendering_mode=False,synchronous_mode=True,fixed_delta_seconds=0.05))

    world.set_weather(weather)
    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.set_autopilot(True)  # if you just wanted some NPCs to drive.

    actor_list.append(vehicle)
    
    cam_bp = None
    cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
    cam_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
    cam_bp.set_attribute('fov', '110')
    cam_location = carla.Location(2,0,2)
    cam_rotation = carla.Rotation(0,0,0)
    cam_transform = carla.Transform(cam_location,cam_rotation)
    ego_cam = world.spawn_actor(cam_bp,cam_transform,attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
    # ego_cam.listen(lambda image: image.save_to_disk('C:/Users/cmastral/Desktop/Town05/dataset--%.6d.jpg' % image.frame))
    ego_cam.save_to_disk('C:/Users/cmastral/Desktop/Town05/dataset--%.6d.jpg' % ego_cam.frame)    
    
    actor_list.append(ego_cam)


    time.sleep(50)

finally:
    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
   