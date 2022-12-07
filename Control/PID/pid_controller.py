from __future__ import print_function

import glob
import math
import os
import numpy.random as random
import sys
import time
import math
import numpy as np


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

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from agents.navigation.global_route_planner import  GlobalRoutePlanner
from agents.navigation.controller import VehiclePIDController


    
def drive(route, vehicle, speed, PID):
    """
    This function drives throught the planned route with the desired speed passed in the argument
    
    """
    i = 0
    target = route[0]
    while True:
        vehicle_loc = vehicle.get_location()
        dist_v = find_distance_veh_to_target(vehicle_loc, target)
        control = PID.run_step(speed, target)
        vehicle.apply_control(control)
        print('Target:', target)
        print('i:', i)
        print('dist_v:', dist_v)
        
        if i == (len(route)-1):
            print("last waypoint reached")
            break 
        
        
        if (dist_v < 3.5):
            control = PID.run_step(speed,target)
            vehicle.apply_control(control)
            i=i+1
            target=route[i]
            

    control = PID.run_step(0,route[len(route)-1])
    vehicle.apply_control(control)
                

# def find_dist(start, destination):
#     """
#     This function finds euclidian distance from source to destination    
#     """
#     dist = math.sqrt( (start.transform.location.x - destination.transform.location.x)**2 + (start.transform.location.y - destination.transform.location.y)**2 )

#     return dist


def find_distance_veh_to_target(vehicle_loc,target):
    """
    This function finds euclidian distance from current vehicle's location to the next target (waypoint) 
    """
    dist = math.sqrt( (target.transform.location.x - vehicle_loc.x)**2 + (target.transform.location.y - vehicle_loc.y)**2 )
    
    return dist
    

def PID_controller(vehicle):
    
    """
    This function creates a PID controller for the vehicle
    """
    args_lateral = {'K_P': 1.95,'K_D': 0.2,'K_I': 0.07,'dt': 1.0 / 10.0}
    args_longitudinal = {'K_P': 1,'K_D': 0.0,'K_I': 0.75,'dt': 1.0 / 10.0}

    PID = VehiclePIDController(vehicle, args_lateral = args_lateral, args_longitudinal = args_longitudinal) 
    return PID


actor_list = []

def main():

    try:
        print('Started')
        client = carla.Client('localhost',2000)
        client.set_timeout(20)

        world = client.load_world('Town03')
        #world = client.get_world()

        map  =  world.get_map()
        print('Townmap')
        sampling_resolution = 2

        grp = GlobalRoutePlanner(map, sampling_resolution)
        spawn_points = world.get_map().get_spawn_points()

        # Set starting and destination points
        start = carla.Location(spawn_points[0].location)
        print(start)
        destination = carla.Location(spawn_points[100].location)
        print(destination)
        waipoints_route = grp.trace_route(start, destination) 

        # Draw starting and destination points
        world.debug.draw_point(start, color = carla.Color(r=255, g=0, b=100), size=0.5 ,life_time=120.0)
        world.debug.draw_point(destination, color = carla.Color(r=255, g=0, b=100), size=0.5 ,life_time=120.0)

        
        wps=[]

        for i in range(len(waipoints_route)):

            # Waypoints
            wps.append(waipoints_route[i][0])
            world.debug.draw_point(waipoints_route[i][0].transform.location,color=carla.Color(r=0, g=0, b=255), size=0.05, life_time=200.0)

        # Set world
        world = client.get_world()
        spawn_point = carla.Transform(start, carla.Rotation(pitch=0.0, yaw=0.0, roll=0.000000))
            
        blueprint_library = world.get_blueprint_library()
        bp = blueprint_library.filter('a2')[0]

        vehicle = world.spawn_actor(bp, spawn_point)
        PID = PID_controller(vehicle)

        # Set Desired speed
        speed = 30
        drive(wps, vehicle, speed, PID)

        time.sleep(10)     

    finally:

        print('Destroying actors')
        for actor in actor_list:
            actor.destroy()
        print('Done.')    



if __name__ == '__main__':

    main()

