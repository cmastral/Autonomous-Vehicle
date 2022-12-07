# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../EggFiles/carla-0.9.7-py2.7-linux-x86_64.egg')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import carla
import time
from time import sleep
import math
import numpy as np
from scipy import interpolate
import random
import copy

# ==============================================================================
# -- Implementation ------------------------------------------------------------
# ==============================================================================
SEARCH_STEP = 2


class Node:
    """A node class for A* Path"""

    def __init__(self, parent=None, position=None, yaw=0):
        self.parent = parent
        self.position = position
        self.map_yaw = yaw
        self.g = 0.0
        self.h = 1e10
        self.f = 1e10

    def __eq__(self, other):
        return math.hypot(self.position.x - other.position.x, self.position.y - other.position.y) <= math.sqrt(0.5*SEARCH_STEP)

    def goal(self, other):
        return math.hypot(self.position.x - other.position.x, self.position.y - other.position.y) <= math.sqrt(2*SEARCH_STEP)

    def calculate_g(self):
        if self.parent is not None:
            self.g = self.parent.g + 1.0 + self.position.yaw**10
        else:
            self.g = 0.0

    def calcute_h(self, goal_position):
        # Euclidean distance from the goal
        self.h = math.hypot(goal_position.y - self.position.y, goal_position.x - self.position.x)

    def calculate_f(self):
        self.f = self.g + self.h


class Position:
    """The position of a node and the function to check if can be permitted"""
    def __init__(self, x, y, yaw=0.0, lane_id=False, junction=False, lane_type="Driving"):
        self.x = x
        self.y = y

        # change of yaw related to parent node
        self.yaw = yaw

        # Lane id and road
        self.same_lane_id = lane_id
        self.is_junction = junction
        self.lane_type = lane_type

    def position_permitted(self):
        if (not self.same_lane_id and not self.is_junction) or abs(self.yaw) > math.pi/2 or self.lane_type != "Driving":
            return True
        return True


class AStar:
    def __init__(self, start_position, end_position, the_map, world):
        # Start and end node
        self.the_map = the_map
        self.world = world
        self.start_node = Node()
        self.end_node = Node()
        self.init_start_end_positions(start_position, end_position, the_map)
        self.travel_step = SEARCH_STEP
        # Initialize open and closed list
        self.open_list = []
        self.closed_list = []
        # Start node
        self.open_list.append(self.start_node)
        self.initial_path = []
        self.smoothed_path = []
        self.world.debug.draw_string(carla.Location(x=self.start_node.position.x, y=self.start_node.position.y),
                                     "START", draw_shadow=False, color=carla.Color(r=200, g=0, b=255),
                                     life_time=40, persistent_lines=True)
        self.world.debug.draw_string(carla.Location(x=self.end_node.position.x, y=self.end_node.position.y),
                                     "END", draw_shadow=False,
                                     color=carla.Color(r=200, g=0, b=255), life_time=40, persistent_lines=True)

    def init_start_end_positions(self, start_position, end_position, the_map):
        x = start_position[0]
        y = start_position[1]
        w_start = the_map.get_waypoint(carla.Location(x=x, y=y), lane_type=carla.LaneType.Driving)
        yaw_start = math.radians(w_start.transform.rotation.yaw)
        w_start = w_start.transform.location
        x = end_position[0]
        y = end_position[1]
        w_end = the_map.get_waypoint(carla.Location(x=x, y=y), lane_type=carla.LaneType.Driving)
        yaw_end = math.radians(w_end.transform.rotation.yaw)
        w_end = w_end.transform.location
        self.start_node = Node(None, Position(w_start.x, w_start.y, yaw_start))
        self.end_node = Node(None, Position(w_end.x, w_end.y, yaw_end))

    def get_adjacent_positions(self, node):
        waypoint = self.the_map.get_waypoint(carla.Location(x=node.position.x, y=node.position.y), lane_type=carla.LaneType.Any)
        node_lane_id = str(waypoint.road_id) + str(waypoint.section_id) + str(waypoint.lane_id)
        #waypoint = node.position
        positions_list = []
        w_list = waypoint.next(self.travel_step)
        for w_i in w_list:
            w_list = w_list + w_i.next(1.5*self.travel_step)
        lc = str(waypoint.lane_change)
        if waypoint.is_junction and False:
            if lc in ["Right", "Both"]:
                w_r = waypoint.get_right_lane()
                if w_r is not None:
                    w_list.append(w_r)
            if lc in ["Left", "Both"]:
                w_l = waypoint.get_left_lane()
                if w_l is not None:
                    w_list.append(w_l)
                    
        for point in w_list:
            x = point.transform.location.x
            y = point.transform.location.y
            yaw = point.transform.rotation.yaw
            point_lane_id = str(point.road_id) + str(point.section_id) + str(point.lane_id)
            is_junction = point.is_junction or node.position.is_junction
            yaw_change = math.acos(math.cos(yaw) * math.cos(node.map_yaw) + math.sin(yaw) * math.sin(node.map_yaw)) % math.pi
            positions_list.append(Position(x, y, yaw_change, point_lane_id == node_lane_id, is_junction, str(point.lane_type)))
        return positions_list

    def find_path(self):
        """Returns a list of tuples as a path from the given start to the given end in the given maze"""
        # Loop until you find the end
        while len(self.open_list) > 0:
            #sleep(0.005)
            # Get the current node, the node with the lower f value
            f_values = [node.f for node in self.open_list]
            current_index = f_values.index(min(f_values))
            current_node = self.open_list[current_index]
            # Pop current off open list, add to closed list
            self.open_list.pop(current_index)
            self.closed_list.append(current_node)

            if current_node.reach_goal(self.end_node):
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                self.initial_path = path[::-1]
                return self.initial_path  # Return reversed path

            # Generate children
            children = []
            for new_pos in self.get_adjacent_positions(current_node):  # Adjacent squares
                # Make sure is in a driving path
                if not new_pos.position_permitted():
                    continue
                self.world.debug.draw_string(carla.Location(x=new_pos.x, y=new_pos.y), "X", draw_shadow=False,
                                             color=carla.Color(r=2, g=200, b=0),
                                             life_time=5.5, persistent_lines=True)
                # Create new node
                waypoint = self.the_map.get_waypoint(carla.Location(x=new_pos.x, y=new_pos.y),
                                                     lane_type=carla.LaneType.Driving)
                map_yaw = math.radians(waypoint.transform.rotation.yaw)
                new_node = Node(current_node, new_pos, map_yaw)

                children.append(new_node)
            # Loop through children
            for child in children:
                # Child is on the closed list
                in_closed_list = False
                for closed_child in self.closed_list:
                    if child == closed_child:
                        in_closed_list = True
                        break
                if in_closed_list:
                    continue

                # Create the f, g, and h values
                child.calculate_g()
                child.calculate_h(self.end_node.position)
                child.calculate_f()

                # Child is already in the open list
                in_open_list = False
                for i, open_node in enumerate(self.open_list):
                    if child == open_node:
                        if child.g < open_node.g:
                            self.open_list[i] = child
                        in_open_list = True
                        break
                if in_open_list:
                    continue

                # Add the child to the open list
                self.open_list.append(child)


        # If there is no path directly to the points choose the closer
        h_values = [node.h for node in self.closed_list]
        current_index = h_values.index(min(h_values))
        current_node = self.closed_list[current_index]
        path = []
        current = current_node
        while current is not None:
            path.append(current.position)
            current = current.parent
        self.initial_path = path[::-1]

        return self.initial_path  # Return reversed path


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)  # seconds
    world = client.load_world('Town02')
    # world = client.get_world()
    the_map = world.get_map()
    # start = (5, -100)
    # end = (-10, 140)
    spawn_points = world.get_map().get_spawn_points()
    start = (carla.Location(spawn_points[0].location).x, carla.Location(spawn_points[0].location).y)
    # print(start)
    end = (carla.Location(spawn_points[10].location).x, carla.Location(spawn_points[10].location).y)

    a_star = AStar(start, end, the_map, world)
    smoothed_path = a_star.find_path()
    if smoothed_path is not None:
        if True:
            for p in smoothed_path:
                sleep(0.001)
                world.debug.draw_string(carla.Location(x=p.x, y=p.y), "o", draw_shadow=False,
                                        color=carla.Color(r=200, g=0, b=20),
                                        life_time=15, persistent_lines=True)
    else:
        print("Path not found")
        world.debug.draw_string(carla.Location(x=start[1], y=start[1]), "Path not found", draw_shadow=False,
                                color=carla.Color(r=200, g=0, b=20),
                                life_time=5, persistent_lines=True)


if __name__ == '__main__':
    main()
