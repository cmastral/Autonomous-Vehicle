# Thesis Title

Camera-driven behavioral planning for autonomous vehicles on the CARLA
Simulator

The purpose of the system is to navigate on the road recognized via optical means. The Computer Vision OpenCV library is extensively used throughout the project for preprocessing the received RGB image, so as to make it suitable for analysis. The perception system uses the programming interface of the CARLA simulator to integrate an RGB camera. For the identification of dynamic obstacles, visual information is analyzed by the deep learning model YOLO (You Only Look Once) while using Optical Character Recognition (OCR) to export information from road traffic signals and a histogram analysis is performed for traffic light color recognition. In the global path construction system, a guided graph of the map is created and the algorithm A* is used to search for the optimal route. Finally, a Lane Keeping Assistance system and a PID controller are used for the navigation control system.


Mastralexi Christina Maria\
Electrical & Computer Engineering Department,\
Aristotle University of Thessaloniki, Greece\
September 2022

# Prerequisites
First CARLA must be installed on your machine ([Windows/Linux](https://carla.readthedocs.io/en/latest/start_quickstart/#b-package-installation) ).
Please go through CARLA-Setup-Guideand install CARLA and all other dependencies properly.

Install: 
* [Carla Simulator](https://carla.org/)
* [Python 3](https://www.python.org/download/releases/3.0/)
* [Pygame](https://www.pygame.org/news)


All software is free and available in the links above. After installing this software you can download the latest code of the current repository.

# Running CARLA

The method to start a CARLA server depends on the installation method you used and your operating system:
For Windows:

```
 cd path/to/carla/root

 CarlaUE4.exe

```

This is the server simulator which is now running and waiting for a client to connect and interact with the world. 
You can try some of the example scripts or the scripts of this project by running in a new Terminal:

```
 # Terminal B
        cd PythonAPI\examples

        python3 script.py 
```



# CARLA dataset

Download at: https://drive.google.com/drive/folders/1YeOABbyuXxyu1U1rZykbhRvpYQ23bFfF


