#! /usr/bin/env python
#XXX : You need to install PyGame before using this script
import contextlib
with contextlib.redirect_stdout(None):
    import pygame
import numpy as np
import time
from math import atan2, degrees, pi , sqrt


MAX_HEIGHT_SPEED = 100
WIDTH_SPEED = 30
MAX_SPEED=3
MIN_SPEED=-3
LOC_WIDTH_FROM_BOTTOM = 20
LOC_HEIGHT_FROM_BOTTOM = 20
CANVAS_WIDTH=400
CANVAS_HEIGHT=400

def ppoints_to_angle(x1,x2):
     dx = x1[1] - x1[0]
     dy = x2[1] - x2[0]
     rads = atan2(-dy,dx)
     rads %= 2*pi
     degs = degrees(rads)
     return degs

def rotate(image, rect, angle):
    """Rotate the image while keeping its center."""
    # Rotate the original image without modifying it.
    new_image = pygame.transform.rotate(image, angle)
    # Get a new rect with the center of the old rect.
    rect = new_image.get_rect(center=rect.center)
    return new_image, rect

def Hill(p):
     return p*p+p if p < 0 else p/(sqrt(1+5*p*p))

def save_caronthehill_image(position, speed, out_file):

    #Initialization of variables for visualization
    canvas_width = 400
    canvas_height = 400
    screen = pygame.display.set_mode((canvas_width, canvas_height))
    loc_width_from_bottom = 35
    loc_height_from_bottom = 70
    pt_pos1 = -0.5
    pt_pos2 = 0.5
    max_height_speed = 50
    width_speed = 30
    thickness_speed_line=3

    #Image loading
    car = pygame.image.load("car.png")
    pt = pygame.image.load("pine_tree.png")
    car.convert_alpha()
    pt.convert_alpha()
    size_pt = pt.get_rect().size
    size_car = car.get_rect().size
    width_car = size_car[0]
    height_car = size_car[1]
    width_pt = size_pt[0]
    height_pt = size_pt[1]

    #Initialization of variables related to car on the hill
    max_speed=3
    min_speed=-3
    step_hill = 2.0/canvas_width


    #Coloring
    color_hill = pygame.Color(0, 0, 0, 0)
    color_shill = pygame.Color(64, 163, 191, 0)
    color_phill = pygame.Color(64, 191, 114, 0)
    color_acc_line = pygame.Color(0, 0, 0, 0)

    #Surface loading
    surf = pygame.Surface((CANVAS_WIDTH,CANVAS_HEIGHT))
    surf.convert()

    #hill function plot
    points = list(np.arange(-1,1,step_hill))
    hl = list(map(Hill,points))

    #Discretization of the hill function steps


    #Draw the background and the hill function altogether
    range_h = range(canvas_height)
    pix=0
    for h in hl:
        x = pix
        y = ((canvas_height)/2) * (1+h)


        y = int(round(y))
        for yo in range_h:
            if yo < y:
                c = color_phill
            elif yo > y:
                c = color_shill
            surf.set_at((x, canvas_height - yo), c)

        surf.set_at((x, canvas_height - y), color_hill)
        pix += 1

    #Display pine trees
    surf.blit(pt,(round((canvas_width/2)*(1+pt_pos1)) - width_pt/2, canvas_height - round(((canvas_height)/2) * (1+Hill(pt_pos1))) - height_pt))
    surf.blit(pt,(round((canvas_width/2)*(1+pt_pos2)) - width_pt/2, canvas_height - round(((canvas_height)/2) * (1+Hill(pt_pos2))) - height_pt))

    #Display the car
    x_car = round((canvas_width/2)*(1+position)) - width_car/2
    h_car = Hill(position)
    h_car_next = Hill(position + step_hill)
    y_car = canvas_height - round(((canvas_height)/2) * (1+h_car)) - height_car
    angle= ppoints_to_angle((position,position+step_hill),(h_car,h_car_next))
    rot_car, rect = rotate(car, pygame.Rect(x_car,y_car, width_car, height_car), 360-angle)
    surf.blit(rot_car, rect)

    #Display car speed

    #Display black line
    rect = (canvas_width-loc_width_from_bottom - width_speed, canvas_height - loc_height_from_bottom, width_speed, thickness_speed_line)
    surf.fill(color_acc_line, rect)

    pct_speed = abs(speed)/max_speed
    color_speed = (pct_speed * 255,(1-pct_speed)*255,0)
    height_speed = max_height_speed*(pct_speed)



    loc_width = canvas_width - width_speed - loc_width_from_bottom
    loc_height = canvas_height - loc_height_from_bottom + thickness_speed_line  if speed < 0 else canvas_height - loc_height_from_bottom - height_speed
    rect = (loc_width,loc_height,width_speed,height_speed)
    surf.fill(color_speed, rect)

    pygame.image.save(surf, out_file)
    pygame.display.quit()

#Execution example
if __name__=="__main__":
    save_caronthehill_image(0,1,"out.jpeg")
