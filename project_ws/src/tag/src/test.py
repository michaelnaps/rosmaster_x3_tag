#! /usr/bin/env python3

import os

import sys
import rospy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped

import numpy as np
import Geometry as gm


rospy.init_node('tag');


def callback(msg):
    print(msg.pose.pose.position.x);
    return;


def get_name_and_id():
    # THIS ROBOTS NAME AND ID ARE:
    my_name = rospy.get_namespace();

    my_id = np.nan;
    robots = ('bernard', 'scrappy', 'oswaldo');
    for i, robot in enumerate(robots):
        if robot == my_name:
            my_id = i;

    return my_name, my_id;


def init_environment(my_name, publisher):
    # establish environment
    bound_w = 3;
    bound_h = 3;
    bounds = np.array([
        [-bound_w/2, bound_w/2, bound_w/2, -bound_w/2],
        [bound_h/2, bound_h/2, -bound_h/2, -bound_h/2]
    ]);

    env_center = np.array([[0],[0]])
    env_radius = -np.sqrt((bound_w/2)**2 + (bound_h/2)**2);

    wall_gain = 1;
    walls = (gm.Polygon(bounds), gm.Sphere(env_center, env_radius));

    # establish robot variables
    robot_radius = 0.15;
    tag_radius = 0.15;
    pursuer_gain = 1;

    b_c = np.array([[0.5],[0.5]]);
    sphere_temp = gm.Sphere(b_c, robot_radius, tag_radius);
    bernard = gm.Robot(sphere_temp, 'pursuer', 'yellowgreen', 'bernard');

    s_c = np.array([[0],[0]]);
    sphere_temp = gm.Sphere(s_c, robot_radius, tag_radius);
    scrappy = gm.Robot(sphere_temp, 'evader', 'firebrick', 'scrappy');

    o_c = np.array([[-0.5],[-0.5]]);
    sphere_temp = gm.Sphere(o_c, robot_radius, tag_radius);
    oswaldo = gm.Robot(sphere_temp, 'evader', 'mediumpurple', 'oswaldo');

    robots = (bernard, scrappy, oswaldo);

    return (walls, robots);


if __name__ == "__main__":
    rate = rospy.Rate(1);
    DesiredTrajectory = Twist();

    my_name, my_id = get_name_and_id();

    pub = rospy.Publisher('cmd_vel', Twist, queue_size=10);

    walls, robots = init_environment(my_name, pub);
    World = gm.RobotEnvironment(walls, robots, pub);

    sub = rospy.Subscriber('/oswaldo/amcl_pose', PoseWithCovarianceStamped, World.oswaldo_callback);

    rospy.spin();
