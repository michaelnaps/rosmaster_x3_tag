#! /usr/bin/env python3

import os

import sys
import rospy
from std_msgs.msg import Int64
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist

import numpy as np
import Geometry as gm


rospy.init_node('tag');


def get_name_and_id():
    # THIS ROBOTS NAME AND ID ARE:
    my_name = sys.argv[1];

    my_id = np.nan;
    robots = ('bernard', 'scrappy', 'oswaldo');
    for i, robot in enumerate(robots):
        if robot == my_name:
            my_id = i;

    return my_name, my_id;


def init_environment(my_name, publisher):
    # establish environment
    bound_w = 2.75;
    bound_h = 2.10;
    bounds = np.array([
        [-bound_w/2, bound_w/2, bound_w/2, -bound_w/2],
        [bound_h/2, bound_h/2, -bound_h/2, -bound_h/2]
    ]);

    env_center = np.array([[0],[0]])
    env_radius = -np.sqrt((bound_w/2)**2 + (bound_h/2)**2);

    wall_gain = 1;
    walls = (gm.Polygon(bounds), gm.Sphere(env_center, env_radius));

    # establish robot variables
    robot_radius = 0.11;
    tag_radius = 0.11;
    pursuer_gain = 1;

    b_c = np.array([[0.5],[0.5]]);
    sphere_temp = gm.Sphere(b_c, robot_radius, tag_radius);
    bernard = gm.Robot(sphere_temp, 'evader', 'yellowgreen', 'bernard');

    s_c = np.array([[-0.5],[0.5]]);
    sphere_temp = gm.Sphere(s_c, robot_radius, tag_radius);
    scrappy = gm.Robot(sphere_temp, 'evader', 'firebrick', 'scrappy');

    o_c = np.array([[0.5],[-0.5]]);
    sphere_temp = gm.Sphere(o_c, robot_radius, tag_radius);
    oswaldo = gm.Robot(sphere_temp, 'pursuer', 'mediumpurple', 'oswaldo');

    robots = (bernard, scrappy, oswaldo);

    return (walls, robots);


if __name__ == "__main__":
    rate = rospy.Rate(10);
    DesiredTrajectory = Twist();

    my_name, my_id = get_name_and_id();

    pub = rospy.Publisher(my_name + '_cmd_vel', Twist, queue_size=10);

    InitialPush = Twist();

    InitialPush.linear.x = -0.5;
    InitialPush.linear.y = -0.5;
    InitialPush.linear.z = 0;

    InitialPush.angular.x = 0;
    InitialPush.angular.y = 0;
    InitialPush.angular.z = 0;

    pub.publish(InitialPush);

    walls, robots = init_environment(my_name, pub);
    World = gm.RobotEnvironment(walls, robots, my_id, pub);

    sub = rospy.Subscriber('bernard_role', Int64, World.bernard_role_callback);
    sub = rospy.Subscriber('scrappy_role', Int64, World.scrappy_role_callback);
    sub = rospy.Subscriber('oswaldo_role', Int64, World.oswaldo_role_callback);

    sub = rospy.Subscriber('bernard_pose', PoseWithCovarianceStamped, World.bernard_position_callback);
    sub = rospy.Subscriber('scrappy_pose', PoseWithCovarianceStamped, World.scrappy_position_callback);
    sub = rospy.Subscriber('oswaldo_pose', PoseWithCovarianceStamped, World.oswaldo_position_callback);

    rospy.spin();
