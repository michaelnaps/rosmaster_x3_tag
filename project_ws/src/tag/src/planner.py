#! /usr/bin/env python3

import sys
import rospy
from geometry_msgs.msg import Twist

import numpy as np
import Geometry as gm

def get_name_and_id():
    # THIS ROBOTS NAME AND ID ARE:
    my_argv = rospy.myargv(argv=sys.argv);
    my_name = my_argv[1];

    my_id = np.nan;
    robots = ('bernard', 'scrappy', 'oswaldo');
    for i, robot in enumerate(robots):
        if robot == my_name:
            my_id = i;

    return my_name, my_id;

def init_environment(my_name):
    # establish environment
    bound_w = 5.0;
    bound_h = 3.5;
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

    b_c = np.array([[0.3],[0.3]]);
    sphere_temp = gm.Sphere(b_c, robot_radius, tag_radius);
    bernard = gm.Robot(sphere_temp, 'pursuer', 'yellowgreen', 'bernard');

    s_c = np.array([[0],[0]]);
    sphere_temp = gm.Sphere(s_c, robot_radius, tag_radius);
    scrappy = gm.Robot(sphere_temp, 'evader', 'firebrick', 'scrappy');

    o_c = np.array([[-0.3],[-0.3]]);
    sphere_temp = gm.Sphere(o_c, robot_radius, tag_radius);
    oswaldo = gm.Robot(sphere_temp, 'evader', 'mediumpurple', 'oswaldo');

    robots = (bernard, scrappy, oswaldo);

    return (walls, robots);

def init_pub_and_sub(my_name):
    publisher = rospy.Publisher('/' + my_name + '/cmd_vel', Twist, queue_size=10);

    robots = ('bernard', 'scrappy', 'oswaldo');
    # subscribers = (
    #     rospy.Subscriber('/' + robots[0] + '/amcl_pose', ),
    #     rospy.Subscriber('/' + robots[0] + '/amcl_pose', ),
    #     rospy.Subscriber('/' + robots[0] + '/', )
    # )

    subscribers = [];

    return publisher, subscribers;

if __name__ == "__main__":
    rospy.init_node('tag');
    rate = rospy.Rate(100);

    DesiredTrajectory = Twist();

    my_name, my_id = get_name_and_id();
    walls, robots  = init_environment(my_name);
    # pub, sub = init_pub_and_sub(my_name);
    pub = rospy.Publisher('/' + my_name + '/cmd_vel', Twist, queue_size=10)

    while not rospy.is_shutdown():

        DesiredTrajectory.linear.x = 1;
        DesiredTrajectory.linear.y = 0;
        DesiredTrajectory.linear.z = 0;

        DesiredTrajectory.angular.x = 0;
        DesiredTrajectory.angular.y = 0;
        DesiredTrajectory.angular.z = 0;

        pub.publish(DesiredTrajectory);
        rate.sleep();
