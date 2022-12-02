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


if __name__ == "__main__":
    rospy.init_node('tag');
    rate = rospy.Rate(100);

    DesiredTrajectory = Twist();

    my_name, my_id = get_name_and_id();

    pub = rospy.Publisher('/' + my_name + '/cmd_vel', Twist, queue_size=10)

    while not rospy.is_shutdown():
        DesiredTrajectory.linear.x = 1;
        DesiredTrajectory.linear.y = 0;
        DesiredTrajectory.linear.z = 0;

        DesiredTrajectory.angular.x = 0;
        DesiredTrajectory.angular.y = 0;
        DesiredTrajectory.angular.z = 0;

        print(DesiredTrajectory);

        pub.publish(DesiredTrajectory);
        rate.sleep();
