#! /usr/bin/env python3

import os

import sys
import rospy
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseWithCovarianceStamped

import numpy as np


rospy.init_node('tag');

pub = rospy.Publisher('initialpose', PoseWithCovarianceStamped, queue_size=10);

InitialPose = PoseWithCovarianceStamped();

InitialPose.header.stamp=rospy.get_time();
InitialPose.header.frame_id="map";

InitialPose.pose.covariance = np.ravel(np.eye(6));
InitialPose.pose.pose = Pose(
    position=Point(x=0, y=0, z=0),
    orientation=Quaternion(x=0, y=0, z=0, w=0)
);

pub.publish(InitialPose);
