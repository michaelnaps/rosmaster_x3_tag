# primary testing file for algorithm development

import numbers
import numpy as np
import matplotlib.pyplot as plt
import Geometry as gm


def init_environment():

    bounds = 2*np.array([
        [-2, 2, 2, -2],
        [1, 1, -1, -1]
    ]);
    walls = gm.Polygon(bounds);

    robot_radius = 0.30;  # safety radius
    b_c = np.array([[0],[0]]);
    sphere_temp = gm.Sphere(b_c, robot_radius);
    bernard = gm.Robot(sphere_temp, 'pursuer', 'yellowgreen');

    s_c = np.array([[0.5], [0.5]]);
    sphere_temp = gm.Sphere(s_c, robot_radius);
    scrappy = gm.Robot(sphere_temp, 'evader', 'firebrick');

    o_c = np.array([[0.5], [-0.5]]);
    sphere_temp = gm.Sphere(o_c, robot_radius);
    oswaldo = gm.Robot(sphere_temp, 'evader', 'mediumpurple');

    robots = (bernard, scrappy, oswaldo);
    world = gm.RobotEnvironment(walls, robots);

    return world, robots[0];


if __name__ == "__main__":

    world, robots = init_environment();

    # x_ticks = np.linspace(-3, 3, 100);
    # y_ticks = np.linspace(-2, 2, 100);
    # grid_var = gm.Grid(x_ticks, y_ticks);

    # threshold = 5;
    # xrange = [-3, 3];
    # yrange = [-2, 2];
    # grid_var.plot_threshold(world.walls.min_distance, threshold, xrange, yrange)
    # plt.show()
