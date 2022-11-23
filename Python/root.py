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
    bernard = gm.Robot(sphere_temp, 'pursuer', 'yellowgreen', 'bernard');

    s_c = np.array([[0.5], [0.5]]);
    sphere_temp = gm.Sphere(s_c, robot_radius);
    scrappy = gm.Robot(sphere_temp, 'evader', 'firebrick', 'scrappy');

    o_c = np.array([[0.5], [-0.5]]);
    sphere_temp = gm.Sphere(o_c, robot_radius);
    oswaldo = gm.Robot(sphere_temp, 'evader', 'mediumpurple', 'oswaldo');

    robots = (bernard, scrappy, oswaldo);
    world = gm.RobotEnvironment(walls, robots);

    return world, robots;


if __name__ == "__main__":

    world, robots = init_environment();

    point = np.array([[1], [0.25]]);
    # print(world.walls.total_distance_grad(point));

    x_ticks = 2*np.linspace(-2, 2, 60);
    y_ticks = 2*np.linspace(-1, 1, 60);
    grid_var = gm.Grid(x_ticks, y_ticks);

    # threshold = 5;
    # xrange = [-4.5, 4.5];
    # yrange = [-1.5, 1.5];
    # world.plot();
    # grid_var.plot_threshold(world.distance_grad, threshold, xrange, yrange)
    # plt.show()

    world.animate(1000, 0.01);
