# primary testing file for algorithm development

import numbers
import numpy as np
import matplotlib.pyplot as plt
import Geometry as gm


def init_environment(wall_type):

    if wall_type == 'polygon':
        bounds = np.array([
            [-3, 3, 3, -3],
            [3, 3, -3, -3]
        ]);
        walls = gm.Polygon(bounds);
        wall_gain = 1;

    elif wall_type =='sphere':
        env_center = np.array([[0.],[0.]]);
        env_radius = -3;
        walls = gm.Sphere(env_center, env_radius);
        wall_gain = 1;

    robot_radius = 0.15;  # safety radius
    tag_radius = 0.15;
    pursuer_gain = 1;

    b_c = np.array([[2],[-2]]);
    sphere_temp = gm.Sphere(b_c, robot_radius, tag_radius);
    bernard = gm.Robot(sphere_temp, 'pursuer', 'yellowgreen', 'bernard');

    s_c = np.array([[-1], [1]]);
    sphere_temp = gm.Sphere(s_c, robot_radius, tag_radius);
    scrappy = gm.Robot(sphere_temp, 'evader', 'firebrick', 'scrappy');

    o_c = np.array([[-2], [2]]);
    sphere_temp = gm.Sphere(o_c, robot_radius, tag_radius);
    oswaldo = gm.Robot(sphere_temp, 'evader', 'mediumpurple', 'oswaldo');

    robots = (bernard, scrappy, oswaldo);
    world = gm.RobotEnvironment(walls, robots, wall_gain, pursuer_gain);

    return world, robots;


if __name__ == "__main__":

    polyworld, robots = init_environment('polygon');

    sphereworld, _ = init_environment('sphere');

    point = np.array([[1.5], [1.5]]);
    # print(polyworld.distance_grad(point));
    # print(sphereworld.distance_grad(point));

    x_ticks = np.linspace(-3, 3, 60);
    y_ticks = np.linspace(-3, 3, 60);
    grid_var = gm.Grid(x_ticks, y_ticks);

    threshold = 10;
    xrange = [-3.25, 3.25];
    yrange = xrange;

    ans1 = input("See threshold plots? [y/n] ");
    if ans1 == 'y':
        polyworld.plot();
        grid_var.plot_threshold(polyworld.distance_grad, threshold, xrange, yrange);
        plt.show();

        sphereworld.plot();
        grid_var.plot_threshold(sphereworld.distance_grad, threshold, xrange, yrange);
        plt.show();

    ans2 = input("See animation? [y/n] ");
    if ans2 == 'y':
        N = 1000;
        alpha = 0.025;

        ans3 = input("Which environment? [p/s] ")
        if ans3 == 'p':
            polyworld.animate(N, alpha);
        elif ans3 =='s':
            sphereworld.animate(N, alpha);

        print("Animation complete.");
