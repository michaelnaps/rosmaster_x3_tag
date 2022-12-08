# primary testing file for algorithm development

import numbers
import numpy as np
import matplotlib.pyplot as plt
import Geometry as gm


def init_environment(wall_type):
    bound_w = 2.75;
    bound_h = 2.10;

    if wall_type == 'polygon':
        # establish environment
        bounds = np.array([
            [-bound_w/2, bound_w/2, bound_w/2, -bound_w/2],
            [bound_h/2, bound_h/2, -bound_h/2, -bound_h/2]
        ]);

        walls = (gm.Polygon(bounds), );
        wall_gain = 1;

    elif wall_type =='sphere':
        # establish environment
        env_center = np.array([[0.],[0.]]);
        env_radius = -3;
        walls = (gm.Sphere(env_center, env_radius), );
        wall_gain = 1;

    elif wall_type == 'all':
        # establish environment
        bounds = np.array([
            [-bound_w/2, bound_w/2, bound_w/2, -bound_w/2],
            [bound_h/2, bound_h/2, -bound_h/2, -bound_h/2]
        ]);

        env_center = np.array([[0],[0]])
        env_radius = -np.sqrt((bound_w/2)**2 + (bound_h/2)**2);

        wall_gain = 1.5;
        walls = (gm.Polygon(bounds), gm.Sphere(env_center, env_radius));

    # establish robot variables
    robot_radius = 0.11;
    tag_radius = 0.11;
    pursuer_gain = 1;

    b_c = np.array([[0.6],[0.55]]);
    sphere_temp = gm.Sphere(b_c, robot_radius, tag_radius);
    bernard = gm.Robot(sphere_temp, 'pursuer', 'yellowgreen', 'bernard');

    s_c = np.array([[-0.3],[0.5]]);
    sphere_temp = gm.Sphere(s_c, robot_radius, tag_radius);
    scrappy = gm.Robot(sphere_temp, 'evader', 'firebrick', 'scrappy');

    o_c = np.array([[-0.2],[-0.25]]);
    sphere_temp = gm.Sphere(o_c, robot_radius, tag_radius);
    oswaldo = gm.Robot(sphere_temp, 'evader', 'mediumpurple', 'oswaldo');

    robots = (bernard, scrappy, oswaldo);

    world = gm.RobotEnvironment(walls, robots);

    return world, robots;


if __name__ == "__main__":

    polyworld, robots = init_environment('polygon');
    sphereworld, _ = init_environment('sphere');
    allworld, _ = init_environment('all');

    # point = np.array([[1.5], [1.5]]);
    # print(polyworld.distance_grad(point));
    # print(sphereworld.distance_grad(point));

    bound_w = 2.75;
    bound_h = 3.25;

    x_ticks = np.linspace(-bound_w, bound_w, 60);
    y_ticks = np.linspace(-bound_h, bound_h, 60);
    grid_var = gm.Grid(x_ticks, y_ticks);

    threshold = 10;
    xrange = [-bound_w, bound_w];
    yrange = [-bound_h, bound_h];

    # robots[0].control(0.025, allworld.walls, robots)

    ans11 = input("See threshold plots? [y/n] ");
    if ans11 == 'y':

        ans12 = input("Which environment? [p/s/a] ")
        if ans12 == 'p':
            polyworld.plot();
            grid_var.plot_threshold(polyworld.distance_grad, threshold, xrange, yrange);
            plt.show();

        if ans12 == 's':
            sphereworld.plot();
            grid_var.plot_threshold(sphereworld.distance_grad, threshold, xrange, yrange);
            plt.show();

        if ans12 == 'a':
            allworld.plot();
            grid_var.plot_threshold(allworld.distance_grad, threshold, xrange, yrange);
            plt.show();

    ans21 = input("See animation? [y/n] ");
    if ans21 == 'y':
        N = 1000;
        alpha = 0.025;

        ans22 = input("Which environment? [p/s/a] ")
        if ans22 == 'p':
            polyworld.animate(N, alpha);
        elif ans22 =='s':
            sphereworld.animate(N, alpha);
        elif ans22 =='a':
            allworld.animate(N, alpha);

        print("Animation complete.");
