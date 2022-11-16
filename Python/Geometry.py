
import math
import numbers

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.linalg import block_diag


def gca_3d():
    """
    Get current Matplotlib axes, and if they do not support 3-D plotting,
    add new axes that support it
    """
    fig = plt.gcf()
    if len(fig.axes) == 0 or not hasattr(plt.gca(), 'plot3D'):
        axis = fig.add_subplot(111, projection='3d')
    else:
        axis = plt.gca()
    return axis


def numel(var):
    """
    Counts the number of entries in a numpy array, or returns 1 for fundamental
    numerical types
    """
    if isinstance(var, numbers.Number):
        size = int(1)
    elif isinstance(var, np.ndarray):
        size = var.size
    else:
        raise NotImplementedError(f'number of elements for type {type(var)}')
    return size


def rot2d(theta):
    """
    Create a 2-D rotation matrix from the angle  theta according to (1).
    """
    rot_theta = np.array([[math.cos(theta), -math.sin(theta)],
                          [math.sin(theta), math.cos(theta)]])
    return rot_theta


def line_linspace(a_line, b_line, t_min, t_max, nb_points):
    """
    Generates a discrete number of  nb_points points along the curve
    (t)=( a(1)t + b(1), a(2)t + b(2))  R^2 for t ranging from  tMin to  tMax.
    """
    t_sequence = np.linspace(t_min, t_max, nb_points)
    theta_points = a_line * t_sequence + b_line
    return theta_points


def angle(vertex0, vertex1, vertex2, angle_type='unsigned'):
    """
    Compute the angle between two edges  vertex0-- vertex1 and  vertex0--
    vertex2 having an endpoint in common. The angle is computed by starting
    from the edge  vertex0-- vertex1, and then ``walking'' in a
    counterclockwise manner until the edge  vertex0-- vertex2 is found.
    """
    # tolerance to check for coincident points
    tol = 2.22e-16

    # compute vectors corresponding to the two edges, and normalize
    vec1 = vertex1 - vertex0
    vec2 = vertex2 - vertex0

    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 < tol or norm_vec2 < tol:
        # vertex1 or vertex2 coincides with vertex0, abort
        edge_angle = math.nan
        return edge_angle

    vec1 = vec1 / norm_vec1
    vec2 = vec2 / norm_vec2

    # Transform vec1 and vec2 into flat 3-D vectors,
    # so that they can be used with np.inner and np.cross
    vec1flat = np.vstack([vec1, 0]).flatten()
    vec2flat = np.vstack([vec2, 0]).flatten()

    c_angle = np.inner(vec1flat, vec2flat)
    s_angle = np.inner(np.array([0, 0, 1]), np.cross(vec1flat, vec2flat))

    edge_angle = math.atan2(s_angle, c_angle)

    angle_type = angle_type.lower()
    if angle_type == 'signed':
        # nothing to do
        pass
    elif angle_type == 'unsigned':
        edge_angle = (edge_angle + 2 * math.pi) % (2 * math.pi)
    else:
        raise ValueError('Invalid argument angle_type')

    return edge_angle


def clip(val, threshold):
    """
    If val is a scalar, threshold its value; if it is a vector, normalized it
    """
    if isinstance(val, np.ndarray):
        val_norm = np.linalg.norm(val)
        if val_norm > threshold:
            val = val * threshold / val_norm
    elif isinstance(val, numbers.Number):
        if val > threshold:
            val = threshold
        if np.isnan(val):
            val = threshold
    else:
        raise ValueError('Numeric format not recognized')

    return val


class Grid:
    """
    A function to store the coordinates of points on a 2-D grid and evaluate
    arbitrary functions on those points.
    """
    def __init__(self, xx_ticks, yy_ticks):
        """
        Stores the input arguments in attributes.
        """
        self.xx_ticks = xx_ticks
        self.yy_ticks = yy_ticks

    def eval(self, fun):
        """
        This function evaluates the function  fun (which should be a function)
        on each point defined by the grid.
        """

        dim_domain = [numel(self.xx_ticks), numel(self.yy_ticks)]
        dim_range = [numel(fun(np.array([[0], [0]])))]
        fun_eval = np.nan * np.ones(dim_domain + dim_range)
        for idx_x in range(0, dim_domain[0]):
            for idx_y in range(0, dim_domain[1]):
                x_eval = np.array([[self.xx_ticks[idx_x]],
                                   [self.yy_ticks[idx_y]]])
                fun_eval[idx_x, idx_y, :] = np.reshape(fun(x_eval),
                                                       [1, 1, dim_range[0]])

        # If the last dimension is a singleton, remove it
        if dim_range == [1]:
            fun_eval = np.reshape(fun_eval, dim_domain)

        return fun_eval

    def mesh(self):
        """
        Shorhand for calling meshgrid on the points of the grid
        """

        return np.meshgrid(self.xx_ticks, self.yy_ticks)

    def plot_threshold(self, f_handle, threshold=10,
    xrange=[-10,10], yrange=[-10,10]):
        """
        The function evaluates the function f_handle on points placed on the grid.
        """
        def f_handle_clip(val):
            return clip(f_handle(val), threshold)

        f_eval = self.eval(f_handle_clip)
        # print(f_eval)

        [xx_mesh, yy_mesh] = self.mesh()
        f_dim = numel(f_handle_clip(np.zeros((2, 1))))
        if f_dim == 1:
            # scalar field
            fig = plt.gcf()
            axis = fig.add_subplot(111, projection='3d')

            axis.plot_surface(xx_mesh,
                              yy_mesh,
                              f_eval.transpose(),
                              cmap=cm.gnuplot2)
            axis.set_zlim(0, threshold)
        elif f_dim == 2:
            # vector field
            # grid.eval gives the result transposed with respect to
            # what meshgrid expects
            f_eval = f_eval.transpose((1, 0, 2))

            # print(f_eval)
            # vector field
            plt.quiver(xx_mesh,
                       yy_mesh,
                       f_eval[:, :, 0],
                       f_eval[:, :, 1],
                       angles='xy',
                       scale_units='xy')
            axis = plt.gca()
        else:
            raise NotImplementedError(
                'Field plotting for dimension greater than two not implemented'
            )

        axis.set_xlim(xrange[0], xrange[1])
        axis.set_ylim(yrange[0], yrange[1])
        plt.xlabel('x')
        plt.ylabel('y')


class Edge:
    """ Class for storing edges and checking collisions among them. """
    def __init__(self, vertices):
        """
        Save the input coordinates to the internal attribute  vertices.
        """
        self.vertices = vertices

    @property
    def direction(self):
        """ Difference between tip and base """
        return self.vertices[:, [1]] - self.vertices[:, [0]]

    @property
    def base(self):
        """ Coordinates of the first vertex"""
        return self.vertices[:, [0]]

    def plot(self, *args, **kwargs):
        """ Plot the edge """
        plt.plot(self.vertices[0, :], self.vertices[1, :], *args, **kwargs)

    def is_collision(self, edge):
        """
         Returns  True if the two edges intersect.  Note: if the two edges
        overlap but are colinear, or they overlap only at a single endpoint,
        they are not considered as intersecting (i.e., in these cases the
        function returns  False). If one of the two edges has zero length, the
        function should always return the result that edges are
        non-intersecting.
        """

        # Write the lines from the two edges as
        # x_i(t_i)=edge_base+edge.direction*t_i
        # Then finds the parameters for the intersection by solving the linear
        # system obtained from x_1(t_1)=x_2(t_2)

        # Tolerance for cases involving parallel lines and endpoints
        tol = 1e-6

        # The matrix of the linear system
        a_directions = np.hstack([self.direction, -edge.direction])
        if abs(np.linalg.det(a_directions)) < tol:
            # Lines are practically parallel
            return False
        # The vector of the linear system
        b_bases = np.hstack([edge.base - self.base])

        # Solve the linear system
        t_param = np.linalg.solve(a_directions, b_bases)
        t_self = t_param[0, 0]
        t_other = t_param[1, 0]

        # Check that collision point is strictly between endpoints of each edge
        flag_collision = tol < t_self < 1.0 - tol and tol < t_other < 1.0 - tol

        return flag_collision


class Polygon:
    """ Class for plotting, drawing, checking visibility and collision with polygons. """
    def __init__(self, vertices):
        """
        Save the input coordinates to the internal attribute  vertices.
        """
        self.vertices = vertices

    @property
    def nb_vertices(self):
        """ Number of vertices """
        return self.vertices.shape[1]

    @property
    def vertices_loop(self):
        """
        Returns self.vertices with the first vertex repeated at the end
        """
        return np.hstack((self.vertices, self.vertices[:, [0]]))

    def flip(self):
        """
        Reverse the order of the vertices (i.e., transform the polygon from
        filled in to hollow and viceversa).
        """
        self.vertices = np.fliplr(self.vertices)

    def plot(self, color='#9C9C9C', q_color='k'):

        # choose color based on rotation direction
        if self.is_filled():
            inside = color
            outside = '#FCFCFC'
        else:
            inside = '#FCFCFC'
            outside = color

        plt.fill(
            self.vertices[0], self.vertices[1],
            color=inside
        )

        directions = np.diff(self.vertices_loop)
        plt.quiver(
            self.vertices[0, :], self.vertices[1, :],
            directions[0, :],    directions[1, :],
            color=q_color, angles='xy',
            scale_units='xy', scale=1.
        )

    def is_filled(self):
        """
        Checks the ordering of the vertices, and returns whether the polygon is
        filled in or not.
        """
        num_cols = self.vertices.shape[1]
        running_sum = 0

        for i in range(num_cols - 1):
            x_vals = self.vertices[0,:]
            y_vals = self.vertices[1,:]

            # modulus is for the last element to be compared with the first
            # to close the shape
            running_sum += (x_vals[(i+1) % num_cols] - x_vals[i]) * \
                (y_vals[i] + y_vals[(i+1) % num_cols])

        return running_sum < 0

    def is_self_occluded(self, idx_vertex, point):
        vertex = self.vertices[:, [idx_vertex]]
        vertex_next = self.vertices[:, [(idx_vertex + 1) % self.nb_vertices]]
        vertex_prev = self.vertices[:, [(idx_vertex - 1) % self.nb_vertices]]

        angle_p_prev = angle(vertex, point, vertex_prev, 'unsigned')
        angle_p_next = angle(vertex, point, vertex_next, 'unsigned')

        return angle_p_prev < angle_p_next

    def is_visible(self, idx_vertex, test_points):
        """
        Checks whether a point p is visible from a vertex v of a polygon. In
        order to be visible, two conditions need to be satisfied:
         - The point p should not be self-occluded with respect to the vertex
        v\\ (see Polygon.is_self_occluded).
         - The segment p--v should not collide with  any of the edges of the
        polygon (see Edge.is_collision).
        """
        nb_test_points = test_points.shape[1]
        nb_vertices = self.vertices.shape[1]

        # Initial default: all flags are True
        flag_points = [True] * nb_test_points
        vertex = self.vertices[:, [idx_vertex]]
        for idx_point in range(0, nb_test_points):
            point = test_points[:, [idx_point]]

            # If it is self occluded, bail out
            if self.is_self_occluded(idx_vertex, point):
                flag_points[idx_point] = False
            else:
                # Build the vertex-point edge (it is the same for all other
                # edges)
                edge_vertex_point = Edge(np.hstack([point, vertex]))
                # Then iterate over all edges in the polygon
                for idx_vertex_collision in range(0, self.nb_vertices):
                    edge_vertex_vertex = Edge(self.vertices[:, [
                        idx_vertex_collision,
                        (idx_vertex_collision + 1) % nb_vertices
                    ]])
                    # The final result is the and of all the checks with individual edges
                    flag_points[
                        idx_point] &= not edge_vertex_point.is_collision(
                            edge_vertex_vertex)

                    # Early bail out after one collision
                    if not flag_points[idx_point]:
                        break

        return flag_points

    def is_collision(self, test_points):
        """
        Checks whether the a point is in collsion with a polygon (that is,
        inside for a filled in polygon, and outside for a hollow polygon). In
        the context of this homework, this function is best implemented using
        Polygon.is_visible.
        """
        flag_points = [False] * test_points.shape[1]
        # We iterate over the polygon vertices, and process all the test points
        # in parallel
        for idx_vertex in range(0, self.nb_vertices):
            flag_points_vertex = self.is_visible(idx_vertex, test_points)
            # Accumulate the new flags with the previous ones
            flag_points = [
                flag_prev or flag_new
                for flag_prev, flag_new in zip(flag_points, flag_points_vertex)
            ]
        flag_points = [not flag for flag in flag_points]
        return flag_points

    def distance(self, points):
        dist = [];

        for point in points.transpose():
            d_list = [];
            for i_vx, vertex in enumerate(self.vertices.transpose()):
                next_vertex = self.vertices_loop[:,i_vx+1];

                # print("v  =", vertex);
                # print("vn =", next_vertex);

                l_i = np.linalg.norm(point - vertex);
                l_j = np.linalg.norm(point - next_vertex);
                w_ij = np.linalg.norm(vertex.T - next_vertex);

                s_ij = 1/2*(l_i + l_j + w_ij);
                A_ij = np.sqrt(s_ij*(s_ij - l_i)*(s_ij - l_j)*(s_ij - w_ij));

                d_list.append(2*A_ij/w_ij);

            dist.append(d_list);

        return np.array(dist);

    def min_distance(self, point):
        dist = self.distance(point)[0];
        return np.min(dist);

    def distance_grad(self, point, h=1e-3):
        # distances from point
        d = self.distance(point);

        # 2-point central difference method
        x_adj = np.array([[h],[0]]);
        y_adj = np.array([[0],[h]]);

        ptn1_x = point - x_adj;
        ptp1_x = point + x_adj;
        dn1_x = self.distance(ptn1_x)[0];
        dp1_x = self.distance(ptp1_x)[0];

        ptn1_y = point - y_adj;
        ptp1_y = point + y_adj;
        dn1_y = self.distance(ptn1_y)[0];
        dp1_y = self.distance(ptp1_y)[0];

        g = np.array([
            (dp1_x - dn1_x),
            (dp1_y - dn1_y)
        ]) / (2*h*d);

        return g;

    def max_distance_grad(self, point, h=1e-3):
        dist_grad = self.distance_grad(point, h=h);

        i_max = 0;
        for i, g in enumerate(dist_grad.T):
            if np.linalg.norm(g) > np.linalg.norm(dist_grad.T[i_max]):
                i_max = i;

        return dist_grad[:,i_max];



class Sphere:
    """ Class for plotting and computing distances to spheres (circles, in 2-D). """
    def __init__(self, center, radius, distance_influence=0):
        """
        Save the parameters describing the sphere as internal attributes.
        """
        self.center = center
        self.radius = radius
        self.distance_influence = distance_influence

    def plot(self, color='#9C9C9C'):
        """
        This function draws the sphere (i.e., a circle) of the given radius, and the specified color,
        and then draws another circle in gray with radius equal to the distance of influence.
        """
        # Get current axes
        ax = plt.gca()

        # # adjust axes
        plt.axis('equal')

        # Add circle as a patch
        if self.radius > 0:
            # Circle is filled in
            kwargs = {'facecolor': color}
            radius_influence = self.radius + self.distance_influence
        else:
            # Circle is hollow
            kwargs = {'fill': False}
            radius_influence = -self.radius - self.distance_influence

        center = (self.center[0, 0], self.center[1, 0])
        ax.add_patch(
            plt.Circle(center,
                       radius=abs(self.radius),
                       edgecolor=color,
                       **kwargs))

        ax.add_patch(
            plt.Circle(center,
                       radius=abs(self.radius),
                       edgecolor='#9C9C9C',
                       fill=False))

        ax.add_patch(
            plt.Circle(center,
                       radius=radius_influence,
                       edgecolor='k', linestyle='--',
                       fill=False))

    def distance(self, points):
        center = np.array([self.center[0][0], self.center[1][0]])
        points_dist = [];

        # d(x, x_c) = 1/2 * ||x - x_c||^2
        for point in points.transpose():
            dist = np.linalg.norm(point - center)
            dist -= self.radius
            # dist -= self.distance_influence
            points_dist.append(dist)

        return np.array(points_dist)

    def distance_grad(self, points):
        """
        Computes the gradient of the signed distance between points and the
        sphere, consistently with the definition of Sphere.distance.
        """
        center = np.array([self.center[0][0], self.center[1][0]])

        # grad d(x, x_c) = (x - x_c) / ||x - x_c||
        points_grad = []
        points_dist = self.distance(points)
        for i, point in enumerate(points.transpose()):
            if np.abs(points_dist[i]) > 0:
                points_grad.append((point - center) / points_dist[i])
            else:
                points_grad.append([0, 0])
        points_grad = np.array(points_grad)
        return points_grad.transpose()

    def is_collision(self, points):
        points_dist = self.distance(points)
        return [distance < 0 for distance in points_dist]


class Robot:
    def __init__(self, sphere, role, color):
        self.q = np.array([
            [sphere.center[0][0]],
            [0],
            [sphere.center[1][0]],
            [0]
        ]);

        self.r = sphere.radius;
        self.role = role;
        self.color = color;

    @property
    def position(self):
        x = np.array([[self.q[0][0]], [self.q[2][0]]]);
        return x;

    @property
    def velocity(self):
        v = np.array([[self.q[1][0]], [self.q[3][0]]]);
        return v;

    @property
    def sphere(self):
        center = np.array([[self.q[0][0]], [self.q[2][0]]]);
        return Sphere(center, 0.5*self.r, 0.5*self.r);

    def plot(self):
        self.sphere.plot(color=self.color);

    def move(self, dt=0.001, u=None, walls=None, enemy=None):
        m = 1;

        if u is None:
            u = 2*np.random.rand(2,1) - 1;

        self.q = self.q + dt*np.array([
            [self.q[1][0]],
            [u[0][0]/m],
            [self.q[3][0]],
            [u[1][0]/m]
        ]);

    def impact(self, enemy):
        d = np.linalg.norm(
            self.position - enemy.position
        );
        d -= (self.r + enemy.r);
        return d < 0;

    def distance(self, points):
        return self.sphere.distance(points);

    def distance_grad(self, points):
        return self.sphere.distance_grad(points);


class RobotEnvironment:
    def __init__(self, walls, robots):
        if walls.is_filled():
            self.walls = walls.flip();
        else:
            self.walls = walls;

        self.robots = robots;

    def update(self, dt=0.001):

        for robot in self.robots:
            robot.move(dt=dt);

    def plot(self):

        self.walls.plot();

        for robot in self.robots:
            robot.plot();

    def animate(self, Nt, dt=0.001):

        for i in range(Nt):
            t = i*dt;

            plt.clf();

            self.update(dt=dt);
            self.plot();

            plt.show(block=0);
            plt.pause(dt);
