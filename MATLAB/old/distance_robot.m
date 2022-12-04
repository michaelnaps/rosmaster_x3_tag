function [dist] = distance_robot(robot1, robot2)
    dist = norm(robot1.x - robot2.x);
end