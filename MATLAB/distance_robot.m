function [dist] = distance_robot(robot1,robot2)
    dist = norm(robot1 - robot2);
end