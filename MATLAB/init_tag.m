clean;


%% initialize environment variables
robot_names = ["bernard", "scrappy", "oswaldo"];
Nr = length(robot_names);

my_name = "bernard";
center = [0; 0];
radius = 0.15;
tag_radius = 0.15;


%% function declaration
d = @(r1, r2) distance_robot(r1, r2);
