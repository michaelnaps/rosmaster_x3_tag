
%% initialize environment variables
robot_names = ["bernard", "scrappy", "oswaldo"];
Nr = length(robot_names);

my_name = "bernard";
center = [0; 0];
radius = 0.15;
tag_radius = 0.44;

%A = [0.69,-0.57];
%B = [0.50, 0.50];
%C = [-0.5,-0.50];


%% function declaration
d = @(r1, r2) distance_robot(r1, r2);
