
%this function takes the structure output from /name/amcl_pose and reduces
%the data to a 2x1 vector of position data. 
function [out] = poseOut(data)



%data.bernard_pose.Pose.Pose.Position.X.Data


if (data.scrappy_pose.Pose.Pose.Position.X)
    
    out = data.scrappy_pose.Pose.Pose.Position.X.data(end);

elseif (data.oswaldo_pose.Pose.Pose.Position.X)
    
     out = data.oswaldo_pose.Pose.Pose.Position.X.data(end);

elseif (data.bernard_pose.Pose.Pose.Position.X)

     out = data.bernard_pose.Pose.Pose.Position.X.data(end);

end


