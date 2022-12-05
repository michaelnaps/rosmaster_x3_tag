function rosmsgOut = Int64(slBusIn, rosmsgOut)
%#codegen
%   Copyright 2021 The MathWorks, Inc.
    rosmsgOut.Data = int64(slBusIn.Data);
end
