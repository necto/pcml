function [optimalKernelScale, optimalBoxConstraint, optimalBias] = ...
    OptimalSVMParams( positiveClass, useNegs )
%OPTIMALSVMPARAMS Summary of this function goes here
%   Detailed explanation goes here
    if (positiveClass == 1)
        optimalKernelScale = 115.4782;
        optimalBoxConstraint = 2.3713;
        optimalBias = 3.1623;
    elseif (positiveClass == 2)
        optimalKernelScale = 84.8343;
        optimalBoxConstraint = 1.3895;
        optimalBias = 4.5409;
    elseif (positiveClass == 3)
        optimalKernelScale = 100;
        optimalBoxConstraint = 2.6367;
        if (useNegs)
            optimalBias = 19.9526;
        else
            optimalBias = 7.017;
        end
    else
        fprintf('wrong positive class: %d\n', positiveClass);
        optimalKernelScale = 0;
        optimalBoxConstraint = 0;
        optimalBias = 0;
    end
end

