function [optimalKernelScale, optimalBoxConstraint, optimalBias] = ...
    OptimalSVMParams( positiveClass, useNegs )

    if (positiveClass == 1)
        optimalKernelScale = 115.4782;
        optimalBoxConstraint = 2.3713;
        if (useNegs)
            optimalBias = 28.84;
        else
            optimalBias = 3.1623;
        end
    elseif (positiveClass == 2)
        optimalKernelScale = 84.8343;
        optimalBoxConstraint = 1.3895;
        if (useNegs)
            optimalBias = 10;
        else
            optimalBias = 4.5409;
        end
    elseif (positiveClass == 3)
        optimalKernelScale = 100;
        optimalBoxConstraint = 2.6367;
        if (useNegs)
            optimalBias = 15.8489;
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

