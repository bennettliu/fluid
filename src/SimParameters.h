#ifndef SIMPARAMETERS_H
#define SIMPARAMETERS_H

struct SimParameters
{
    SimParameters()
    {
        timeStep = 0.001;
        NewtonMaxIters = 20;
        NewtonTolerance = 1e-8;
        
        gravityEnabled = true;
        gravityG = 9.8;
        penaltyEnabled = true;
        penaltyStiffness = 1000.0;
        coefficientOfRestitution = 0.9;
    }

    double timeStep;
    double NewtonTolerance;
    int NewtonMaxIters;
    
    bool gravityEnabled;
    double gravityG;
    bool penaltyEnabled;
    double penaltyStiffness;
    double coefficientOfRestitution;
};

#endif