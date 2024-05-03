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
        particleAdditionMode = false;
        particleDragMode = false;
        example = 2;
        interp = 0.1;
        iters = 10;
        dispersalForce = 2.0;
        density = 31.25;
    }

    double timeStep;
    double NewtonTolerance;
    int NewtonMaxIters;
    
    bool gravityEnabled;
    double gravityG;
    bool penaltyEnabled;
    double penaltyStiffness;
    double coefficientOfRestitution;
    bool particleAdditionMode;
    bool particleDragMode;
    int example;
    float interp;
    int iters;
    float dispersalForce;
    float density;
};

#endif