#ifndef RIGIDBODYTEMPLATE_H
#define RIGIDBODYTEMPLATE_H

#include <string>
#include <Eigen/Core>
#include <set>

class SignedDistanceField;

constexpr int gridSize = 100;

class RigidBodyTemplate
{
public:
    RigidBodyTemplate(const std::string &meshFilename, double scale);
    ~RigidBodyTemplate();

    double getVolume() const {return volume_;}
    const Eigen::Matrix3d getInertiaTensor() const {return inertiaTensor_;}    

    double getBoundingRadius() const {return radius_;}
    const Eigen::MatrixX3d &getVerts() const {return V;}
    const Eigen::MatrixX3i &getFaces() const {return F;}       

private:
    RigidBodyTemplate(const RigidBodyTemplate &other) = delete;
    RigidBodyTemplate &operator=(const RigidBodyTemplate &other) = delete;

    void initialize();

    void computeVolume();
    Eigen::Vector3d computeCenterOfMass();
    void computeInertiaTensor();

    Eigen::MatrixX3d V;
    Eigen::MatrixX3i F;
    
    double volume_;
    double radius_;
    Eigen::Matrix3d inertiaTensor_;    
};

#endif // RIGIDBODYTEMPLATE_H
