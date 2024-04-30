#include "RigidBodyTemplate.h"
#include <iostream>
#include <igl/readOBJ.h>
#include <igl/signed_distance.h>
#include <Eigen/Dense>
#include <fstream>
#include <map>
#include <Eigen/Sparse>

using namespace std;
using namespace Eigen;

RigidBodyTemplate::RigidBodyTemplate(const std::string &meshFilename, double scale) : volume_(0), radius_(0)
{
    inertiaTensor_.setZero();

    igl::readOBJ(meshFilename, V, F);

    V *= scale;

    initialize();
}

RigidBodyTemplate::~RigidBodyTemplate()
{    
}

void RigidBodyTemplate::initialize()
{
    computeVolume();
    Vector3d cm = computeCenterOfMass();
    for(int i=0; i<V.rows(); i++)
        V.row(i) -= cm;

    radius_ = 0;
    for(int i=0; i<V.rows(); i++)
    {
        radius_ = max(radius_, V.row(i).norm());
    }
    computeInertiaTensor();    
}


void RigidBodyTemplate::computeVolume()
{
    volume_ = 0;
    for (int i = 0; i < F.rows(); i++)
    {
        Vector3d pts[3];
        Vector3d centroid(0, 0, 0);
        for (int j = 0; j < 3; j++)
        {
            pts[j] = V.row(F(i, j));
            centroid += pts[j];
        }
        Vector3d normal = (pts[1] - pts[0]).cross(pts[2] - pts[0]);
        double area = 0.5 * normal.norm();
        normal /= normal.norm();

        centroid /= 3.0;
        volume_ += centroid.dot(normal) * area / 3.0;
    }
}

Vector3d RigidBodyTemplate::computeCenterOfMass()
{
    Vector3d cm(0, 0, 0);
    for (int i = 0; i < F.rows(); i++)
    {
        Vector3d pts[3];
        for (int j = 0; j < 3; j++)
        {
            pts[j] = V.row(F(i, j));
        }
        Vector3d normal = (pts[1] - pts[0]).cross(pts[2] - pts[0]);
        double area = 0.5 * normal.norm();
        normal /= normal.norm();

        Vector3d term(0, 0, 0);
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                for (int l = k; l < 3; l++)
                    term[j] += pts[k][j] * pts[l][j];
            }
            term[j] *= area*normal[j] / 12.0;
        }

        cm += term;
    }

    return cm / volume_;
}

void RigidBodyTemplate::computeInertiaTensor()
{
    Vector3d quads(0, 0, 0);
    Vector3d mixed(0, 0, 0);
    for (int i = 0; i < F.rows(); i++)
    {
        Vector3d pts[3];
        for (int j = 0; j < 3; j++)
        {
            pts[j] = V.row(F(i, j));
        }
        Vector3d normal = (pts[1] - pts[0]).cross(pts[2] - pts[0]);
        double area = 0.5 * normal.norm();
        normal /= normal.norm();


        Vector3d term(0, 0, 0);
        Vector3d mixterm(0, 0, 0);
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                for (int l = k; l < 3; l++)
                {
                    for (int m = l; m < 3; m++)
                        term[j] += pts[k][j] * pts[l][j] * pts[m][j];
                }
            }
            term[j] *= area*normal[j] / 30.0;
        }
        double mix = 0;
        for (int j = 0; j < 3; j++)
        {
            mix += 6.0*pts[j][0] * pts[j][1] * pts[j][2];
            for (int k = 0; k < 3; k++)
            {
                mix += 2.0*pts[j][k] * pts[j][(k + 1) % 3] * pts[(j + 1) % 3][(k + 2) % 3];
                mix += 2.0*pts[j][k] * pts[j][(k + 1) % 3] * pts[(j + 2) % 3][(k + 2) % 3];
            }
            mix += pts[j][0] * pts[(j + 1) % 3][1] * pts[(j + 2) % 3][2];
            mix += pts[j][2] * pts[(j + 1) % 3][1] * pts[(j + 2) % 3][0];
        }
        for (int j = 0; j < 3; j++)
            mixterm[j] = mix*area*normal[j] / 60.0;

        quads += term;
        mixed += mixterm;
    }

    inertiaTensor_ << quads[1] + quads[2], -mixed[2], -mixed[1],
        -mixed[2], quads[0] + quads[2], -mixed[0],
        -mixed[1], -mixed[0], quads[0] + quads[1];
}