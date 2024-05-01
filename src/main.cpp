#include "polyscope/polyscope.h"

#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"

#include <iostream>
#include <unordered_set>
#include <utility>
#include <deque>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "SimParameters.h"
#include "RigidBodyInstance.h"
#include "RigidBodyTemplate.h"
#include "VectorMath.h"
#include <fstream>
#include "misc/cpp/imgui_stdlib.h"

struct Vec3Hash {
    template <typename T>
    size_t operator()(const Eigen::Vector3<T>& v) const {
        auto hash1 = std::hash<T>{}(v[0]);
        auto hash2 = std::hash<T>{}(v[1]);
        auto hash3 = std::hash<T>{}(v[2]);
        return hash1 ^ hash2 ^ hash3;
    }
};

bool running_;
double time_;
SimParameters params_;
std::string sceneFile_;
bool launch_;

std::vector<RigidBodyTemplate*> templates_;
std::vector<RigidBodyInstance*> bodies_;
std::vector<Eigen::Vector3d> points;
std::vector<Eigen::Vector3d> vel;

Eigen::MatrixXd groundV;
Eigen::MatrixXi groundF;
double edgeLen = 10.0;
double voxelLen = 2.0;

void updateRenderGeometry()
{           
    double groundSize = edgeLen / 2.0;

    groundV.resize(5, 3);
    groundV << 0, -groundSize, 0,
        -groundSize, -groundSize, -groundSize,
        -groundSize, -groundSize, groundSize,
        groundSize, -groundSize, groundSize,
        groundSize, -groundSize, -groundSize;
    groundF.resize(4, 3);
    groundF << 0, 1, 2,
        0, 2, 3,
        0, 3, 4,
        0, 4, 1;

}

void loadScene()
{
    for (RigidBodyInstance* rbi : bodies_)
        delete rbi;
    for (RigidBodyTemplate* rbt : templates_)
        delete rbt;
    bodies_.clear();
    templates_.clear();

    points.clear();
    // generate points
    for (size_t i = 0; i < 100; i++) {
        Eigen::Vector3d p = {polyscope::randomUnit() * (edgeLen / 2.0), 
                    polyscope::randomUnit() * edgeLen - (edgeLen / 2.0), 
                    polyscope::randomUnit() * edgeLen - (edgeLen / 2.0)};
        Eigen::Vector3d v = {0, 0, 0};
        points.push_back(p);
        vel.push_back(v);
    }
}

void initSimulation()
{
    time_ = 0;
    loadScene();
}

void makegrid(int ind, std::unordered_map<Eigen::Vector3i, double, Vec3Hash> &grid) {
    std::vector<int> permutation = std::vector<int>();
    permutation.push_back(ind);
    for (int i = 0; i < 3; i++) {
        if (i == ind) continue;
        permutation.push_back(i);
    }
    int n = points.size();
    std::unordered_map<Eigen::Vector3i, double, Vec3Hash> localgrid;
    std::unordered_map<Eigen::Vector3i, double, Vec3Hash> localweight;
    for (int i = 0; i < n; i++) {
        Eigen::Vector3d localq;
        Eigen::Vector3d localqdot;
        Eigen::Vector3d localCorner;
        Eigen::Vector3i localCorneri;
        for (int j = 0; j < 3; j++) {
            localq[j] = points[i][permutation[j]];
            localqdot[j] = vel[i][permutation[j]];
            if (j == 0) {
                localCorneri[j] = (int) ((localq[j] + (edgeLen / 2.0)) / voxelLen);
                localCorner[j] = localCorneri[j] * voxelLen - (edgeLen / 2.0);
            }
            else {
                localCorneri[j] = ((int) ((localq[j] + (edgeLen / 2.0) + (voxelLen / 2.0)) / voxelLen)) - 1;
                localCorner[j] = localCorneri[j] * voxelLen - (edgeLen / 2.0) + (voxelLen / 2.0);
            }
        }
        Eigen::Vector3i newCorneri;
        Eigen::Vector3i newCorner;
        for (int ax1 = 0; ax1 <= 1; ax1++) {
            newCorneri[0] = localCorneri[0] + ax1;
            newCorner[0] = localCorner[0] + ((double) ax1) * voxelLen;
            for (int ax2 = 0; ax2 <= 1; ax2++) {
                newCorneri[1] = localCorneri[1] + ax2;
                newCorner[1] = localCorner[1] + ((double) ax2) * voxelLen;
                for (int ax3 = 0; ax3 <= 1; ax3++) {
                    newCorneri[2] = localCorneri[2] + ax3;
                    newCorner[2] = localCorner[2] + ((double) ax3) * voxelLen;
                    // compute contribution to corner (ax1, ax2, ax3)
                    double weight = 1.0;
                    for (int j = 0; j < 3; j++) {
                        weight *= voxelLen - std::fabs(newCorner[j] - localq[j]);
                    }
                    // change grid values
                    if (localgrid.count(newCorneri) == 0) {
                        localgrid[newCorneri] = 0;
                        localweight[newCorneri] = 0;
                    }
                    localgrid[newCorneri] += weight * vel[i][ind];
                    localweight[newCorneri] += weight;
                }
            }
        }
    }
    for (auto iter = localgrid.begin(); iter != localgrid.end(); iter++) {
        Eigen::Vector3i localCorneri = iter->first;
        Eigen::Vector3i globalCorneri;
        for (int i = 0; i < 3; i++) {
            globalCorneri[permutation[i]] = localCorneri[i];
        }
        grid[globalCorneri] = localgrid[localCorneri] / localweight[localCorneri];
    }
}

void incompressibility(std::unordered_map<Eigen::Vector3i, double, Vec3Hash> &u, std::unordered_map<Eigen::Vector3i, double, Vec3Hash> &v, std::unordered_map<Eigen::Vector3i, double, Vec3Hash> &w) {
    int n = points.size();
    int minCoord = 0;
    int maxCoord = std::ceil(edgeLen / voxelLen);

    // Get set of cells
    std::unordered_set<Eigen::Vector3i, Vec3Hash> waterCells = {};
    for (int i = 0; i < n; i++) {
        Eigen::Vector3i pt;
        for (int j = 0; j < 3; j++) {
            pt[j] = (int) ((points[i][j] + (edgeLen / 2.0)) / voxelLen);
        }
        waterCells.insert(pt);
    }
    // Initialize all gridpoints
    for (auto iter = waterCells.begin(); iter != waterCells.end(); iter++) {
        Eigen::Vector3i pt = *iter;
        if (u.count(pt) == 0) u[pt] = 0;
        if (v.count(pt) == 0) v[pt] = 0;
        if (w.count(pt) == 0) w[pt] = 0;
        pt[0]++;
        if (u.count(pt) == 0) u[pt] = 0;
        pt[0]--;
        pt[1]++;
        if (v.count(pt) == 0) v[pt] = 0;
        pt[1]--;
        pt[2]++;
        if (w.count(pt) == 0) w[pt] = 0;
    }
    // Apply gravity
    for (auto iter = v.begin(); iter != v.end(); iter++) {
        Eigen::Vector3i pt = iter->first;
        v[iter->first] -= params_.gravityG;
    }

    // Set all walls to zero
    for (auto iter = u.begin(); iter != u.end(); iter++) {
        int ind = (iter->first)[0];
        if (ind == minCoord || ind == maxCoord) u[iter->first] = 0;
    }
    for (auto iter = v.begin(); iter != v.end(); iter++) {
        int ind = (iter->first)[1];
        if (ind == minCoord || ind == maxCoord) v[iter->first] = 0;
    }
    for (auto iter = w.begin(); iter != w.end(); iter++) {
        int ind = (iter->first)[2];
        if (ind == minCoord || ind == maxCoord) w[iter->first] = 0;
    }
    // Repeatedly enforce incompressibility
    // TODO: Parameterize
    for (int i = 0; i < 100; i++) {
        for (auto iter = waterCells.begin(); iter != waterCells.end(); iter++) {
            Eigen::Vector3i pt = *iter;
            double d = 0;
            double s = 0;
            // Sum up d and s
            if (pt[0] != minCoord && pt[0] != maxCoord) {
                d -= u[pt];
                s++;
            }
            if (pt[1] != minCoord && pt[1] != maxCoord) {
                d -= v[pt];
                s++;
            }
            if (pt[2] != minCoord && pt[2] != maxCoord) {
                d -= w[pt];
                s++;
            }
            pt[0]++;
            if (pt[0] != minCoord && pt[0] != maxCoord) {
                d += u[pt];
                s++;
            }
            pt[0]--;
            pt[1]++;
            if (pt[1] != minCoord && pt[1] != maxCoord) {
                d += v[pt];
                s++;
            }
            pt[1]--;
            pt[2]++;
            if (pt[2] != minCoord && pt[2] != maxCoord) {
                d += w[pt];
                s++;
            }
            pt[2]--;
            // Edit d and s
            if (pt[0] != minCoord && pt[0] != maxCoord) {
                u[pt] += d / s;
            }
            if (pt[1] != minCoord && pt[1] != maxCoord) {
                v[pt] += d / s;
            }
            if (pt[2] != minCoord && pt[2] != maxCoord) {
                w[pt] += d / s;
            }
            pt[0]++;
            if (pt[0] != minCoord && pt[0] != maxCoord) {
                u[pt] -= d / s;
            }
            pt[0]--;
            pt[1]++;
            if (pt[1] != minCoord && pt[1] != maxCoord) {
                v[pt] -= d / s;
            }
            pt[1]--;
            pt[2]++;
            if (pt[2] != minCoord && pt[2] != maxCoord) {
                w[pt] -= d / s;
            }
            pt[2]--;
        }
    }
}

void fromgrid(int ind, std::unordered_map<Eigen::Vector3i, double, Vec3Hash> grid) {
    std::vector<int> permutation = std::vector<int>();
    permutation.push_back(ind);
    for (int i = 0; i < 3; i++) {
        if (i == ind) continue;
        permutation.push_back(i);
    }
    int n = points.size();
    for (int i = 0; i < n; i++) {
        Eigen::Vector3d localq;
        Eigen::Vector3d localqdot;
        Eigen::Vector3d localCorner;
        Eigen::Vector3i localCorneri;
        for (int j = 0; j < 3; j++) {
            localq[j] = points[i][permutation[j]];
            localqdot[j] = vel[i][permutation[j]];
            if (j == 0) {
                localCorneri[j] = (int) ((localq[j] + (edgeLen / 2.0)) / voxelLen);
                localCorner[j] = localCorneri[j] * voxelLen - (edgeLen / 2.0);
            }
            else {
                localCorneri[j] = ((int) ((localq[j] + (edgeLen / 2.0) + (voxelLen / 2.0)) / voxelLen)) - 1;
                localCorner[j] = localCorneri[j] * voxelLen - (edgeLen / 2.0) + (voxelLen / 2.0);
            }
        }
        Eigen::Vector3i newCorneri;
        Eigen::Vector3i newCorner;
        double velocity = 0;
        double weightSum = 0;
        for (int ax1 = 0; ax1 <= 1; ax1++) {
            newCorneri[0] = localCorneri[0] + ax1;
            newCorner[0] = localCorner[0] + ((double) ax1) * voxelLen;
            for (int ax2 = 0; ax2 <= 1; ax2++) {
                newCorneri[1] = localCorneri[1] + ax2;
                newCorner[1] = localCorner[1] + ((double) ax2) * voxelLen;
                for (int ax3 = 0; ax3 <= 1; ax3++) {
                    newCorneri[2] = localCorneri[2] + ax3;
                    newCorner[2] = localCorner[2] + ((double) ax3) * voxelLen;
                    // compute contribution to corner (ax1, ax2, ax3)
                    double weight = 1.0;
                    for (int j = 0; j < 3; j++) {
                        weight *= voxelLen - std::fabs(newCorner[j] - localq[j]);
                    }
                    Eigen::Vector3i globalCorneri;
                    for (int j = 0; j < 3; j++) {
                        globalCorneri[permutation[j]] = newCorneri[j];
                    }
                    velocity += weight * grid[globalCorneri];
                    weightSum += weight;
                }
            }
        }
        vel[i][ind] = vel[i][ind] * 0.5 + (velocity / weightSum) * 0.5;
    }
}

void simulateOneStep()
{
    double h = params_.timeStep;
    time_ += h;
    int n = points.size();

    // point vel -> grid vel
    std::unordered_map<Eigen::Vector3i, double, Vec3Hash> u, v, w;
    makegrid(0, u);
    makegrid(1, v);
    makegrid(2, w);

    // enforce incompressibility
    incompressibility(u, v, w);

    // grid vel -> point vel
    fromgrid(0, u);
    fromgrid(1, v);
    fromgrid(2, w);

    // update points
    for (int i = 0; i < n; i++) {
        points[i] += vel[i] * h;
    }
}

void callback()
{
    ImGui::SetNextWindowSize(ImVec2(500., 0.));
    ImGui::Begin("UI", nullptr);

    if (ImGui::Button("Recenter Camera", ImVec2(-1, 0)))
    {
        polyscope::view::resetCameraToHomeView();
    }

    if (ImGui::CollapsingHeader("Simulation Control", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (ImGui::Button("Run/Pause Sim", ImVec2(-1, 0)))
        {
            running_ = !running_;
        }
        if (ImGui::Button("Reset Sim", ImVec2(-1, 0)))
        {
            running_ = false;
            initSimulation();
        }        
    }
    if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::InputText("Filename", &sceneFile_);
        if (ImGui::Button("Load Scene", ImVec2(-1, 0)))
        {
            loadScene();
            initSimulation();
        }
    }
    if (ImGui::CollapsingHeader("Simulation Options", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::InputDouble("Timestep", &params_.timeStep);
        ImGui::InputDouble("Newton Tolerance", &params_.NewtonTolerance);
        ImGui::InputInt("Newton Max Iters", &params_.NewtonMaxIters);
    }
    if (ImGui::CollapsingHeader("Forces", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Checkbox("Gravity Enabled", &params_.gravityEnabled);
        ImGui::InputDouble("Gravity G", &params_.gravityG);
        ImGui::Checkbox("Penalty Forces Enabled", &params_.penaltyEnabled);
        ImGui::InputDouble("Penalty Stiffness", &params_.penaltyStiffness);
        ImGui::InputDouble("Coefficient of Restitution", &params_.coefficientOfRestitution);
    }    

    if (ImGui::IsKeyPressed(ImGui::GetKeyIndex(ImGuiKey_Space)))
    {
        launch_ = true;
    }

    ImGui::End();
}

int main(int argc, char **argv) 
{
  polyscope::view::setWindowSize(1600, 800);
  polyscope::options::buildGui = false;
  polyscope::options::openImGuiWindowForUserCallback = false;
  polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;

  polyscope::options::autocenterStructures = false;
  polyscope::options::autoscaleStructures = false;
  polyscope::options::maxFPS = -1;

  sceneFile_ = "box.scn";
  launch_ = false;

  initSimulation();

  polyscope::init();

  polyscope::state::userCallback = callback;

  while (!polyscope::render::engine->windowRequestsClose())
  {
      if (running_)
          simulateOneStep();
      updateRenderGeometry();

      if (launch_)
      {
          double launchVel = 100;
          Eigen::Vector3d launchPos;
          for (int i = 0; i < 3; i++)
              launchPos[i] = polyscope::view::getCameraWorldPosition()[i];
          
          Eigen::Vector3d launchDir;
          glm::vec3 look;
          glm::vec3 dummy;
          polyscope::view::getCameraFrame(look, dummy, dummy);
          for (int i = 0; i < 3; i++)
              launchDir[i] = look[i];

            // Launch a bird
            RigidBodyTemplate* rbt = new RigidBodyTemplate("../meshes/bird2.obj", 0.5);
            double rho = 1.0;
            Eigen::Vector3d theta = {0, -3.14 / 2.0, 0};
            Eigen::Vector3d w = {100, 0, 0};
            w.setZero();
            RigidBodyInstance* rbi = new RigidBodyInstance(*rbt, launchPos, launchDir, launchVel * launchDir, w, rho);
            templates_.push_back(rbt);
            bodies_.push_back(rbi);

          launch_ = false;
      }

      polyscope::registerSurfaceMesh("Ground", groundV, groundF);


        // visualize!
        polyscope::PointCloud* psCloud = polyscope::registerPointCloud("really great points", points);
        // set some options
        psCloud->setPointRadius(0.005);

      polyscope::frameTick();
  }

  return 0;
}

