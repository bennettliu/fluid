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
#include <igl/signed_distance.h>

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

void updateRenderGeometry()
{           
    double groundSize = 5;

    groundV.resize(5, 3);
    groundV << 0, -1, 0,
        -groundSize, -1, -groundSize,
        -groundSize, -1, groundSize,
        groundSize, -1, groundSize,
        groundSize, -1, -groundSize;
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
    for (size_t i = 0; i < 10; i++) {
        Eigen::Vector3d p = {polyscope::randomUnit() * 10.0 - 5.0, 
                    polyscope::randomUnit() * 5.0 - 1.0, 
                    polyscope::randomUnit() * 10.0 - 5.0};
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

void processGravityForce(Eigen::VectorXd& F)
{
    int n = F.size() / 3;
    for (int i = 0; i < n; i++) {
        double force = -params_.gravityG;
        F[i * 3 + 1] += force;
    }
}

void processPenaltyForce(Eigen::VectorXd& F) {
    int n = points.size();
    // Floor force
    for (int i = 0; i < n; i++) {
        if (points[i][1] < -1.0) {
            double force = params_.penaltyStiffness * (1.0 - points[i][1]);
            F[i * 3 + 1] += force;
        }
    }
}

void computeForce(const Eigen::VectorXd& q, Eigen::VectorXd& F)
{
    F.resize(q.size());
    F.setZero();

    if (params_.gravityEnabled)
        processGravityForce(F);
    if (params_.penaltyEnabled)
        processPenaltyForce(F);
}

void simulateOneStep()
{
    double h = params_.timeStep;
    time_ += h;
    int n = points.size();
    Eigen::VectorXd q;
    Eigen::VectorXd qdot;
    q.resize(n * 3);
    qdot.resize(n * 3);
    Eigen::VectorXd F;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 3; j++) {
            q[i * 3 + j] = points[i][j];
            qdot[i * 3 + j] = vel[i][j];
        }
    }
    q += qdot * h;
    computeForce(q, F);
    qdot += F * h;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 3; j++) {
            points[i][j] = q[i * 3 + j];
            vel[i][j] = qdot[i * 3 + j];
        }
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

