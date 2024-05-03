#include "polyscope/polyscope.h"

#include "polyscope/messages.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"

#include <iostream>
#include <unordered_set>
#include <utility>
#include <algorithm>
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
        auto hash2 = std::hash<T>{}(std::hash<T>{}(v[1]));
        auto hash3 = std::hash<T>{}(std::hash<T>{}(std::hash<T>{}(v[2])));
        return hash1 ^ hash2 ^ hash3;
    }
};

bool running_;
double time_;
SimParameters params_;

std::vector<Eigen::Vector3d> points;
std::vector<Eigen::Vector3d> vel;

Eigen::MatrixXd groundV;
Eigen::MatrixXi groundF;
double edgeLen = 8.0;
double voxelLen = 0.4;

struct MouseClick
{
    double x;
    double y;
    double z;
};

std::deque<MouseClick> mouseClicks_;

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
    points.clear();
    vel.clear();

    switch(params_.example) {
        // CORNER TEST
        case 0:
            for (size_t i = 0; i < 5000; i++) {
                Eigen::Vector3d p = {polyscope::randomUnit() * (edgeLen / 4.0) + (edgeLen / 4.0), 
                            polyscope::randomUnit() * edgeLen - (edgeLen / 2.0), 
                            polyscope::randomUnit() * (edgeLen / 2.0)};
                Eigen::Vector3d v = {0, 0, 0};
                points.push_back(p);
                vel.push_back(v);
            }
            break;

        // WHIRLPOOL TEST
        case 1:
            for (size_t i = 0; i < 8000; i++) {
                double x = polyscope::randomUnit() * edgeLen - (edgeLen / 2.0);
                double y = polyscope::randomUnit() * edgeLen - (edgeLen / 2.0);
                Eigen::Vector3d p = {x, 
                            polyscope::randomUnit() * (edgeLen) / 7.0 - (edgeLen / 2.0), 
                            y};
                Eigen::Vector3d v = {y * 50.0, 0, -x * 50.0};
                points.push_back(p);
                vel.push_back(v);
            }
            break;

        // DROP TEST
        case 2:
            for (size_t i = 0; i < 6000; i++) {
                Eigen::Vector3d p = {
                    polyscope::randomUnit() * edgeLen - (edgeLen / 2.0), 
                    polyscope::randomUnit() * (edgeLen / 4.0) - (edgeLen / 2.0), 
                    polyscope::randomUnit() * edgeLen - (edgeLen / 2.0)};
                Eigen::Vector3d v = {0, 0, 0};
                points.push_back(p);
                vel.push_back(v);
            }
            for (size_t i = 0; i < 500; i++) {
                double sideLen = edgeLen / 10.0;
                Eigen::Vector3d p = {
                    polyscope::randomUnit() * sideLen - (sideLen / 2.0), 
                    edgeLen - polyscope::randomUnit() * sideLen, 
                    polyscope::randomUnit() * sideLen - (sideLen / 2.0)};
                Eigen::Vector3d v = {0, -1000.0, 0};
                points.push_back(p);
                vel.push_back(v);
            }
    }
}

void initSimulation()
{
    time_ = 0;
    loadScene();
}

void dispersal() {
    int n = points.size();
    std::unordered_map<Eigen::Vector3i, std::vector<int>, Vec3Hash> dispersalGrid;
    for (int i = 0; i < n; i++) {
        Eigen::Vector3i voxel;
        for (int j = 0; j < 3; j++) {
            voxel[j] = (int) ((points[i][j] + (edgeLen / 2.0)) / voxelLen);
        }
        if (dispersalGrid.count(voxel) == 0) dispersalGrid[voxel] = {};
        dispersalGrid[voxel].push_back(i);
    }
    for (auto iter = dispersalGrid.begin(); iter != dispersalGrid.end(); iter++) {
        Eigen::Vector3i voxel = iter->first;
        std::vector<int> blockInds = iter->second;
        std::vector<int> closeInds = {};
        Eigen::Vector3i newvoxel;
        for (int ax1 = -1; ax1 <= 1; ax1++) {
            newvoxel[0] = voxel[0] + ax1;
            for (int ax2 = -1; ax2 <= 1; ax2++) {
                newvoxel[1] = voxel[1] + ax2;
                for (int ax3 = -1; ax3 <= 1; ax3++) {
                    newvoxel[2] = voxel[2] + ax3;
                    if (ax1 == 0 && ax2 == 0 && ax3 == 0) continue;
                    if (dispersalGrid.count(newvoxel)) {
                        closeInds.insert(closeInds.end(), dispersalGrid[newvoxel].begin(), dispersalGrid[newvoxel].end());
                    }
                }
            }
        }
        // TODO: Paramaterize
        double dispersalForce = 2.0;
        for (size_t i = 0; i < blockInds.size(); i++) {
            for (size_t j = i + 1; j < blockInds.size(); j++) {
                int x = blockInds[i];
                int y = blockInds[j];
                Eigen::Vector3d diff = points[x] - points[y];
                if (diff.norm() < voxelLen) {
                    Eigen::Vector3d F = dispersalForce * diff.normalized() * (voxelLen - diff.norm());
                    vel[x] += F;
                    vel[y] -= F;
                }
            }
        }
        for (size_t i = 0; i < blockInds.size(); i++) {
            for (size_t j = 0; j < closeInds.size(); j++) {
                int x = blockInds[i];
                int y = closeInds[j];
                Eigen::Vector3d diff = points[x] - points[y];
                if (diff.norm() < voxelLen) {
                    Eigen::Vector3d F = dispersalForce * diff.normalized() * (voxelLen - diff.norm());
                    vel[x] += F;
                }
            }
        }
    }
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
        Eigen::Vector3i localCorneri;
        Eigen::Vector3d localCorner;
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
        Eigen::Vector3d newCorner;
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
    std::unordered_map<Eigen::Vector3i, int, Vec3Hash> waterCells = {};
    for (int i = 0; i < n; i++) {
        Eigen::Vector3i pt;
        for (int j = 0; j < 3; j++) {
            pt[j] = (int) ((points[i][j] + (edgeLen / 2.0)) / voxelLen);
        }
        if (waterCells.count(pt) == 0) waterCells[pt] = 0;
        waterCells[pt]++;
    }
    // Apply gravity
    for (auto iter = v.begin(); iter != v.end(); iter++) {
        int ind = (iter->first)[1];
        if (ind > minCoord) {
            v[iter->first] -= params_.gravityG;
        }
    }

    // Set all walls to zero
    for (auto iter = u.begin(); iter != u.end(); iter++) {
        int ind = (iter->first)[0];
        if (ind == minCoord || ind == maxCoord) u[iter->first] = 0;
    }
    for (auto iter = v.begin(); iter != v.end(); iter++) {
        int ind = (iter->first)[1];
        // TODO: Adding this adds a roof (see two below)
        if (ind == minCoord/* || ind == maxCoord*/) v[iter->first] = 0;
    }
    for (auto iter = w.begin(); iter != w.end(); iter++) {
        int ind = (iter->first)[2];
        if (ind == minCoord || ind == maxCoord) w[iter->first] = 0;
    }
    // Repeatedly enforce incompressibility
    // TODO: Parameterize
    for (int i = 0; i < 10; i++) {
        for (auto iter = waterCells.begin(); iter != waterCells.end(); iter++) {
            Eigen::Vector3i pt = iter->first;
            double density = iter->second;
            double d = 0;
            double s = 0;
            // Sum up d and s
            if (pt[0] != minCoord && pt[0] != maxCoord) {
                d -= u[pt];
                s++;
            }
            if (pt[1] != minCoord) {
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
            if (pt[1] != minCoord) {
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

            // TODO: Add parameterization for density
            d -= 2.0 * (density - 2.0);

            // Edit d and s
            if (pt[0] != minCoord && pt[0] != maxCoord) {
                u[pt] += d / s;
            }
            if (pt[1] != minCoord) {
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
            if (pt[1] != minCoord) {
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

void fromgrid(int ind, std::unordered_map<Eigen::Vector3i, double, Vec3Hash> gridsolve, std::unordered_map<Eigen::Vector3i, double, Vec3Hash> grid) {
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
        Eigen::Vector3i localCorneri;
        Eigen::Vector3d localCorner;
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
        Eigen::Vector3d newCorner;
        double picvel = 0;
        double flipvel = 0;
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
                    picvel += weight * gridsolve[globalCorneri];
                    flipvel += weight * (gridsolve[globalCorneri] - grid[globalCorneri]);
                    weightSum += weight;
                }
            }
        }
        picvel = picvel / weightSum;
        flipvel = vel[i][ind] + (flipvel / weightSum);
        double pic = 0.1;
        double closeness = std::fabs(points[i][ind] + edgeLen / 2.0);
        if (ind != 1) {
            closeness = std::min(closeness, std::fabs(points[i][ind] - edgeLen / 2.0));
        }
        if (closeness < voxelLen) pic = 1.0;
        vel[i][ind] = pic * picvel + (1.0 - pic) * flipvel;
    }
}

void simulateOneStep()
{
    double h = params_.timeStep;
    time_ += h;
    int n = points.size();

    dispersal();

    // point vel -> grid vel
    std::unordered_map<Eigen::Vector3i, double, Vec3Hash> u, v, w;
    makegrid(0, u);
    makegrid(1, v);
    makegrid(2, w);

    std::unordered_map<Eigen::Vector3i, double, Vec3Hash> usolve, vsolve, wsolve;
    usolve = u;
    vsolve = v;
    wsolve = w;

    // enforce incompressibility
    incompressibility(usolve, vsolve, wsolve);

    // grid vel -> point vel
    fromgrid(0, usolve, u);
    fromgrid(1, vsolve, v);
    fromgrid(2, wsolve, w);

    // update points
    for (int i = 0; i < n; i++) {
        points[i] += vel[i] * h;
        // backup clamping
        for (int j = 0; j < 3; j++) {
            if (points[i][j] <= -(edgeLen / 2.0) || (points[i][j] >= (edgeLen / 2.0) && j != 1)) {
                points[i][j] -= vel[i][j] * h;
            }
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
    ImGui::Checkbox("Add Particles", &params_.particleAdditionMode);
    if (ImGui::CollapsingHeader("Example Fluids", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (ImGui::Selectable("Corner")) {
            params_.example = 0;
            loadScene();
        }
        if (ImGui::Selectable("Whirlpool")) {
            params_.example = 1;
            loadScene();
        }
        if (ImGui::Selectable("Drop")) {
            params_.example = 2;
            loadScene();
        }
    }

    ImGuiIO& io = ImGui::GetIO();
    io.DisplayFramebufferScale = ImVec2(1, 1);
    // this now only works on macs with retina displays - maybe funky on older macbook airs?
    // more robust solution is documented here: https://github.com/ocornut/imgui/issues/5081
    #if defined(__APPLE__)
        io.DisplayFramebufferScale = ImVec2(2,2); 
    #endif

    if (params_.particleAdditionMode && (io.MouseClicked[0] || ImGui::IsMouseDragging(0))) { 
        MouseClick mc;
        glm::vec2 screenCoords{ io.MousePos.x, io.MousePos.y };
        int xInd, yInd;
        std::tie(xInd, yInd) = polyscope::view::screenCoordsToBufferInds(screenCoords);

        glm::mat4 view = polyscope::view::getCameraViewMatrix();
        glm::mat4 viewInv = glm::inverse(view);
        glm::mat4 proj = polyscope::view::getCameraPerspectiveMatrix();
        glm::mat4 projInv = glm::inverse(proj);

        glm::vec4 ndc = proj * view * glm::vec4(1,1,1,1);
        ndc /= ndc[3];
        double clickedDepth = ndc[2];

        // convert depth to world units
        glm::vec2 screenPos{ screenCoords.x / static_cast<float>(polyscope::view::windowWidth),
                            1.f - screenCoords.y / static_cast<float>(polyscope::view::windowHeight) };
        float z = clickedDepth;
        glm::vec4 clipPos = glm::vec4(screenPos * 2.0f - 1.0f, z, 1.0f);
        glm::vec4 viewPos = projInv * clipPos;
        viewPos /= viewPos.w;

        glm::vec4 worldPos = viewInv * viewPos;
        worldPos /= worldPos.w;
        mc.x = worldPos[0];
        mc.y = worldPos[1];
        mc.z = worldPos[2];
        mouseClicks_.push_back(mc);
    }

    ImGui::End();
}

int main(int argc, char **argv) 
{
    polyscope::view::setWindowSize(1600, 800);
    polyscope::options::buildGui = true;
    polyscope::options::openImGuiWindowForUserCallback = false;
    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;

    polyscope::options::autocenterStructures = false;
    polyscope::options::autoscaleStructures = false;
    polyscope::options::maxFPS = -1;

    initSimulation();

    polyscope::init();

    polyscope::state::userCallback = callback;

    while (!polyscope::render::engine->windowRequestsClose())
    {
        if (running_)
            simulateOneStep();
        updateRenderGeometry();

        polyscope::registerSurfaceMesh("Ground", groundV, groundF);

        polyscope::PointCloud* psCloud = polyscope::registerPointCloud("Points", points);

        double maxspeed = 300.0;
        for (size_t i = 0; i < vel.size(); i++) {
            maxspeed = std::max(vel[i].norm(), maxspeed);
        }

        std::vector<std::array<double, 3>> speedcolor(points.size());
        for (size_t i = 0; i < vel.size(); i++) {
            double speed = vel[i].norm();
            glm::vec3 blue = {0, 0, 1};
            glm::vec3 white = {1, 1, 1};
            float grad = glm::clamp(speed/maxspeed, 0.0, 1.0);
            glm::vec3 res = grad * white + (1.0f - grad) * blue;
            speedcolor[i] = {{res.x, res.y, res.z}};
        }

        psCloud->addColorQuantity("particle speed", speedcolor)->setEnabled(true);
        // set some options
        psCloud->setPointRadius(0.05, false);

        polyscope::frameTick();
        while (!mouseClicks_.empty())
        {
            MouseClick mc = mouseClicks_.front();
            mouseClicks_.pop_front();
            points.push_back({mc.x, mc.y, mc.z});
            vel.push_back({0, 0, 0});
        }
    }

    return 0;
}

