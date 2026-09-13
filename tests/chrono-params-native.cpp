#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <cstring>
// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Author: Milad Rakhsha, Wei Hu, Luning Bakke
// =============================================================================

#include <assert.h>
#include <stdlib.h>
#include <ctime>

#include "chrono/physics/ChSystemSMC.h"
#include "chrono/assets/ChVisualSystem.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsGenerators.h"
#include "chrono/utils/ChUtilsGeometry.h"

#include "chrono_fsi/sph/ChFsiSystemSPH.h"

#ifdef CHRONO_VSG
    #include "chrono_fsi/sph/visualization/ChSphVisualizationVSG.h"
#endif

#include "chrono_thirdparty/cxxopts/ChCLI.h"

using namespace chrono;
using namespace chrono::fsi;
using namespace chrono::fsi::sph;

// =============================================================================

bool GetProblemSpecs(int argc, char** argv, double& t_end, bool& verbose, bool& output, double& output_fps, bool& render, double& render_fps, bool& snapshots, int& ps_freq) {
    ChCLI cli(argv[0], "Dam Break FSI demo");

    cli.AddOption<double>("Input", "t_end", "Simulation duration [s]", std::to_string(t_end));

    cli.AddOption<bool>("Output", "quiet", "Disable verbose terminal output");

    cli.AddOption<bool>("Output", "output", "Enable collection of output files");
    cli.AddOption<double>("Output", "output_fps", "Output frequency [fps]", std::to_string(output_fps));

    cli.AddOption<bool>("Visualization", "no_vis", "Disable run-time visualization");
    cli.AddOption<double>("Visualization", "render_fps", "Render frequency [fps]", std::to_string(render_fps));
    cli.AddOption<bool>("Visualization", "snapshots", "Enable writing snapshot image files");

    cli.AddOption<int>("Proximity Search", "ps_freq", "Frequency of Proximity Search", std::to_string(ps_freq));

    if (!cli.Parse(argc, argv)) {
        cli.Help();
        return false;
    }

    t_end = cli.GetAsType<double>("t_end");

    verbose = !cli.GetAsType<bool>("quiet");
    output = cli.GetAsType<bool>("output");
    render = !cli.GetAsType<bool>("no_vis");
    snapshots = cli.GetAsType<bool>("snapshots");

    output_fps = cli.GetAsType<double>("output_fps");
    render_fps = cli.GetAsType<double>("render_fps");

    ps_freq = cli.GetAsType<int>("ps_freq");

    return true;
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    double t_end = 10.0;
    double initial_spacing = 0.1;
    double step_size = 1e-4;
    bool verbose = true;
    bool output = false;
    double output_fps = 20;
    bool render = true;
    double render_fps = 100;
    bool snapshots = false;
    int ps_freq = 1;
    if (!GetProblemSpecs(argc, argv, t_end, verbose, output, output_fps, render, render_fps, snapshots, ps_freq))
        return 1;

    // Dimension of the space domain
    double bxDim = 12.0;
    double byDim = 1.0;
    double bzDim = 8.0;

    // Dimension of the fluid domain
    double fxDim = 4.0;
    double fyDim = 1.0;
    double fzDim = 4.0;

    // Create a physics system and an FSI system
    ChSystemSMC sysMBS;
    ChFsiFluidSystemSPH sysSPH;
    ChFsiSystemSPH sysFSI(&sysMBS, &sysSPH);

    sysFSI.SetVerbose(verbose);

    sysFSI.SetStepSizeCFD(step_size);
    sysFSI.SetStepsizeMBD(step_size);

    ChFsiFluidSystemSPH::FluidProperties fluid_props;
    fluid_props.density = 1000;
    fluid_props.viscosity = 5;

    sysSPH.SetCfdSPH(fluid_props);

    // Set gravitational acceleration
    const ChVector3d gravity(0, 0, -9.8);
    sysFSI.SetGravitationalAcceleration(gravity);

    ChFsiFluidSystemSPH::SPHParameters sph_params;
    sph_params.integration_scheme = IntegrationScheme::RK2;
    sph_params.initial_spacing = initial_spacing;
    sph_params.d0_multiplier = 1;
    sph_params.max_velocity = 10.0;
    sph_params.shifting_method = ShiftingMethod::XSPH;
    sph_params.shifting_xsph_eps = 0.5;
    sph_params.artificial_viscosity = 0.03;
    sph_params.viscosity_method = ViscosityMethod::ARTIFICIAL_UNILATERAL;
    sph_params.eos_type = EosType::TAIT;
    sph_params.use_consistent_gradient_discretization = false;
    sph_params.use_consistent_laplacian_discretization = false;
    sph_params.num_proximity_search_steps = ps_freq;
    sph_params.use_delta_sph = true;
    sph_params.delta_sph_coefficient = 0.1;
    sph_params.boundary_method = BoundaryMethod::ADAMI;

    sysSPH.SetSPHParameters(sph_params);

    // Set frequency of proximity search
    sysSPH.SetNumProximitySearchSteps(ps_freq);

    // Set the shifting method
    sysSPH.SetShiftingMethod(ShiftingMethod::XSPH);
    sysSPH.SetShiftingXSPHParameters(0.5);

    // Set up the periodic boundary condition (only in Y direction)
    auto initSpace0 = sysSPH.GetInitialSpacing();
    ChVector3d cMin(-bxDim / 2 - 10 * initSpace0, -byDim / 2 - initSpace0 / 2, -2 * bzDim);
    ChVector3d cMax(+bxDim / 2 + 10 * initSpace0, +byDim / 2 + initSpace0 / 2, +2 * bzDim);
    sysSPH.SetComputationalDomain(ChAABB(cMin, cMax), BC_Y_PERIODIC);

    // Create Fluid region and discretize with SPH particles
    ChVector3d boxCenter(-bxDim / 2 + fxDim / 2, 0.0, fzDim / 2);
    ChVector3d boxHalfDim(fxDim / 2 - initSpace0, fyDim / 2, fzDim / 2 - initSpace0);

    // Use a chrono sampler to create a bucket of points
    chrono::utils::ChGridSampler<> sampler(initSpace0);
    chrono::utils::ChGenerator::PointVector points = sampler.SampleBox(boxCenter, boxHalfDim);

    // Add fluid particles from the sampler points to the FSI system
    size_t numPart = points.size();
    double gz = std::abs(sysSPH.GetGravitationalAcceleration().z());
    for (int i = 0; i < numPart; i++) {
        // Calculate the pressure of a steady state (p = rho*g*h)
        auto pre_ini = sysSPH.GetDensity() * gz * (-points[i].z() + fzDim);
        auto rho_ini = sysSPH.GetDensity() + pre_ini / (sysSPH.GetSoundSpeed() * sysSPH.GetSoundSpeed());
        sysSPH.AddSPHParticle(points[i], rho_ini, pre_ini, sysSPH.GetViscosity());
    }

    // Create container and attach BCE SPH particles
    auto ground = chrono_types::make_shared<ChBody>();
    ground->SetFixed(true);
    ground->EnableCollision(false);
    sysMBS.AddBody(ground);

    auto ground_bce = sysSPH.CreatePointsBoxContainer(ChVector3d(bxDim, byDim, bzDim), {2, 0, 2});
    sysFSI.AddFsiBoundary(ground_bce, ChFrame<>(ChVector3d(0, 0, bzDim / 2), QUNIT));

    // Complete construction of the FSI system
    sysFSI.Initialize();
// Host-only reference capture after the unchanged original initialization.
const auto& p=sysSPH.GetParams();
std::ofstream binary(".local/chrono-params.bin",std::ios::binary);binary.write((const char*)&p,sizeof(p));binary.close();
std::ofstream json(".local/chrono-params.json");json << std::setprecision(17) << "{";
json << "\"constant.paramsD.physics_problem\":" << (int)p.physics_problem;
json << ",\"constant.paramsD.rheology_model_crm\":" << (int)p.rheology_model_crm;
json << ",\"constant.paramsD.integration_scheme\":" << (int)p.integration_scheme;
json << ",\"constant.paramsD.eos_type\":" << (int)p.eos_type;
json << ",\"constant.paramsD.viscosity_method\":" << (int)p.viscosity_method;
json << ",\"constant.paramsD.boundary_method\":" << (int)p.boundary_method;
json << ",\"constant.paramsD.kernel_type\":" << (int)p.kernel_type;
json << ",\"constant.paramsD.shifting_method\":" << (int)p.shifting_method;
json << ",\"constant.paramsD.gridSize.x\":" << p.gridSize.x;
json << ",\"constant.paramsD.gridSize.y\":" << p.gridSize.y;
json << ",\"constant.paramsD.gridSize.z\":" << p.gridSize.z;
json << ",\"constant.paramsD.worldOrigin.x\":" << p.worldOrigin.x;
json << ",\"constant.paramsD.worldOrigin.y\":" << p.worldOrigin.y;
json << ",\"constant.paramsD.worldOrigin.z\":" << p.worldOrigin.z;
json << ",\"constant.paramsD.cellSize.x\":" << p.cellSize.x;
json << ",\"constant.paramsD.cellSize.y\":" << p.cellSize.y;
json << ",\"constant.paramsD.cellSize.z\":" << p.cellSize.z;
json << ",\"constant.paramsD.numBodies\":" << p.numBodies;
json << ",\"constant.paramsD.boxDims.x\":" << p.boxDims.x;
json << ",\"constant.paramsD.boxDims.y\":" << p.boxDims.y;
json << ",\"constant.paramsD.boxDims.z\":" << p.boxDims.z;
json << ",\"constant.paramsD.zombieBoxDims.x\":" << p.zombieBoxDims.x;
json << ",\"constant.paramsD.zombieBoxDims.y\":" << p.zombieBoxDims.y;
json << ",\"constant.paramsD.zombieBoxDims.z\":" << p.zombieBoxDims.z;
json << ",\"constant.paramsD.zombieOrigin.x\":" << p.zombieOrigin.x;
json << ",\"constant.paramsD.zombieOrigin.y\":" << p.zombieOrigin.y;
json << ",\"constant.paramsD.zombieOrigin.z\":" << p.zombieOrigin.z;
json << ",\"constant.paramsD.d0\":" << p.d0;
json << ",\"constant.paramsD.ood0\":" << p.ood0;
json << ",\"constant.paramsD.d0_multiplier\":" << p.d0_multiplier;
json << ",\"constant.paramsD.h\":" << p.h;
json << ",\"constant.paramsD.ooh\":" << p.ooh;
json << ",\"constant.paramsD.h_multiplier\":" << p.h_multiplier;
json << ",\"constant.paramsD.num_neighbors\":" << p.num_neighbors;
json << ",\"constant.paramsD.epsMinMarkersDis\":" << p.epsMinMarkersDis;
json << ",\"constant.paramsD.num_bce_layers\":" << p.num_bce_layers;
json << ",\"constant.paramsD.toleranceZone\":" << p.toleranceZone;
json << ",\"constant.paramsD.base_pressure\":" << p.base_pressure;
json << ",\"constant.paramsD.delta_pressure.x\":" << p.delta_pressure.x;
json << ",\"constant.paramsD.delta_pressure.y\":" << p.delta_pressure.y;
json << ",\"constant.paramsD.delta_pressure.z\":" << p.delta_pressure.z;
json << ",\"constant.paramsD.V_in.x\":" << p.V_in.x;
json << ",\"constant.paramsD.V_in.y\":" << p.V_in.y;
json << ",\"constant.paramsD.V_in.z\":" << p.V_in.z;
json << ",\"constant.paramsD.x_in\":" << p.x_in;
json << ",\"constant.paramsD.gravity.x\":" << p.gravity.x;
json << ",\"constant.paramsD.gravity.y\":" << p.gravity.y;
json << ",\"constant.paramsD.gravity.z\":" << p.gravity.z;
json << ",\"constant.paramsD.bodyForce3.x\":" << p.bodyForce3.x;
json << ",\"constant.paramsD.bodyForce3.y\":" << p.bodyForce3.y;
json << ",\"constant.paramsD.bodyForce3.z\":" << p.bodyForce3.z;
json << ",\"constant.paramsD.rho0\":" << p.rho0;
json << ",\"constant.paramsD.invrho0\":" << p.invrho0;
json << ",\"constant.paramsD.volume0\":" << p.volume0;
json << ",\"constant.paramsD.markerMass\":" << p.markerMass;
json << ",\"constant.paramsD.mu0\":" << p.mu0;
json << ",\"constant.paramsD.v_Max\":" << p.v_Max;
json << ",\"constant.paramsD.shifting_xsph_eps\":" << p.shifting_xsph_eps;
json << ",\"constant.paramsD.shifting_ppst_push\":" << p.shifting_ppst_push;
json << ",\"constant.paramsD.shifting_ppst_pull\":" << p.shifting_ppst_pull;
json << ",\"constant.paramsD.shifting_beta_implicit\":" << p.shifting_beta_implicit;
json << ",\"constant.paramsD.shifting_diffusion_A\":" << p.shifting_diffusion_A;
json << ",\"constant.paramsD.shifting_diffusion_AFSM\":" << p.shifting_diffusion_AFSM;
json << ",\"constant.paramsD.shifting_diffusion_AFST\":" << p.shifting_diffusion_AFST;
json << ",\"constant.paramsD.dT\":" << p.dT;
json << ",\"constant.paramsD.kdT\":" << p.kdT;
json << ",\"constant.paramsD.gammaBB\":" << p.gammaBB;
json << ",\"constant.paramsD.use_default_limits\":" << p.use_default_limits;
json << ",\"constant.paramsD.use_init_pressure\":" << p.use_init_pressure;
json << ",\"constant.paramsD.cMinInit.x\":" << p.cMinInit.x;
json << ",\"constant.paramsD.cMinInit.y\":" << p.cMinInit.y;
json << ",\"constant.paramsD.cMinInit.z\":" << p.cMinInit.z;
json << ",\"constant.paramsD.cMaxInit.x\":" << p.cMaxInit.x;
json << ",\"constant.paramsD.cMaxInit.y\":" << p.cMaxInit.y;
json << ",\"constant.paramsD.cMaxInit.z\":" << p.cMaxInit.z;
json << ",\"constant.paramsD.binSize0\":" << p.binSize0;
unsigned int words[2];std::memcpy(words,&p.pressure_height,8);
json << ",\"constant.paramsD.pressure_height.lo\":" << words[0];
json << ",\"constant.paramsD.pressure_height.hi\":" << words[1];
json << ",\"constant.paramsD.density_reinit_steps\":" << p.density_reinit_steps;
json << ",\"constant.paramsD.Conservative_Form\":" << p.Conservative_Form;
json << ",\"constant.paramsD.gradient_type\":" << p.gradient_type;
json << ",\"constant.paramsD.laplacian_type\":" << p.laplacian_type;
json << ",\"constant.paramsD.use_consistent_gradient_discretization\":" << p.use_consistent_gradient_discretization;
json << ",\"constant.paramsD.use_consistent_laplacian_discretization\":" << p.use_consistent_laplacian_discretization;
json << ",\"constant.paramsD.use_delta_sph\":" << p.use_delta_sph;
json << ",\"constant.paramsD.density_delta\":" << p.density_delta;
json << ",\"constant.paramsD.use_density_based_projection\":" << p.use_density_based_projection;
json << ",\"constant.paramsD.Pressure_Constraint\":" << p.Pressure_Constraint;
json << ",\"constant.paramsD.LinearSolver\":" << (int)p.LinearSolver;
json << ",\"constant.paramsD.Alpha\":" << p.Alpha;
json << ",\"constant.paramsD.LinearSolver_Abs_Tol\":" << p.LinearSolver_Abs_Tol;
json << ",\"constant.paramsD.LinearSolver_Rel_Tol\":" << p.LinearSolver_Rel_Tol;
json << ",\"constant.paramsD.LinearSolver_Max_Iter\":" << p.LinearSolver_Max_Iter;
json << ",\"constant.paramsD.Verbose_monitoring\":" << p.Verbose_monitoring;
json << ",\"constant.paramsD.Max_Pressure\":" << p.Max_Pressure;
json << ",\"constant.paramsD.PPE_relaxation\":" << p.PPE_relaxation;
json << ",\"constant.paramsD.ClampPressure\":" << p.ClampPressure;
json << ",\"constant.paramsD.IncompressibilityFactor\":" << p.IncompressibilityFactor;
json << ",\"constant.paramsD.Cs\":" << p.Cs;
json << ",\"constant.paramsD.Apply_BC_U\":" << p.Apply_BC_U;
json << ",\"constant.paramsD.L_Characteristic\":" << p.L_Characteristic;
json << ",\"constant.paramsD.non_newtonian\":" << p.non_newtonian;
json << ",\"constant.paramsD.rheology_model\":" << (int)p.rheology_model;
json << ",\"constant.paramsD.ave_diam\":" << p.ave_diam;
json << ",\"constant.paramsD.cohesion\":" << p.cohesion;
json << ",\"constant.paramsD.mu_of_I\":" << (int)p.mu_of_I;
json << ",\"constant.paramsD.mu_max\":" << p.mu_max;
json << ",\"constant.paramsD.mu_fric_s\":" << p.mu_fric_s;
json << ",\"constant.paramsD.mu_fric_2\":" << p.mu_fric_2;
json << ",\"constant.paramsD.mu_I0\":" << p.mu_I0;
json << ",\"constant.paramsD.mu_I_b\":" << p.mu_I_b;
json << ",\"constant.paramsD.HB_sr0\":" << p.HB_sr0;
json << ",\"constant.paramsD.HB_k\":" << p.HB_k;
json << ",\"constant.paramsD.HB_n\":" << p.HB_n;
json << ",\"constant.paramsD.HB_tau0\":" << p.HB_tau0;
json << ",\"constant.paramsD.E_young\":" << p.E_young;
json << ",\"constant.paramsD.G_shear\":" << p.G_shear;
json << ",\"constant.paramsD.INV_G_shear\":" << p.INV_G_shear;
json << ",\"constant.paramsD.K_bulk\":" << p.K_bulk;
json << ",\"constant.paramsD.Nu_poisson\":" << p.Nu_poisson;
json << ",\"constant.paramsD.artificial_viscosity\":" << p.artificial_viscosity;
json << ",\"constant.paramsD.Coh_coeff\":" << p.Coh_coeff;
json << ",\"constant.paramsD.free_surface_threshold\":" << p.free_surface_threshold;
json << ",\"constant.paramsD.mcc_M\":" << p.mcc_M;
json << ",\"constant.paramsD.mcc_kappa\":" << p.mcc_kappa;
json << ",\"constant.paramsD.mcc_lambda\":" << p.mcc_lambda;
json << ",\"constant.paramsD.mcc_v_lambda\":" << p.mcc_v_lambda;
json << ",\"constant.paramsD.boxDimX\":" << p.boxDimX;
json << ",\"constant.paramsD.boxDimY\":" << p.boxDimY;
json << ",\"constant.paramsD.boxDimZ\":" << p.boxDimZ;
json << ",\"constant.paramsD.bc_type.x\":" << (int)p.bc_type.x;
json << ",\"constant.paramsD.bc_type.y\":" << (int)p.bc_type.y;
json << ",\"constant.paramsD.bc_type.z\":" << (int)p.bc_type.z;
json << ",\"constant.paramsD.x_periodic\":" << p.x_periodic;
json << ",\"constant.paramsD.y_periodic\":" << p.y_periodic;
json << ",\"constant.paramsD.z_periodic\":" << p.z_periodic;
json << ",\"constant.paramsD.minBounds.x\":" << p.minBounds.x;
json << ",\"constant.paramsD.minBounds.y\":" << p.minBounds.y;
json << ",\"constant.paramsD.minBounds.z\":" << p.minBounds.z;
json << ",\"constant.paramsD.maxBounds.x\":" << p.maxBounds.x;
json << ",\"constant.paramsD.maxBounds.y\":" << p.maxBounds.y;
json << ",\"constant.paramsD.maxBounds.z\":" << p.maxBounds.z;
json << ",\"constant.paramsD.cMin.x\":" << p.cMin.x;
json << ",\"constant.paramsD.cMin.y\":" << p.cMin.y;
json << ",\"constant.paramsD.cMin.z\":" << p.cMin.z;
json << ",\"constant.paramsD.cMax.x\":" << p.cMax.x;
json << ",\"constant.paramsD.cMax.y\":" << p.cMax.y;
json << ",\"constant.paramsD.cMax.z\":" << p.cMax.z;
json << ",\"constant.paramsD.zombieMin.x\":" << p.zombieMin.x;
json << ",\"constant.paramsD.zombieMin.y\":" << p.zombieMin.y;
json << ",\"constant.paramsD.zombieMin.z\":" << p.zombieMin.z;
json << ",\"constant.paramsD.zombieMax.x\":" << p.zombieMax.x;
json << ",\"constant.paramsD.zombieMax.y\":" << p.zombieMax.y;
json << ",\"constant.paramsD.zombieMax.z\":" << p.zombieMax.z;
json << ",\"constant.paramsD.free_flow_duration\":" << p.free_flow_duration;
json << ",\"constant.paramsD.num_proximity_search_steps\":" << p.num_proximity_search_steps;
json << ",\"constant.paramsD.use_variable_time_step\":" << p.use_variable_time_step;
json << "}";json.close();std::cout << "Captured params: " << sizeof(p) << " bytes\n";std::ofstream positions(".local/chrono-initial-positions.bin",std::ios::binary);for(const auto& point:points){float xyz[3]={(float)point.x(),(float)point.y(),(float)point.z()};positions.write((const char*)xyz,sizeof(xyz));}positions.close();
const auto allPositions=sysSPH.GetPositions();const auto view=sysSPH.GetMarkerDeviceView();
std::vector<Real4> markers(allPositions.size());
if(cudaMemcpy(markers.data(),view.pos_rad,markers.size()*sizeof(Real4),cudaMemcpyDeviceToHost)!=cudaSuccess)return 1;
std::ofstream markerFile(".local/chrono-marker-posrad.bin",std::ios::binary);markerFile.write((const char*)markers.data(),markers.size()*sizeof(Real4));markerFile.close();
std::ofstream markerInfo(".local/chrono-marker-info.json");markerInfo << "{\"markers\":" << markers.size() << ",\"fluid\":" << view.num_fluid_markers << "}";markerInfo.close();
const auto properties=sysSPH.GetProperties();const auto velocities=sysSPH.GetVelocities();
if(properties.size()!=markers.size()||velocities.size()!=markers.size())return 2;
std::ofstream propertyFile(".local/chrono-marker-properties.bin",std::ios::binary);propertyFile.write((const char*)properties.data(),properties.size()*sizeof(Real3));propertyFile.close();
std::ofstream velocityFile(".local/chrono-marker-velocities.bin",std::ios::binary);velocityFile.write((const char*)velocities.data(),velocities.size()*sizeof(Real3));velocityFile.close();
std::cout << "Captured " << markers.size() << " markers, including " << view.num_fluid_markers << " fluid markers\n";
return 0;
}
