// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2024 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Author: Radu Serban
// =============================================================================
//
// Miscellaneous enumerations for the SPH-based Chrono::FSI fluid solver
//
// =============================================================================

// Unchanged scoped enums from Chrono a92c6f72f422fbcafe0b37125d4070cb6a3b5803.
enum class PhysicsProblem {
    CFD,  ///< incompressible fluid problem
    CRM   ///< continuous granular problem
};
enum class IntegrationScheme {
    EULER,        ///< Explicit Euler
    RK2,          ///< Runge-Kutta 2
    VERLET,       ///< Velocity Verlet
    SYMPLECTIC,   ///< Symplectic Euler
    IMPLICIT_SPH  ///< Implicit SPH
};
enum class ShiftingMethod { NONE, PPST, XSPH, PPST_XSPH, DIFFUSION, DIFFUSION_XSPH };
enum class EosType { TAIT, ISOTHERMAL };
enum class KernelType { QUADRATIC, CUBIC_SPLINE, QUINTIC_SPLINE, WENDLAND };
enum class ViscosityMethod { LAMINAR, ARTIFICIAL_UNILATERAL, ARTIFICIAL_BILATERAL };
enum class BoundaryMethod { ADAMI, HOLMES };
enum class Rheology { INERTIA_RHEOLOGY, NONLOCAL_FLUIDITY };
enum class FrictionLaw { CONSTANT, LINEAR, NONLINEAR };
enum class SolverType { JACOBI, BICGSTAB, GMRES, CR, CG, SAP };
enum class RheologyCRM { MU_OF_I, MCC };
enum class BCType {
    NONE,         ///< no boundary conditions enforced
    PERIODIC,     ///< periodic boundary conditions
    INLET_OUTLET  ///< inlet-outlet boundary conditions
};
enum class NodeDirections { NONE, AVERAGE, EXACT };
enum class BcePatternMesh1D { FULL, STAR };
enum class BcePatternMesh2D { CENTERED, OUTWARD, INWARD };
enum class OutputLevel {
    STATE,           ///< marker state, velocity, and acceleration
    STATE_PRESSURE,  ///< STATE plus density and pressure
    CFD_FULL,        ///< STATE_PRESSURE plus various CFD parameters
    CRM_FULL         ///< STATE_PRESSURE plus normal and shear stress
};

// MIT validation wrapper; no fluid solver is implemented here.
struct Flags { bool periodic; bool invert; };
__constant__ Flags flags;
__global__ void chronoEnumValues(int* output){
output[0]=(int)PhysicsProblem::CFD;
output[1]=(int)PhysicsProblem::CRM;
output[2]=(int)IntegrationScheme::EULER;
output[3]=(int)IntegrationScheme::RK2;
output[4]=(int)IntegrationScheme::VERLET;
output[5]=(int)IntegrationScheme::SYMPLECTIC;
output[6]=(int)IntegrationScheme::IMPLICIT_SPH;
output[7]=(int)ShiftingMethod::NONE;
output[8]=(int)ShiftingMethod::PPST;
output[9]=(int)ShiftingMethod::XSPH;
output[10]=(int)ShiftingMethod::PPST_XSPH;
output[11]=(int)ShiftingMethod::DIFFUSION;
output[12]=(int)ShiftingMethod::DIFFUSION_XSPH;
output[13]=(int)EosType::TAIT;
output[14]=(int)EosType::ISOTHERMAL;
output[15]=(int)KernelType::QUADRATIC;
output[16]=(int)KernelType::CUBIC_SPLINE;
output[17]=(int)KernelType::QUINTIC_SPLINE;
output[18]=(int)KernelType::WENDLAND;
output[19]=(int)ViscosityMethod::LAMINAR;
output[20]=(int)ViscosityMethod::ARTIFICIAL_UNILATERAL;
output[21]=(int)ViscosityMethod::ARTIFICIAL_BILATERAL;
output[22]=(int)BoundaryMethod::ADAMI;
output[23]=(int)BoundaryMethod::HOLMES;
output[24]=(int)Rheology::INERTIA_RHEOLOGY;
output[25]=(int)Rheology::NONLOCAL_FLUIDITY;
output[26]=(int)FrictionLaw::CONSTANT;
output[27]=(int)FrictionLaw::LINEAR;
output[28]=(int)FrictionLaw::NONLINEAR;
output[29]=(int)SolverType::JACOBI;
output[30]=(int)SolverType::BICGSTAB;
output[31]=(int)SolverType::GMRES;
output[32]=(int)SolverType::CR;
output[33]=(int)SolverType::CG;
output[34]=(int)SolverType::SAP;
output[35]=(int)RheologyCRM::MU_OF_I;
output[36]=(int)RheologyCRM::MCC;
output[37]=(int)BCType::NONE;
output[38]=(int)BCType::PERIODIC;
output[39]=(int)BCType::INLET_OUTLET;
output[40]=(int)NodeDirections::NONE;
output[41]=(int)NodeDirections::AVERAGE;
output[42]=(int)NodeDirections::EXACT;
output[43]=(int)BcePatternMesh1D::FULL;
output[44]=(int)BcePatternMesh1D::STAR;
output[45]=(int)BcePatternMesh2D::CENTERED;
output[46]=(int)BcePatternMesh2D::OUTWARD;
output[47]=(int)BcePatternMesh2D::INWARD;
output[48]=(int)OutputLevel::STATE;
output[49]=(int)OutputLevel::STATE_PRESSURE;
output[50]=(int)OutputLevel::CFD_FULL;
output[51]=(int)OutputLevel::CRM_FULL;
output[52]=flags.periodic?7:3;
output[53]=flags.invert?-1:1;
}
