// ActiveDomain, Counters, activity kernel and helpers: Copyright (c) 2014 projectchrono.org.
// Real3/Real4 definitions: Copyright (c) 2025 projectchrono.org.
// Parameter record and grid helpers: Copyright (c) 2014 projectchrono.org.
// BSD terms are reproduced in THIRD_PARTY_NOTICES.md.
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

typedef float Real;
typedef unsigned int uint;
struct Real3 {
    Real x;
    Real y;
    Real z;
};
struct Real4 {
    Real x;
    Real y;
    Real z;
    Real w;
};
struct BoundaryConditions {
    BCType x;
    BCType y;
    BCType z;
};
struct ChFsiParamsSPH {
    PhysicsProblem physics_problem;        ///< Physics problem type: (CFD or CRM)
    RheologyCRM rheology_model_crm;        ///< Rheology model for CRM problems (MU_OF_I or MCC)

    IntegrationScheme integration_scheme;  ///< Integration scheme
    EosType eos_type;                      ///< Equation of state type (Tait or isothermal)
    ViscosityMethod viscosity_method;      ///< Viscosity treatment type (physics-based laminar flow or artificial)
    BoundaryMethod boundary_method;        ///< Boundary type (Adami or Holmes)
    KernelType kernel_type;                ///< Kernel type (Quadratic, cubic spline, quintic spline, quintic Wendland)
    ShiftingMethod shifting_method;        ///< Shifting method (NONE, PPST, XSPH, PPST_XSPH)

    int3 gridSize;          ///< dx, dy, dz distances between particle centers
    Real3 worldOrigin;      ///< Origin point
    Real3 cellSize;         ///< Cell size for the neighbor particle search
    uint numBodies;         ///< Number of FSI bodies.
    Real3 boxDims;          ///< Dimensions (AABB) of the domain
    Real3 zombieBoxDims;    ///< Dimensions (AABB) of the zombie domain
    Real3 zombieOrigin;     ///< Origin point of the zombie domain
    Real d0;                ///< Initial separation of SPH particles
    Real ood0;              ///< 1 / d0
    Real d0_multiplier;     ///< Multiplier to obtain the interaction length, h = d0_multiplier * d0
    Real h;                 ///< Kernel interaction length
    Real ooh;               ///< 1 / h
    Real h_multiplier;      ///< Multiplier to obtain kernel radius, r = h_multiplier * h (depends on kernel type)
    int num_neighbors;      ///< Number of neighbor particles
    Real epsMinMarkersDis;  ///< Multiplier for minimum distance between markers (d_min = eps * h)
    int num_bce_layers;     ///< Number of BCE marker layers attached to boundary and solid surfaces (default: 3)
    Real toleranceZone;     ///< Helps determine the particles that are in the domain but are outside the boundaries, so
                            ///< they are not considered fluid particles and are dropped at the beginning of the simulation.

    Real base_pressure;    ///< Relative value of pressure applied to the whole domain
    Real3 delta_pressure;  ///< Change in Pressure for periodic BC (when particle moves from one side to the other)

    Real3 V_in;  ///< Inlet velocity for inlet BC
    Real x_in;   ///< Inlet position for inlet BC

    Real3 gravity;     ///< Gravitational acceleration
    Real3 bodyForce3;  ///< Constant force applied to the fluid particles (solids not directly affected)

    Real rho0;     ///< Density
    Real invrho0;  ///< 1 / rho0
    Real volume0;  ///< Initial particle volume

    Real markerMass;               ///< marker mass
    Real mu0;                      ///< Viscosity
    Real v_Max;                    ///< Max velocity of fluid used in equation of state. Run simulation once to be able to determine it.
    Real shifting_xsph_eps;        ///< Coefficient for XSPH shifting
    Real shifting_ppst_push;       ///< Coefficient for PPST shifting - this is applied when penetration with fictitious
                                   ///< sphere is detected
    Real shifting_ppst_pull;       ///< Coefficient for PPST pulling - this is applied when penetration with fictitious
    Real shifting_beta_implicit;   ///< Coefficient for shifting used in implicit scheme
    Real shifting_diffusion_A;     ///< Fickian shifting coefficient. Scales the shifting velocity
                                   ///< -A h |v_i| sum_j (m_j / rho_j) grad W_ij, whose sum is the discrete
                                   ///< gradient of particle concentration (default: 1.0, range 1 to 6)
    Real shifting_diffusion_AFSM;  ///< Upper anchor of the free-surface taper applied to diffusion
                                   ///< shifting. Particles whose position-field divergence reaches AFSM are
                                   ///< given the full shift; between AFST and AFSM the shift ramps linearly.
                                   ///< Complete 3D kernel support yields about 2.9 (default: 2.9)
    Real shifting_diffusion_AFST;  ///< Lower anchor of the free-surface taper applied to diffusion
                                   ///< shifting. Particles whose position-field divergence is at or below
                                   ///< AFST are not shifted at all (default: 2.0)

    Real dT;  ///< Time step. Depending on the model this will vary and the only way to determine what time step to
              ///< use is to run simulations multiple time and find which one is the largest dT that produces a
              ///< stable simulation.

    Real kdT;      ///< Implicit integration parameter
    Real gammaBB;  ///< Equation of state parameter

    bool use_default_limits;  ///< true if cMin and cMax are not user-provided (default: true)
    bool use_init_pressure;   ///< true if pressure set based on height (default: false)

    Real3 cMinInit;  ///< Minimum point of the fluid domain.
    Real3 cMaxInit;  ///< Maximum point of the fluid domain.
    Real binSize0;   ///< Suggests the length of the bin each particle occupies. Normally this would be 2*hsml since
                     ///< hsml is the radius of the particle, but when we have periodic boundary condition varies a
                     ///< little from 2 hsml.This may change slightly due to the location of the periodic BC.

    double pressure_height;  ///< height for pressure initialization

    // Note: more frequent re-initialization helps in getting more accurate incompressible fluid,
    // but more stable solution is obtained for larger values of density_reinit_steps
    int density_reinit_steps;  ///< reinitialize density after density_reinit_steps steps

    bool Conservative_Form;  ///< use conservative or consistent discretization
    int gradient_type;       ///< Type of the gradient operator
    int laplacian_type;      ///< Type of the Laplacian operator

    bool use_consistent_gradient_discretization;   ///< use consistent discretization for gradient operator
    bool use_consistent_laplacian_discretization;  ///< use consistent discretization for Laplacian operator
    bool use_delta_sph;                            ///< use delta SPH
    Real density_delta;                            ///< parameter for delta SPH

    bool use_density_based_projection;  ///< Set true to use density based projection scheme in ISPH solver

    bool Pressure_Constraint;  ///< Whether the singularity of the pressure equation should be fixed
    SolverType LinearSolver;   ///< Type of the linear solver

    Real Alpha;  ///< Poisson Pressure Equation source term constant. Used to control the noise in the FS forces

    Real LinearSolver_Abs_Tol;  ///< Poisson Pressure Equation residual
    Real LinearSolver_Rel_Tol;  ///< Poisson Pressure Equation Absolute residual
    int LinearSolver_Max_Iter;  ///< Linear Solver maximum number of iteration
    bool Verbose_monitoring;    ///< Poisson Pressure Equation Absolute residual

    Real Max_Pressure;             ///< Max Pressure in the pressure solver
    Real PPE_relaxation;           ///< PPE_relaxation
    bool ClampPressure;            ///< Clamp pressure to 0 if negative, based on the ISPH paper by Ihmsen et al. (2013)
    Real IncompressibilityFactor;  ///< Incompressibility factor (default: 1)
    Real Cs;                       ///< Speed of sound

    bool Apply_BC_U;        ///< This option lets you apply a velocity BC on the BCE markers
    Real L_Characteristic;  ///< Characteristic for Re number computation

    bool non_newtonian;       ///< Set true to model non-Newtonian fluid
    Rheology rheology_model;  ///< Model of the rheology
    Real ave_diam;            ///< average particle diameter
    Real cohesion;            ///< c in the stress model sigma=(mu*p+c)/|D|
    FrictionLaw mu_of_I;      ///< Constant I in granular material dynamics
    Real mu_max;              ///< maximum viscosity
    Real mu_fric_s;           ///< friction mu_s
    Real mu_fric_2;           ///< mu_2 constant in mu=mu(I)
    Real mu_I0;               ///< Reference Inertia number
    Real mu_I_b;              ///< b constant in mu=mu(I)=mu_s+b*I

    Real HB_sr0;   ///< Herschel–Bulkley consistency index
    Real HB_k;     ///< Herschel–Bulkley consistency index
    Real HB_n;     ///< Herschel–Bulkley  power
    Real HB_tau0;  ///< Herschel–Bulkley yield stress

    Real E_young;                 ///< Young's modulus
    Real G_shear;                 ///< Shear modulus
    Real INV_G_shear;             ///< 1.0 / G_shear
    Real K_bulk;                  ///< Bulk modulus
    Real Nu_poisson;              ///< Poisson's ratio
    Real artificial_viscosity;    ///< Artificial viscosity coefficient
    Real Coh_coeff;               ///< Cohesion coefficient
    Real free_surface_threshold;  ///< threshold for identifying free surface. The divergence of the position
    ///< field is computed and compared to this threshold. Particles with divergence
    ///< less than this threshold are considered free surface particles. Evaluated for
    ///< both CFD and CRM problems, but currently only the CRM solution consumes the
    ///< result (the stress state is zeroed at flagged particles); default: 2.4
    Real mcc_M;         ///< Cam-Clay critical state line slope, q = M p
    Real mcc_kappa;     ///< Cam-Clay swelling index: slope of the elastic unload/reload line in
                        ///< v-ln(p). Sets the elastic bulk modulus, K = v p / kappa.
                        ///< Must satisfy 0 < mcc_kappa < mcc_lambda
    Real mcc_lambda;    ///< Cam-Clay compression index: slope of the normal consolidation line in
                        ///< v-ln(p). Governs virgin compressibility and the hardening rate, which
                        ///< divides by (mcc_lambda - mcc_kappa). Must exceed mcc_kappa
    Real mcc_v_lambda;  ///< Specific volume at reference pressure of 1000 Pa

    Real boxDimX;  ///< Dimension of the space domain - X
    Real boxDimY;  ///< Dimension of the space domain - Y
    Real boxDimZ;  ///< Dimension of the space domain - Z

    BoundaryConditions bc_type;  ///< boundary condition types in the 3 domain directions
    bool x_periodic;             ///< periodic boundary conditions in x direction?
    bool y_periodic;             ///< periodic boundary conditions in y direction?
    bool z_periodic;             ///< periodic boundary conditions in z direction?

    int3 minBounds;  ///< Lower limit point of the grid (in grid index)
    int3 maxBounds;  ///< Upper limit point of the grid (in grid index)

    Real3 cMin;  ///< Lower limit point (in world coordinates)
    Real3 cMax;  ///< Upper limit point (in world coordinates)

    Real3 zombieMin;  ///< Lower limit point of the zombie domain -> All particles outside this will be frozen
    Real3 zombieMax;  ///< Upper limit point of the zombie domain -> All particles outside this will be frozen

    Real free_flow_duration;  ///< initial  duration for free flow CRM material (default: 0)

    int num_proximity_search_steps;  ///< Number of steps between updates to neighbor lists
    bool use_variable_time_step;     ///< use variable time step (default: false)
};
__constant__ static ChFsiParamsSPH paramsD;

// Unchanged active-domain and counter declarations from SphDataManager.cuh.
struct ActiveDomain {
    bool inverted;  ///< inverted (invalid) AABB
    Real3 a_min;    ///< min corner of active AABB
    Real3 a_max;    ///< max corner of active AABB
    Real3 e_min;    ///< min corner of extended active AABB
    Real3 e_max;    ///< max corner of extended active AABB
};
struct Counters {
    size_t numFsiBodies;      ///< number of rigid bodies
    size_t numFsiMeshes1D;    ///< number of 1-D FEA meshes
    size_t numFsiMeshes2D;    ///< number of 2-D FEA meshes
    size_t numFsiNodes1D;     ///< number of nodes in 1-D FEA mesh segments
    size_t numFsiNodes2D;     ///< number of nodes in 2-D FEA mesh faces
    size_t numFsiElements1D;  ///< number of 1-D FEA mesh segments
    size_t numFsiElements2D;  ///< number of 2-D FEA mesh faces

    size_t numGhostMarkers;     ///< number of Ghost SPH particles for Variable Resolution methods
    size_t numHelperMarkers;    ///< number of helper SPH particles used for merging particles
    size_t numFluidMarkers;     ///< number of fluid SPH particles
    size_t numBoundaryMarkers;  ///< number of BCE markers on boundaries
    size_t numRigidMarkers;     ///< number of BCE markers on rigid bodies
    size_t numMesh1DMarkers;    ///< number of BCE markers on flexible segments
    size_t numMesh2DMarkers;    ///< number of BCE markers on flexible faces
    size_t numBceMarkers;       ///< total number of BCE markers
    size_t numAllMarkers;       ///< total number of particles in the simulation

    size_t startBoundaryMarkers;  ///< index of first BCE marker on boundaries
    size_t startRigidMarkers;     ///< index of first BCE marker on first rigid body
    size_t startMesh1DMarkers;    ///< index of first BCE marker on first flex segment
    size_t startMesh2DMarkers;    ///< index of first BCE marker on first flex face
    size_t numActiveParticles;    ///< number of active particles
    size_t numExtendedParticles;  ///< number of extended particles
};
__constant__ static Counters countersD;
typedef int int32_t;
#define mR3 make_Real3
__host__ __device__ inline Real3 make_Real3(Real a, Real b, Real c)  ///
{
    Real3 d;
    d.x = a;
    d.y = b;
    d.z = c;
    return d;
}
__host__ __device__ inline Real3 make_Real3(Real4 a) {
    return make_Real3(a.x, a.y, a.z);
}
__host__ __device__ inline Real3 make_Real3(Real s) {
    return make_Real3(s, s, s);
}
__host__ __device__ inline bool IsFluidParticle(Real code) {
    return code < -0.5;
}
__device__ int32_t inAABB(const Real3& pos, const Real3& min, const Real3& max) {
    if ((pos.x >= min.x && pos.x <= max.x) &&  //
        (pos.y >= min.y && pos.y <= max.y) &&  //
        (pos.z >= min.z && pos.z <= max.z))
        return 1;
    return 0;
}
__device__ void checkActivityD(const Real3& pos,
                               const ActiveDomain* __restrict__ ad_body_D,
                               const ActiveDomain* __restrict__ ad_node1D_D,
                               const ActiveDomain* __restrict__ ad_node2D_D,
                               int32_t& active,
                               int32_t& ext_active) {
    active = 0;
    ext_active = 0;

    for (uint ib = 0; ib < countersD.numFsiBodies; ib++) {
        if (ad_body_D[ib].inverted)
            continue;
        active |= inAABB(pos, ad_body_D[ib].a_min, ad_body_D[ib].a_max);
        ext_active |= inAABB(pos, ad_body_D[ib].e_min, ad_body_D[ib].e_max);
        if (active == 1 && ext_active == 1)
            return;
    }

    for (uint ib = 0; ib < countersD.numFsiNodes1D; ib++) {
        if (ad_node1D_D[ib].inverted)
            continue;
        active |= inAABB(pos, ad_node1D_D[ib].a_min, ad_node1D_D[ib].a_max);
        ext_active |= inAABB(pos, ad_node1D_D[ib].e_min, ad_node1D_D[ib].e_max);
        if (active == 1 && ext_active == 1)
            return;
    }

    for (uint ib = 0; ib < countersD.numFsiNodes2D; ib++) {
        if (ad_node2D_D[ib].inverted)
            continue;
        active |= inAABB(pos, ad_node2D_D[ib].a_min, ad_node2D_D[ib].a_max);
        ext_active |= inAABB(pos, ad_node2D_D[ib].e_min, ad_node2D_D[ib].e_max);
        if (active == 1 && ext_active == 1)
            return;
    }
}
__global__ void UpdateActivityD(const Real4* posRadD,
                                Real3* velMasD,
                                const Real3* pos_bodies_D,
                                const Real3* pos_nodes1D_D,
                                const Real3* pos_nodes2D_D,
                                bool has_ad,
                                const ActiveDomain* __restrict__ ad_body_D,
                                const ActiveDomain* __restrict__ ad_node1D_D,
                                const ActiveDomain* __restrict__ ad_node2D_D,
                                int32_t* activityIdentifierD,
                                int32_t* extendedActivityIdD,
                                const Real4* rhoPreMuD,
                                double time) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= countersD.numAllMarkers) {
        return;
    }

    // Particle position
    Real3 pos = mR3(posRadD[index]);

    // Set particle activity: a particle is active if it is within the active AABB of a body or mesh node
    // All particles are considered active during a settling phase
    activityIdentifierD[index] = 1;
    extendedActivityIdD[index] = 1;
    if (has_ad && time >= paramsD.free_flow_duration) {
        checkActivityD(pos, ad_body_D, ad_node1D_D, ad_node2D_D, activityIdentifierD[index], extendedActivityIdD[index]);
        if (activityIdentifierD[index] == 0)
            velMasD[index] = mR3(0.0);
    }

    // Check if the particle is outside the zombie domain
    Real3 domainDims = paramsD.boxDims;
    Real3 domainOrigin = paramsD.worldOrigin;
    bool x_periodic = paramsD.x_periodic;
    bool y_periodic = paramsD.y_periodic;
    bool z_periodic = paramsD.z_periodic;

    if (IsFluidParticle(rhoPreMuD[index].w)) {
        bool outside_domain = false;

        // Check X boundaries - only inactivate if not periodic
        if (!x_periodic && (pos.x < domainOrigin.x || pos.x > domainOrigin.x + domainDims.x)) {
            outside_domain = true;
        }

        // Check Y boundaries - only inactivate if not periodic
        if (!y_periodic && (pos.y < domainOrigin.y || pos.y > domainOrigin.y + domainDims.y)) {
            outside_domain = true;
        }

        // Check Z boundaries - only inactivate if not periodic
        if (!z_periodic && (pos.z < domainOrigin.z || pos.z > domainOrigin.z + domainDims.z)) {
            outside_domain = true;
        }

        if (outside_domain) {
            activityIdentifierD[index] = -1;
            extendedActivityIdD[index] = -1;
            velMasD[index] = mR3(0.0);
        }
    }

    return;
}
