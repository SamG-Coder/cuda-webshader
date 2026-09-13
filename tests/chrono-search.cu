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
__device__ inline int3 calcGridPos(Real3 p) {
    int3 gridPos;

    gridPos.x = (int)floor((p.x - paramsD.worldOrigin.x) / (paramsD.cellSize.x));
    gridPos.y = (int)floor((p.y - paramsD.worldOrigin.y) / (paramsD.cellSize.y));
    gridPos.z = (int)floor((p.z - paramsD.worldOrigin.z) / (paramsD.cellSize.z));
    return gridPos;
}
__device__ inline int reduceGridIndex(int i, int n, bool periodic) {
    if (periodic) {
        i %= n;
        return (i < 0) ? i + n : i;
    }
    return (i < 0) ? 0 : (i >= n) ? n - 1 : i;
}
__device__ inline uint calcGridHash(int3 gridPos) {
    gridPos.x = reduceGridIndex(gridPos.x, paramsD.gridSize.x, paramsD.x_periodic);
    gridPos.y = reduceGridIndex(gridPos.y, paramsD.gridSize.y, paramsD.y_periodic);
    gridPos.z = reduceGridIndex(gridPos.z, paramsD.gridSize.z, paramsD.z_periodic);

    return gridPos.z * paramsD.gridSize.y * paramsD.gridSize.x + gridPos.y * paramsD.gridSize.x + gridPos.x;
}

#define mR3 make_Real3
#define mI3 make_int3
#define CH_FSI_SPH_NOINLINE __declspec(noinline)
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
__host__ __device__ inline Real3 operator-(Real3 a, Real3 b) {
    return make_Real3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ inline CH_FSI_SPH_NOINLINE Real MinimumImageShiftMultiPeriod(Real dist, Real period) {
    return period * rint(dist / period);
}
__device__ inline Real MinimumImageShift(Real dist, Real period) {
    // A non-periodic axis exits first and must: it has period == 0, and falling through to the
    // division would evaluate 0 * rint(dist / 0), which is 0 * inf, which is NaN, in the distance
    // function every force kernel uses.
    if (period <= 0)
        return Real(0);
    // The common case costs two comparisons and no call, which is what it cost before the fix.
    Real half = Real(0.5) * period;
    if (dist <= half && dist >= -half)
        return Real(0);
    return MinimumImageShiftMultiPeriod(dist, period);
}
__device__ inline Real3 Modify_Local_PosB(Real3& b, Real3 a) {
    Real3 dist3 = a - b;
    b.x += MinimumImageShift(dist3.x, paramsD.x_periodic ? paramsD.boxDims.x : Real(0));
    b.y += MinimumImageShift(dist3.y, paramsD.y_periodic ? paramsD.boxDims.y : Real(0));
    b.z += MinimumImageShift(dist3.z, paramsD.z_periodic ? paramsD.boxDims.z : Real(0));

    dist3 = a - b;
    // modifying the markers perfect overlap
    Real dd = dist3.x * dist3.x + dist3.y * dist3.y + dist3.z * dist3.z;
    Real MinD = paramsD.epsMinMarkersDis * paramsD.h;
    Real sq_MinD = MinD * MinD;
    if (dd < sq_MinD) {
        dist3 = mR3(MinD, 0, 0);
    }
    b = a - dist3;
    return (dist3);
}
__device__ inline Real3 Distance(Real3 a, Real3 b) {
    return Modify_Local_PosB(b, a);
}
inline __host__ __device__ int3 operator+(int3 a, int3 b) {
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__global__ void neighborSearchNum(const Real4* sortedPosRad,
                                  const Real4* sortedRhoPreMu,
                                  const uint* cellStart,
                                  const uint* cellEnd,
                                  const uint numActive,
                                  uint* numNeighborsPerPart) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numActive) {
        return;
    }

    Real3 posRadA = mR3(sortedPosRad[index]);
    int3 gridPos = calcGridPos(posRadA);
    Real SuppRadii = 2.0f * paramsD.h;
    Real SqRadii = SuppRadii * SuppRadii;
    uint j_num = 0;

    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int3 neighborPos = gridPos + mI3(x, y, z);
                // Check if we need to skip this neighbor position (out of bounds for non-periodic dimensions)
                if (neighborPos.x < paramsD.minBounds.x || neighborPos.x > paramsD.maxBounds.x || neighborPos.y < paramsD.minBounds.y || neighborPos.y > paramsD.maxBounds.y ||
                    neighborPos.z < paramsD.minBounds.z || neighborPos.z > paramsD.maxBounds.z) {
                    continue;
                }
                uint gridHash = calcGridHash(neighborPos);
                uint startIndex = cellStart[gridHash];
                uint endIndex = cellEnd[gridHash];
                for (uint j = startIndex; j < endIndex; j++) {
                    Real3 posRadB = mR3(sortedPosRad[j]);
                    Real3 dist3 = Distance(posRadA, posRadB);
                    Real dd = dist3.x * dist3.x + dist3.y * dist3.y + dist3.z * dist3.z;
                    if (dd < SqRadii) {
                        j_num++;
                    }
                }
            }
        }
    }
    numNeighborsPerPart[index] = j_num;
}
__global__ void neighborSearchID(const Real4* sortedPosRad,
                                 const Real4* sortedRhoPreMu,
                                 const uint* cellStart,
                                 const uint* cellEnd,
                                 const uint numActive,
                                 const uint* numNeighborsPerPart,
                                 uint* neighborList) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numActive) {
        return;
    }
    Real3 posRadA = mR3(sortedPosRad[index]);
    int3 gridPos = calcGridPos(posRadA);
    Real SuppRadii = 2.0f * paramsD.h;
    Real SqRadii = SuppRadii * SuppRadii;
    uint j_num = 1;
    neighborList[numNeighborsPerPart[index]] = index;

    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int3 neighborPos = gridPos + mI3(x, y, z);
                // Check if we need to skip this neighbor position (out of bounds for non-periodic dimensions)
                if (neighborPos.x < paramsD.minBounds.x || neighborPos.x > paramsD.maxBounds.x || neighborPos.y < paramsD.minBounds.y || neighborPos.y > paramsD.maxBounds.y ||
                    neighborPos.z < paramsD.minBounds.z || neighborPos.z > paramsD.maxBounds.z) {
                    continue;
                }
                uint gridHash = calcGridHash(neighborPos);
                uint startIndex = cellStart[gridHash];
                uint endIndex = cellEnd[gridHash];
                for (uint j = startIndex; j < endIndex; j++) {
                    if (j != index) {
                        Real3 posRadB = mR3(sortedPosRad[j]);
                        Real3 dist3 = Distance(posRadA, posRadB);
                        Real dd = dist3.x * dist3.x + dist3.y * dist3.y + dist3.z * dist3.z;
                        if (dd < SqRadii) {
                            neighborList[numNeighborsPerPart[index] + j_num] = j;
                            j_num++;
                        }
                    }
                }
            }
        }
    }
}


#define UINT_MAX 4294967295u
__global__ void findCellStartEndD(uint* cellStartD,        // output: cell start index
                                  uint* cellEndD,          // output: cell end index
                                  uint* gridMarkerHashD,   // input: sorted grid hashes
                                  uint* gridMarkerIndexD,  // input: sorted particle indices
                                  uint numActive) {
    extern __shared__ uint sharedHash[];  // blockSize + 1 elements
    // Get the particle index the current thread is supposed to be looking at.
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    uint hash;

    if (index >= numActive)
        return;

    hash = gridMarkerHashD[index];
    if (hash == UINT_MAX)
        return;
    // Load hash data into shared memory so that we can look at neighboring
    // particle's hash value without loading two hash values per thread
    sharedHash[threadIdx.x + 1] = hash;

    // first thread in block must load neighbor particle hash
    if (index > 0 && threadIdx.x == 0)
        sharedHash[0] = gridMarkerHashD[index - 1];

    __syncthreads();
    if (sharedHash[threadIdx.x] == UINT_MAX)
        return;

    // If this particle has a different cell index to the previous
    // particle then it must be the first particle in the cell,
    // so store the index of this particle in the cell. As it
    // isn't the first particle, it must also be the cell end of
    // the previous particle's cell.
    if (index == 0 || hash != sharedHash[threadIdx.x]) {
        cellStartD[hash] = index;
        if (index > 0)
            cellEndD[sharedHash[threadIdx.x]] = index;
    }

    if (index == numActive - 1)
        cellEndD[hash] = index + 1;
}

// MIT data preparation wrappers; distance and search code above is unchanged.
__global__ void hashMarkers(const Real4* positions,uint* hashes,uint* indices,uint n){uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){hashes[i]=calcGridHash(calcGridPos(mR3(positions[i])));indices[i]=i;}}
__global__ void gatherMarkers(const Real4* positions,const uint* indices,Real4* sortedPosRad,uint n){uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)sortedPosRad[i]=positions[indices[i]];}
// MIT selection adapter: preserve original marker IDs through the existing grid helpers.
__global__ void hashSelected(const Real4* positions,const uint* activeList,uint* hashes,uint* indices,uint n){uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){uint original=activeList[i];hashes[i]=calcGridHash(calcGridPos(mR3(positions[original])));indices[i]=original;}}
