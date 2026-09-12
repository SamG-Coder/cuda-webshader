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
// Author: Milad Rakhsha, Arman Pazouki, Wei Hu, Radu Serban
// =============================================================================
//

// Original Counters declaration from Chrono a92c6f72f422fbcafe0b37125d4070cb6a3b5803.
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

// MIT validation probe: exercise the original record with exact launch-time words.
__global__ void counterTimeProbe(unsigned int* output, double time) {
    output[0] = (unsigned int)countersD.numFsiBodies;
    output[1] = (unsigned int)(countersD.numFsiBodies >> 32);
    output[2] = countersD.numFsiBodies > 4294967295u;
    output[3] = countersD.numFsiBodies < -1;
    output[4] = (unsigned int)countersD.numFsiMeshes1D;
    output[5] = (unsigned int)(countersD.numFsiMeshes1D >> 32);
    output[6] = countersD.numFsiMeshes1D > 4294967295u;
    output[7] = countersD.numFsiMeshes1D < -1;
    output[8] = (unsigned int)countersD.numFsiMeshes2D;
    output[9] = (unsigned int)(countersD.numFsiMeshes2D >> 32);
    output[10] = countersD.numFsiMeshes2D > 4294967295u;
    output[11] = countersD.numFsiMeshes2D < -1;
    output[12] = (unsigned int)countersD.numFsiNodes1D;
    output[13] = (unsigned int)(countersD.numFsiNodes1D >> 32);
    output[14] = countersD.numFsiNodes1D > 4294967295u;
    output[15] = countersD.numFsiNodes1D < -1;
    output[16] = (unsigned int)countersD.numFsiNodes2D;
    output[17] = (unsigned int)(countersD.numFsiNodes2D >> 32);
    output[18] = countersD.numFsiNodes2D > 4294967295u;
    output[19] = countersD.numFsiNodes2D < -1;
    output[20] = (unsigned int)countersD.numFsiElements1D;
    output[21] = (unsigned int)(countersD.numFsiElements1D >> 32);
    output[22] = countersD.numFsiElements1D > 4294967295u;
    output[23] = countersD.numFsiElements1D < -1;
    output[24] = (unsigned int)countersD.numFsiElements2D;
    output[25] = (unsigned int)(countersD.numFsiElements2D >> 32);
    output[26] = countersD.numFsiElements2D > 4294967295u;
    output[27] = countersD.numFsiElements2D < -1;
    output[28] = (unsigned int)countersD.numGhostMarkers;
    output[29] = (unsigned int)(countersD.numGhostMarkers >> 32);
    output[30] = countersD.numGhostMarkers > 4294967295u;
    output[31] = countersD.numGhostMarkers < -1;
    output[32] = (unsigned int)countersD.numHelperMarkers;
    output[33] = (unsigned int)(countersD.numHelperMarkers >> 32);
    output[34] = countersD.numHelperMarkers > 4294967295u;
    output[35] = countersD.numHelperMarkers < -1;
    output[36] = (unsigned int)countersD.numFluidMarkers;
    output[37] = (unsigned int)(countersD.numFluidMarkers >> 32);
    output[38] = countersD.numFluidMarkers > 4294967295u;
    output[39] = countersD.numFluidMarkers < -1;
    output[40] = (unsigned int)countersD.numBoundaryMarkers;
    output[41] = (unsigned int)(countersD.numBoundaryMarkers >> 32);
    output[42] = countersD.numBoundaryMarkers > 4294967295u;
    output[43] = countersD.numBoundaryMarkers < -1;
    output[44] = (unsigned int)countersD.numRigidMarkers;
    output[45] = (unsigned int)(countersD.numRigidMarkers >> 32);
    output[46] = countersD.numRigidMarkers > 4294967295u;
    output[47] = countersD.numRigidMarkers < -1;
    output[48] = (unsigned int)countersD.numMesh1DMarkers;
    output[49] = (unsigned int)(countersD.numMesh1DMarkers >> 32);
    output[50] = countersD.numMesh1DMarkers > 4294967295u;
    output[51] = countersD.numMesh1DMarkers < -1;
    output[52] = (unsigned int)countersD.numMesh2DMarkers;
    output[53] = (unsigned int)(countersD.numMesh2DMarkers >> 32);
    output[54] = countersD.numMesh2DMarkers > 4294967295u;
    output[55] = countersD.numMesh2DMarkers < -1;
    output[56] = (unsigned int)countersD.numBceMarkers;
    output[57] = (unsigned int)(countersD.numBceMarkers >> 32);
    output[58] = countersD.numBceMarkers > 4294967295u;
    output[59] = countersD.numBceMarkers < -1;
    output[60] = (unsigned int)countersD.numAllMarkers;
    output[61] = (unsigned int)(countersD.numAllMarkers >> 32);
    output[62] = countersD.numAllMarkers > 4294967295u;
    output[63] = countersD.numAllMarkers < -1;
    output[64] = (unsigned int)countersD.startBoundaryMarkers;
    output[65] = (unsigned int)(countersD.startBoundaryMarkers >> 32);
    output[66] = countersD.startBoundaryMarkers > 4294967295u;
    output[67] = countersD.startBoundaryMarkers < -1;
    output[68] = (unsigned int)countersD.startRigidMarkers;
    output[69] = (unsigned int)(countersD.startRigidMarkers >> 32);
    output[70] = countersD.startRigidMarkers > 4294967295u;
    output[71] = countersD.startRigidMarkers < -1;
    output[72] = (unsigned int)countersD.startMesh1DMarkers;
    output[73] = (unsigned int)(countersD.startMesh1DMarkers >> 32);
    output[74] = countersD.startMesh1DMarkers > 4294967295u;
    output[75] = countersD.startMesh1DMarkers < -1;
    output[76] = (unsigned int)countersD.startMesh2DMarkers;
    output[77] = (unsigned int)(countersD.startMesh2DMarkers >> 32);
    output[78] = countersD.startMesh2DMarkers > 4294967295u;
    output[79] = countersD.startMesh2DMarkers < -1;
    output[80] = (unsigned int)countersD.numActiveParticles;
    output[81] = (unsigned int)(countersD.numActiveParticles >> 32);
    output[82] = countersD.numActiveParticles > 4294967295u;
    output[83] = countersD.numActiveParticles < -1;
    output[84] = (unsigned int)countersD.numExtendedParticles;
    output[85] = (unsigned int)(countersD.numExtendedParticles >> 32);
    output[86] = countersD.numExtendedParticles > 4294967295u;
    output[87] = countersD.numExtendedParticles < -1;
    output[88] = time < 0.5;
    output[89] = time >= 0.5;
    output[90] = time == time;
    output[91] = time < 0.0;
    output[92] = time == 0.0;
    output[93] = (time - 0.5) == 0.0000000000009094947017729282379150390625;
    output[94] = (time - 0.5) == -0.0000000000009094947017729282379150390625;
    output[95] = time > 0.0;
}
