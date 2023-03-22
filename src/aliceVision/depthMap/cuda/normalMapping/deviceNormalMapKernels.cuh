// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <aliceVision/mvsData/ROI.hpp>
#include <aliceVision/depthMap/cuda/device/buffer.cuh>
#include <aliceVision/depthMap/cuda/device/matrix.cuh>
#include <aliceVision/depthMap/cuda/device/eig33.cuh>

#include <math_constants.h>

namespace aliceVision {
namespace depthMap {

__device__ static inline
float3 get3DPointForPixelAndDepthFromRC(const DeviceCameraParams& rcDeviceCamParams, const float2& pix, float depth)
{
    float3 rpv = M3x3mulV2(rcDeviceCamParams.iP, pix);
    normalize(rpv);
    return rcDeviceCamParams.C + rpv * depth;
}

__device__ static inline
float3 get3DPointForPixelAndDepthFromRC(const DeviceCameraParams& rcDeviceCamParams, const int2& pixi, float depth)
{
    float2 pix;
    pix.x = float(pixi.x);
    pix.y = float(pixi.y);
    return get3DPointForPixelAndDepthFromRC(rcDeviceCamParams, pix, depth);
}

__device__ static inline
float orientedPointPlaneDistanceNormalizedNormal(const float3& point, const float3& planePoint,
                                                 const float3& planeNormalNormalized)
{
    return (dot(point, planeNormalNormalized) - dot(planePoint, planeNormalNormalized));
}

__global__ void computeNormalMap_kernel(const DeviceCameraParams& rcDeviceCamParams,
                                        float* depthMap_d, int depthMap_p, //cudaTextureObject_t depthsTex,
                                        float3* nmap_d, int nmap_p,
                                        int width, int height, int wsh, const float gammaC, const float gammaP)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x >= width) || (y >= height))
    return;

  float depth = *get2DBufferAt<float>(depthMap_d, depthMap_p, x, y); // tex2D<float>(depthsTex, x, y);
  if(depth <= 0.0f)
  {
    *get2DBufferAt(nmap_d, nmap_p, x, y) = make_float3(-1.f, -1.f, -1.f);
    return;
  }

  int2 pix1 = make_int2(x, y);
  float3 p = get3DPointForPixelAndDepthFromRC(rcDeviceCamParams, pix1, depth);
  float pixSize = 0.0f;
  {
    int2 pix2 = make_int2(x + 1, y);
    float3 p2 = get3DPointForPixelAndDepthFromRC(rcDeviceCamParams, pix2, depth);
    pixSize = size(p - p2);
  }

  cuda_stat3d s3d = cuda_stat3d();

  for (int yp = -wsh; yp <= wsh; ++yp)
  {
    for (int xp = -wsh; xp <= wsh; ++xp)
    {
      float depthn = *get2DBufferAt<float>(depthMap_d, depthMap_p, x + xp, y + yp); // tex2D<float>(depthsTex, x + xp, y + yp);
      if ((depth > 0.0f) && (fabs(depthn - depth) < 30.0f * pixSize))
      {
        float w = 1.0f;
        float2 pixn = make_float2(x + xp, y + yp);
        float3 pn = get3DPointForPixelAndDepthFromRC(rcDeviceCamParams, pixn, depthn);
        s3d.update(pn, w);
      }
    }
  }

  float3 pp = p;
  float3 nn = make_float3(-1.f, -1.f, -1.f);
  if(!s3d.computePlaneByPCA(pp, nn))
  {
    *get2DBufferAt(nmap_d, nmap_p, x, y) = make_float3(-1.f, -1.f, -1.f);
    return;
  }

  float3 nc = rcDeviceCamParams.C - p;
  normalize(nc);
  if (orientedPointPlaneDistanceNormalizedNormal(pp + nn, pp, nc) < 0.0f)
  {
    nn.x = -nn.x;
    nn.y = -nn.y;
    nn.z = -nn.z;
  }
  *get2DBufferAt(nmap_d, nmap_p, x, y) = nn;
}

} // namespace depthMap
} // namespace aliceVision

