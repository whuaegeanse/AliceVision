// This file is part of the AliceVision project.
// Copyright (c) 2022 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "deviceDepthSimilarityMap.hpp"
#include "deviceDepthSimilarityMapKernels.cuh"

#include <aliceVision/depthMap/cuda/host/divUp.hpp>

#include <utility>

namespace aliceVision {
namespace depthMap {

__host__ void cuda_depthSimMapCopyDepthOnly(CudaDeviceMemoryPitched<float2, 2>& out_depthSimMap_dmp,
                                            const CudaDeviceMemoryPitched<float2, 2>& in_depthSimMap_dmp,
                                            float defaultSim, 
                                            cudaStream_t stream)
{
    const CudaSize<2>& depthSimMapSize = out_depthSimMap_dmp.getSize();

    const int blockSize = 16;
    const dim3 block(blockSize, blockSize, 1);
    const dim3 grid(divUp(depthSimMapSize.x(), blockSize), divUp(depthSimMapSize.y(), blockSize), 1);

    depthSimMapCopyDepthOnly_kernel<<<grid, block, 0, stream>>>(
      out_depthSimMap_dmp.getBuffer(), 
      out_depthSimMap_dmp.getPitch(), 
      in_depthSimMap_dmp.getBuffer(), 
      in_depthSimMap_dmp.getPitch(),
      depthSimMapSize.x(),
      depthSimMapSize.y(),
      defaultSim);

    CHECK_CUDA_ERROR();
}

__host__ void cuda_normalMapUpscale(CudaDeviceMemoryPitched<float3, 2>& out_upscaledMap_dmp,
                                    const CudaDeviceMemoryPitched<float3, 2>& in_map_dmp,
                                    const ROI& roi,
                                    cudaStream_t stream)
{
    const CudaSize<2>& out_mapSize = out_upscaledMap_dmp.getSize();
    const CudaSize<2>& in_mapSize = in_map_dmp.getSize();

    const float ratio = float(in_mapSize.x()) / float(out_mapSize.x());

    const int blockSize = 16;
    const dim3 block(blockSize, blockSize, 1);
    const dim3 grid(divUp(roi.width(), blockSize), divUp(roi.height(), blockSize), 1);

    mapUpscale_kernel<float3><<<grid, block, 0, stream>>>(
      out_upscaledMap_dmp.getBuffer(),
      out_upscaledMap_dmp.getPitch(),
      in_map_dmp.getBuffer(),
      in_map_dmp.getPitch(),
      roi,
      ratio);

    CHECK_CUDA_ERROR();
}

__host__ void cuda_depthSimMapUpscaleAndFilter(CudaDeviceMemoryPitched<float2, 2>& out_upscaledDepthSimMap_dmp,
                                               const CudaDeviceMemoryPitched<float2, 2>& in_otherDepthSimMap_dmp,
                                               const DeviceCamera& rcDeviceCamera,
                                               const RefineParams& refineParams,
                                               const ROI& roi,
                                               cudaStream_t stream)
{
    const CudaSize<2>& out_depthSimMapSize = out_upscaledDepthSimMap_dmp.getSize();
    const CudaSize<2>& in_depthSimMapSize = in_otherDepthSimMap_dmp.getSize();

    const float ratio = float(in_depthSimMapSize.x()) / float(out_depthSimMapSize.x());

    const int blockSize = 16;
    const dim3 block(blockSize, blockSize, 1);
    const dim3 grid(divUp(roi.width(), blockSize), divUp(roi.height(), blockSize), 1);

    depthSimMapUpscaleAndFilter_kernel<<<grid, block, 0, stream>>>(
      rcDeviceCamera.getTextureObject(),
      out_upscaledDepthSimMap_dmp.getBuffer(), 
      out_upscaledDepthSimMap_dmp.getPitch(),
      in_otherDepthSimMap_dmp.getBuffer(), 
      in_otherDepthSimMap_dmp.getPitch(),
      refineParams.stepXY,
      roi,
      ratio);

    CHECK_CUDA_ERROR();
}

__host__ void cuda_depthSimMapComputePixSize(CudaDeviceMemoryPitched<float2, 2>& inout_depthPixSizeMap_dmp,
                                             const DeviceCamera& rcDeviceCamera,
                                             const RefineParams& refineParams,
                                             const ROI& roi,
                                             cudaStream_t stream)
{
    const int blockSize = 16;
    const dim3 block(blockSize, blockSize, 1);
    const dim3 grid(divUp(roi.width(), blockSize), divUp(roi.height(), blockSize), 1);

    depthSimMapComputePixSize_kernel<<<grid, block, 0, stream>>>(
      rcDeviceCamera.getDeviceCamId(), 
      inout_depthPixSizeMap_dmp.getBuffer(), 
      inout_depthPixSizeMap_dmp.getPitch(),
      refineParams.stepXY,
      roi);

    CHECK_CUDA_ERROR();
}

__host__ void cuda_depthSimMapComputeNormal(CudaDeviceMemoryPitched<float3, 2>& out_normalMap_dmp,
                                            const CudaDeviceMemoryPitched<float2, 2>& in_depthSimMap_dmp,
                                            const DeviceCamera& rcDeviceCamera, 
                                            const SgmParams& sgmParams,
                                            const ROI& roi,
                                            cudaStream_t stream)
{
    // default parameters
    const int wsh = 4;
    const float gammaC = 1.0f;
    const float gammaP = 1.0f;

    const dim3 block(8, 8, 1);
    const dim3 grid(divUp(roi.width(), block.x), divUp(roi.height(), block.y), 1);

    depthSimMapComputeNormal_kernel<<<grid, block, 0, stream>>>(
      rcDeviceCamera.getDeviceCamId(),
      out_normalMap_dmp.getBuffer(),
      out_normalMap_dmp.getPitch(),
      in_depthSimMap_dmp.getBuffer(),
      in_depthSimMap_dmp.getPitch(),
      wsh,
      gammaC,
      gammaP,
      sgmParams.stepXY,
      roi);

    CHECK_CUDA_ERROR();
}

__host__ void cuda_depthSimMapOptimizeGradientDescent(CudaDeviceMemoryPitched<float2, 2>& out_optimizeDepthSimMap_dmp,
                                                      CudaDeviceMemoryPitched<float, 2>& inout_imgVariance_dmp,
                                                      CudaDeviceMemoryPitched<float, 2>& inout_tmpOptDepthMap_dmp,
                                                      const CudaDeviceMemoryPitched<float2, 2>& in_sgmDepthPixSizeMap_dmp,
                                                      const CudaDeviceMemoryPitched<float2, 2>& in_refineDepthSimMap_dmp,
                                                      const DeviceCamera& rcDeviceCamera, 
                                                      const RefineParams& refineParams,
                                                      const ROI& roi,
                                                      cudaStream_t stream)
{
    // initialize depth/sim map optimized with SGM depth/pixSize map
    out_optimizeDepthSimMap_dmp.copyFrom(in_sgmDepthPixSizeMap_dmp, stream);

    {
        // setup block and grid
        const dim3 lblock(32, 2, 1);
        const dim3 lgrid(divUp(roi.width(), lblock.x), divUp(roi.height(), lblock.y), 1);

        optimize_varLofLABtoW_kernel<<<lgrid, lblock, 0, stream>>>(
            rcDeviceCamera.getTextureObject(), 
            inout_imgVariance_dmp.getBuffer(), 
            inout_imgVariance_dmp.getPitch(),
            refineParams.stepXY,
            roi);
    }

    CudaTexture<float> imgVarianceTex(inout_imgVariance_dmp);
    CudaTexture<float> depthTex(inout_tmpOptDepthMap_dmp);

    // setup block and grid
    const int blockSize = 16;
    const dim3 block(blockSize, blockSize, 1);
    const dim3 grid(divUp(roi.width(), blockSize), divUp(roi.height(), blockSize), 1);

    for(int iter = 0; iter < refineParams.optimizationNbIterations; ++iter) // default nb iterations is 100
    {
        // copy depths values from out_depthSimMapOptimized_dmp to inout_tmpOptDepthMap_dmp
        optimize_getOptDeptMapFromOptDepthSimMap_kernel<<<grid, block, 0, stream>>>(
            inout_tmpOptDepthMap_dmp.getBuffer(), 
            inout_tmpOptDepthMap_dmp.getPitch(), 
            out_optimizeDepthSimMap_dmp.getBuffer(), // initialized with SGM depth/sim map
            out_optimizeDepthSimMap_dmp.getPitch(),
            roi);

        // adjust depth/sim by using previously computed depths
        optimize_depthSimMap_kernel<<<grid, block, 0, stream>>>(
            rcDeviceCamera.getDeviceCamId(), 
            imgVarianceTex.textureObj,
            depthTex.textureObj, 
            out_optimizeDepthSimMap_dmp.getBuffer(),
            out_optimizeDepthSimMap_dmp.getPitch(),
            in_sgmDepthPixSizeMap_dmp.getBuffer(),
            in_sgmDepthPixSizeMap_dmp.getPitch(),
            in_refineDepthSimMap_dmp.getBuffer(),
            in_refineDepthSimMap_dmp.getPitch(),
            iter, 
            roi);
    }

    CHECK_CUDA_ERROR();
}

} // namespace depthMap
} // namespace aliceVision
