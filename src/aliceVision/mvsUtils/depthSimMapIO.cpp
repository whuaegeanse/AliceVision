// This file is part of the AliceVision project.
// Copyright (c) 2022 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "depthSimMapIO.hpp"

#include <aliceVision/mvsData/Point2d.hpp>
#include <aliceVision/mvsUtils/fileIO.hpp>
#include <aliceVision/numeric/numeric.hpp>
#include <aliceVision/image/io.hpp>

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

namespace fs = boost::filesystem;

namespace aliceVision {
namespace mvsUtils {

/**
 * @brief Get tile map ROI from file metadata
 * @param[in] mapTilePath the tile map file path
 * @param[in,out] out_roi the corresponding region-of-interest read from file metadata
 */
void getRoiFromMetadata(const std::string& mapTilePath, ROI& out_roi)
{
    const oiio::ParamValueList metadata = image::readImageMetadata(mapTilePath);

    const auto roiBeginXIt = metadata.find("AliceVision:roiBeginX");
    const auto roiBeginYIt = metadata.find("AliceVision:roiBeginY");
    const auto roiEndXIt  = metadata.find("AliceVision:roiEndX");
    const auto roiEndYIt  = metadata.find("AliceVision:roiEndY");

    if(roiBeginXIt != metadata.end() && roiBeginXIt->type() == oiio::TypeDesc::INT)
        out_roi.x.begin = roiBeginXIt->get_int();

    if(roiBeginYIt != metadata.end() && roiBeginYIt->type() == oiio::TypeDesc::INT)
        out_roi.y.begin = roiBeginYIt->get_int();

    if(roiEndXIt != metadata.end() && roiEndXIt->type() == oiio::TypeDesc::INT)
        out_roi.x.end = roiEndXIt->get_int();

    if(roiEndYIt != metadata.end() && roiEndYIt->type() == oiio::TypeDesc::INT)
        out_roi.y.end = roiEndYIt->get_int();

    // invalid or no roi metadata
    if((out_roi.x.begin < 0) || (out_roi.y.begin < 0) || (out_roi.x.end <= 0) || (out_roi.y.end <= 0))
    {
        ALICEVISION_THROW_ERROR("Cannot find ROI information in file: " << mapTilePath);
    }
}

/**
 * @brief Get tile map TileParams from file metadata
 * @param[in] mapTilePath the tile map file path
 * @param[in,out] out_tileParams the corresponding TileParams read from file metadata
 */
void getTileParamsFromMetadata(const std::string& mapTilePath, TileParams& out_tileParams)
{
    const oiio::ParamValueList metadata = image::readImageMetadata(mapTilePath);

    const auto tileWidthIt   = metadata.find("AliceVision:tileBufferWidth");
    const auto tileHeightIt  = metadata.find("AliceVision:tileBufferHeight");
    const auto tilePaddingIt = metadata.find("AliceVision:tilePadding");

    if(tileWidthIt != metadata.end() && tileWidthIt->type() == oiio::TypeDesc::INT)
        out_tileParams.bufferWidth = tileWidthIt->get_int();

    if(tileHeightIt != metadata.end() && tileHeightIt->type() == oiio::TypeDesc::INT)
        out_tileParams.bufferHeight = tileHeightIt->get_int();

    if(tilePaddingIt != metadata.end() && tilePaddingIt->type() == oiio::TypeDesc::INT)
        out_tileParams.padding = tilePaddingIt->get_int();

    // invalid or no tile metadata
    if((out_tileParams.bufferWidth <= 0) || (out_tileParams.bufferHeight <= 0) || (out_tileParams.padding < 0))
    {
        ALICEVISION_THROW_ERROR("Cannot find tile parameters in file: " << mapTilePath);
    }
}

/**
 * @brief Get the tile map path list for a R camera et a given scale / stepXY
 * @param[in] rc the related R camera index
 * @param[in] mp the multi-view parameters
 * @param[in] scale the depth/sim map downscale factor
 * @param[in] step the depth/sim map step factor
 * @param[in] customSuffix the filename custom suffix
 * @param[in,out] out_mapTilePathList the tile map path list
 */
void getTilePathList(int rc,
                     const MultiViewParams& mp,
                     EFileType fileType,
                     int scale,
                     int step,
                     const std::string& customSuffix,
                     std::vector<std::string>& out_mapTilePathList)
{
  const fs::path mapPath(getFileNameFromIndex(mp, rc, fileType, scale, customSuffix));
  const fs::path mapDirectory(mapPath.parent_path());

  if(!is_directory(mapDirectory))
    ALICEVISION_THROW_ERROR("Cannot find depth/similarity map directory (rc: " << rc << ").");

  const boost::regex mapPattern(mapPath.stem().string() + "_\\d+_\\d+" + mapPath.extension().string());

  for(auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(mapDirectory), {}))
  {
      if(boost::regex_match(entry.path().filename().string(), mapPattern))
        out_mapTilePathList.push_back(entry.path().string());
  }
}

/**
 * @brief Weight one of the corners/edges of a tile according to the size of the padding
 *
 * When merging tiles, there are 8 intersection areas:
 *  * 4 corners (intersection of 4 tiles or 2 tiles when the tile is on one image edge)
 *  * 4 edges (intersection of 2 tiles)
 *
 * @param a alpha for top-left
 * @param b alpha for top-right
 * @param c alpha for bottom-right
 * @param d alpha for bottom-left
 * @param borderWidth tiles intersection area width (could be the intersection between 2 or 4 tiles)
 * @param borderHeight tiles intersection area height
 * @param lu left-up corner of the intersection area in the tile coordinate system
 * @param in_tileMap image of the tile
 */
void weightTileBorder(int a, int b, int c, int d, 
                      int borderWidth, 
                      int borderHeight,
                      const Point2d& lu, 
                      image::Image<float>& in_tileMap)
{
    const Point2d rd = lu + Point2d(borderWidth, borderHeight);

    const int endX = std::min(int(rd.x), in_tileMap.Width());
    const int endY = std::min(int(rd.y), in_tileMap.Height());

    // Add small margin where alpha is 0 for corners (lu and rd)
    static const double margin = 2.0;
    const Point2d lu_m(lu.x + margin, lu.y + margin);
    const Point2d rd_m(rd.x - margin, rd.y - margin);
    const double borderWidth_m = borderWidth - 2.0 * margin;
    const double borderHeight_m = borderHeight - 2.0 * margin;

    for(int x = lu.x; x < endX; ++x)
    {
        for(int y = lu.y; y < endY; ++y)
        {
            // bilinear interpolation
            const float r_x = clamp((rd_m.x - x) / borderWidth_m, 0.0, 1.0);
            const float r_y = clamp((rd_m.y - y) / borderHeight_m, 0.0, 1.0);
            const float l_x = clamp((x - lu_m.x) / borderWidth_m, 0.0, 1.0);
            const float l_y = clamp((y - lu_m.y) / borderHeight_m, 0.0, 1.0);

            const float weight = r_y * (r_x * a + l_x * b) + l_y * (r_x * d + l_x * c);

            // apply weight to tile depth/sim map
            in_tileMap(y, x) *= weight;
        }
    }
}

void addTileMapWeighted(int rc,
                         const MultiViewParams& mp, 
                         const TileParams& tileParams,
                         const ROI& roi, 
                         int downscale,
                         image::Image<float>& in_tileMap,
                         image::Image<float>& inout_map)
{
    // get downscaled ROI
    const ROI downscaledRoi = downscaleROI(roi, downscale);

    // get tile border size
    const int tileWidth = downscaledRoi.width();
    const int tileHeight = downscaledRoi.height();
    const int tilePadding = tileParams.padding / downscale;

    // get tile position information
    const bool firstColumn = (roi.x.begin == 0);
    const bool lastColumn = (roi.x.end == mp.getWidth(rc));
    const bool firstRow = (roi.y.begin == 0);
    const bool lastRow = (roi.y.end == mp.getHeight(rc));

    // weight the top left corner
    if(!firstColumn || !firstRow)
    {
        const Point2d lu(0, 0);
        const int b = (firstRow) ? 1 : 0;
        const int d = (firstColumn) ? 1 : 0;
        weightTileBorder(0, b, 1, d, tilePadding, tilePadding, lu, in_tileMap);
    }

    // weight the bottom left corner
    if(!firstColumn || !lastRow)
    {
        const Point2d lu(0, tileHeight - tilePadding);
        const int a = (firstColumn) ? 1 : 0;
        const int c = (lastRow) ? 1 : 0;
        weightTileBorder(a, 1, c, 0, tilePadding, tilePadding, lu, in_tileMap);
    }

    // weight the top right corner
    if(!lastColumn || !firstRow)
    {
        const Point2d lu(tileWidth - tilePadding, 0);
        const int a = (firstRow) ? 1 : 0;
        const int c = (lastColumn) ? 1 : 0;
        weightTileBorder(a, 0, c, 1, tilePadding, tilePadding, lu, in_tileMap);
    }

    // weight the bottom right corner
    if(!lastColumn || !lastRow)
    {
        const Point2d lu(tileWidth - tilePadding, tileHeight - tilePadding);
        const int b = (lastColumn) ? 1 : 0;
        const int d = (lastRow) ? 1 : 0;
        weightTileBorder(1, b, 0, d, tilePadding, tilePadding, lu, in_tileMap);
    }

    // weight the top border
    if(!firstRow)
    {
        const Point2d lu(tilePadding, 0);
        weightTileBorder(0, 0, 1, 1, tileWidth - 2 * tilePadding, tilePadding, lu, in_tileMap);
    }

    // weight the bottom border
    if(!lastRow)
    {
        const Point2d lu(tilePadding, tileHeight - tilePadding);
        weightTileBorder(1, 1, 0, 0, tileWidth - 2 * tilePadding, tilePadding, lu, in_tileMap);
    }

    // weight the left border
    if(!firstColumn)
    {
        const Point2d lu(0, tilePadding);
        weightTileBorder(0, 1, 1, 0, tilePadding, tileHeight - 2 * tilePadding, lu, in_tileMap);
    }

    // weight the right border
    if(!lastColumn)
    {
        const Point2d lu(tileWidth - tilePadding, tilePadding);
        weightTileBorder(1, 0, 0, 1, tilePadding, tileHeight - 2 * tilePadding, lu, in_tileMap);
    }

    // add weighted tile to the depth/sim map
    for(int x = downscaledRoi.x.begin; x < downscaledRoi.x.end; ++x)
    {
        for(int y = downscaledRoi.y.begin; y < downscaledRoi.y.end; ++y)
        {
            const int tx = x - downscaledRoi.x.begin;
            const int ty = y - downscaledRoi.y.begin;

            inout_map(y, x) += in_tileMap(ty, tx);
        }
    }
}

void readMapFromTiles(int rc, 
                      const MultiViewParams& mp, 
                      EFileType fileType,
                      image::Image<float>& out_map, 
                      int scale,
                      int step, 
                      const std::string& customSuffix)
{
    const ROI imageRoi(Range(0, mp.getWidth(rc)), Range(0, mp.getHeight(rc)));

    const int scaleStep = std::max(scale, 1) * step; // avoid 0 special case (reserved for depth map filtering)
    const int width  = divideRoundUp(mp.getWidth(rc) , scaleStep);
    const int height = divideRoundUp(mp.getHeight(rc), scaleStep);

    // the output full map
    out_map.resize(width, height, true, 0.f); // should be initialized, additive process

    // get tile map path list for the given R camera
    std::vector<std::string> mapTilePathList;
    getTilePathList(rc, mp, fileType, scale, step, customSuffix, mapTilePathList);

    if(mapTilePathList.empty())
    {
      // map can be empty
      ALICEVISION_LOG_INFO("Cannot find any map tile file (rc: " << rc << ").");
      return; // nothing to do, already initialized
    }

    // get tileParams from first tile file metadata
    TileParams tileParams;
    getTileParamsFromMetadata(mapTilePathList.front(), tileParams);

    // get tile roi list from each file metadata
    std::vector<ROI> tileRoiList;
    tileRoiList.resize(mapTilePathList.size());
    for(size_t i = 0; i < mapTilePathList.size(); ++i)
    {
      getRoiFromMetadata(mapTilePathList.at(i), tileRoiList.at(i));
    }

    // read and add each tile to the full map
    for(size_t i = 0; i < tileRoiList.size(); ++i)
    {
        const ROI roi = intersect(tileRoiList.at(i), imageRoi);
        const std::string mapTilePath = getFileNameFromIndex(mp, rc, fileType, scale, customSuffix, roi.x.begin, roi.y.begin);

        if(roi.isEmpty())
            continue;

        try
        {
            // read tile
            image::Image<float> tileMap;
            image::readImage(mapTilePath, tileMap, image::EImageColorSpace::NO_CONVERSION);

            // add tile to the full map
            addTileMapWeighted(rc, mp, tileParams, roi, scaleStep, tileMap, out_map);
        }
        catch(const std::exception& e)
        {
            ALICEVISION_LOG_WARNING("Cannot find depth/sim map (rc: " << rc << "): " << mapTilePath);
        }
    }
}

void writeDepthSimMap(int rc, 
                      const MultiViewParams& mp, 
                      const TileParams& tileParams, 
                      const ROI& roi,
                      const image::Image<float>& depthMap, 
                      const image::Image<float>& simMap, 
                      int scale,
                      int step,
                      const std::string& customSuffix)
{
    const int scaleStep = std::max(scale, 1) * step; // avoid 0 special case (reserved for depth map filtering)

    // get image dimensions at scale / stepXY
    const int imageWidth  = divideRoundUp(mp.getWidth(rc) , scaleStep);
    const int imageHeight = divideRoundUp(mp.getHeight(rc), scaleStep);

    // get downscaled ROI
    const ROI downscaledROI = downscaleROI(roi, scaleStep);

    // OIIO roi for depth / similarity map writing
    // displayRoi is the image region of interest for display (image size)
    // pixelRoi is the buffer region of interest within the displayRoi (tile size)
    // no tiling if displayRoi == pixelRoi
    const oiio::ROI displayRoi(0, imageWidth, 0, imageHeight);
    const oiio::ROI pixelRoi(downscaledROI.x.begin, downscaledROI.x.end, downscaledROI.y.begin, downscaledROI.y.end, 0, 1, 0, 1);

    // output map path
    std::string depthMapPath;
    std::string simMapPath;

    if(downscaledROI.width() != imageWidth || downscaledROI.height() != imageHeight) // is a tile
    {
        // tiled depth/sim map
        depthMapPath = getFileNameFromIndex(mp, rc, EFileType::depthMap, scale, customSuffix, roi.x.begin, roi.y.begin);
        simMapPath = getFileNameFromIndex(mp, rc, EFileType::simMap, scale, customSuffix, roi.x.begin, roi.y.begin);
    }
    else
    {
        // fullsize depth/sim map
        depthMapPath = getFileNameFromIndex(mp, rc, EFileType::depthMap, scale, customSuffix);
        simMapPath = getFileNameFromIndex(mp, rc, EFileType::simMap, scale, customSuffix);
    }

    oiio::ParamValueList metadata = image::getMetadataFromMap(mp.getMetadata(rc));

    // downscale metadata
    metadata.push_back(oiio::ParamValue("AliceVision:downscale", mp.getDownscaleFactor(rc) * scaleStep));

    // roi metadata
    {
      metadata.push_back(oiio::ParamValue("AliceVision:roiBeginX", int(roi.x.begin)));
      metadata.push_back(oiio::ParamValue("AliceVision:roiBeginY", int(roi.y.begin)));
      metadata.push_back(oiio::ParamValue("AliceVision:roiEndX",   int(roi.x.end)));
      metadata.push_back(oiio::ParamValue("AliceVision:roiEndY",   int(roi.y.end)));
    }

    // tile params metadata
    {
        metadata.push_back(oiio::ParamValue("AliceVision:tileBufferWidth",  tileParams.bufferWidth));
        metadata.push_back(oiio::ParamValue("AliceVision:tileBufferHeight", tileParams.bufferHeight));
        metadata.push_back(oiio::ParamValue("AliceVision:tilePadding",      tileParams.padding));
    }

    // projection matrix metadata
    {
        std::vector<double> matrixP = mp.getOriginalP(rc);
        metadata.push_back(oiio::ParamValue("AliceVision:P", oiio::TypeDesc(oiio::TypeDesc::DOUBLE, oiio::TypeDesc::MATRIX44), 1, matrixP.data()));
    }

    // CArr & iCamArr metadata
    {
      Point3d C = mp.CArr[rc];
      Matrix3x3 iP = mp.iCamArr[rc];

      if (scaleStep > 1)
      {
          Matrix3x4 P = mp.camArr[rc];
          for (int i = 0; i < 8; ++i)
              P.m[i] /= double(scaleStep);
          Matrix3x3 K, iK;
          Matrix3x3 R, iR;

          P.decomposeProjectionMatrix(K, R, C); // replace C
          iK = K.inverse();
          iR = R.inverse();
          iP = iR * iK; // replace iP
      }

      metadata.push_back(oiio::ParamValue("AliceVision:CArr", oiio::TypeDesc(oiio::TypeDesc::DOUBLE, oiio::TypeDesc::VEC3), 1, C.m));
      metadata.push_back(oiio::ParamValue("AliceVision:iCamArr", oiio::TypeDesc(oiio::TypeDesc::DOUBLE, oiio::TypeDesc::MATRIX33), 1, iP.m));
    }

    // min/max/nb depth metadata
    { 
        const int nbDepthValues = std::count_if(depthMap.data(), depthMap.data() + depthMap.size(), [](float v) { return v > 0.0f; });
        float maxDepth = -1.0f;
        float minDepth = std::numeric_limits<float>::max();

        for(int i = 0; i < depthMap.size(); ++i)
        {
            const float depth = depthMap(i);

            if(depth <= -1.0f)
                continue;

            maxDepth = std::max(maxDepth, depth);
            minDepth = std::min(minDepth, depth);
        }

        metadata.push_back(oiio::ParamValue("AliceVision:nbDepthValues", nbDepthValues));
        metadata.push_back(oiio::ParamValue("AliceVision:minDepth", minDepth));
        metadata.push_back(oiio::ParamValue("AliceVision:maxDepth", maxDepth));
    }

    // write depth map
    if(!depthMap.size() <= 0)
    {
        image::writeImage(depthMapPath, 
                          depthMap,
                          image::ImageWriteOptions()
                              .toColorSpace(image::EImageColorSpace::NO_CONVERSION)
                              .storageDataType(image::EStorageDataType::Float),
                          metadata, 
                          displayRoi,
                          pixelRoi);
    }

    // write sim map
    if(!simMap.size() <= 0)
    {
        image::writeImage(simMapPath,  
                          simMap,
                          image::ImageWriteOptions()
                              .toColorSpace(image::EImageColorSpace::NO_CONVERSION)
                              .storageDataType(image::EStorageDataType::Half),
                          metadata, 
                          displayRoi,
                          pixelRoi);
    }
}

void writeDepthSimMap(int rc, 
                      const MultiViewParams& mp,
                      const image::Image<float>& depthMap, 
                      const image::Image<float>& simMap, 
                      int scale,
                      int step,
                      const std::string& customSuffix)
{
    const TileParams tileParams; // default tile parameters, no tiles
    const ROI roi = ROI(0, mp.getWidth(rc), 0, mp.getHeight(rc)); // full roi
    writeDepthSimMap(rc, mp, tileParams, roi, depthMap, simMap, scale, step, customSuffix);
}


void writeDepthMap(int rc, 
                   const MultiViewParams& mp,
                   const image::Image<float>& depthMap, 
                   int scale,
                   int step,
                   const std::string& customSuffix)
{
    const TileParams tileParams;  // default tile parameters, no tiles
    const ROI roi = ROI(0, mp.getWidth(rc), 0, mp.getHeight(rc)); // full roi
    image::Image<float> simMap; // empty simMap, write only depth map
    writeDepthSimMap(rc, mp, tileParams, roi, depthMap, simMap, scale, step, customSuffix);
}

void readDepthSimMap(int rc, 
                     const MultiViewParams& mp,
                     image::Image<float>& out_depthMap, 
                     image::Image<float>& out_simMap, 
                     int scale,
                     int step, 
                     const std::string& customSuffix)
{
    const std::string depthMapPath = getFileNameFromIndex(mp, rc,EFileType::depthMap, scale, customSuffix);
    const std::string simMapPath = getFileNameFromIndex(mp, rc, EFileType::simMap, scale, customSuffix);

    if (fs::exists(depthMapPath) && fs::exists(simMapPath))
    {
        image::readImage(depthMapPath, out_depthMap, image::EImageColorSpace::NO_CONVERSION);
        image::readImage(simMapPath, out_simMap, image::EImageColorSpace::NO_CONVERSION);
    }
    else
    {
        readMapFromTiles(rc, mp, EFileType::depthMap, out_depthMap, scale, step, customSuffix);
        readMapFromTiles(rc, mp, EFileType::simMap, out_simMap, scale, step, customSuffix);
    }
}

void readDepthMap(int rc, 
                  const MultiViewParams& mp,
                  image::Image<float>& out_depthMap, 
                  int scale,
                  int step,
                  const std::string& customSuffix)
{
    const std::string depthMapPath = getFileNameFromIndex(mp, rc, EFileType::depthMap, scale, customSuffix);
        
    if (fs::exists(depthMapPath))
    {
        image::readImage(depthMapPath, out_depthMap, image::EImageColorSpace::NO_CONVERSION);
    }
    else
    {
        readMapFromTiles(rc, mp, EFileType::depthMap, out_depthMap, scale, step, customSuffix);
    }
}

void readSimMap(int rc, 
                const MultiViewParams& mp, 
                image::Image<float>& out_simMap, 
                int scale, 
                int step, 
                const std::string& customSuffix)
{
    const std::string simMapPath = getFileNameFromIndex(mp, rc, EFileType::simMap, scale, customSuffix);

    if (fs::exists(simMapPath))
    {
        image::readImage(simMapPath, out_simMap, image::EImageColorSpace::NO_CONVERSION);
    }
    else
    {
        readMapFromTiles(rc, mp, EFileType::simMap, out_simMap, scale, step, customSuffix);
    }
}

unsigned long getNbDepthValuesFromDepthMap(int rc, 
                                           const MultiViewParams& mp,
                                           int scale,
                                           int step,
                                           const std::string& customSuffix)
{
    const std::string depthMapPath = getFileNameFromIndex(mp, rc, EFileType::depthMap, scale, customSuffix);
    int nbDepthValues = -1;

    // get nbDepthValues from metadata
    if (fs::exists(depthMapPath)) // untilled
    {
        const oiio::ParamValueList metadata = image::readImageMetadata(depthMapPath);
        nbDepthValues = metadata.get_int("AliceVision:nbDepthValues", -1);
    }
    else // tilled
    {
        std::vector<std::string> mapTilePathList;
        getTilePathList(rc, mp, EFileType::depthMap, scale, step, customSuffix, mapTilePathList);

        if(mapTilePathList.empty()) // depth map can be empty
          ALICEVISION_LOG_INFO("Cannot find any depth map tile file (rc: " << rc << ").");

        for(const std::string& mapTilePath : mapTilePathList)
        {
            const oiio::ParamValueList metadata = image::readImageMetadata(mapTilePath);

            const int nbTileDepthValues = metadata.get_int("AliceVision:nbDepthValues", -1);

            if(nbTileDepthValues < 0)
                ALICEVISION_THROW_ERROR("Cannot find or incorrect 'AliceVision:nbDepthValues' metadata in depth map tile (rc: " << rc << ")");

            nbDepthValues += nbTileDepthValues;
        }
    }

    // no metadata compute number of depth values
    if(nbDepthValues < 0)
    {
        image::Image<float> depthMap;

        ALICEVISION_LOG_WARNING("Can't find or invalid 'nbDepthValues' metadata in depth map (rc: " << rc << "). Recompute the number of valid values.");

        readDepthMap(rc, mp, depthMap, scale, step, customSuffix);

        nbDepthValues = std::count_if(depthMap.data(), depthMap.data() + depthMap.size(), [](float v) { return v > 0.0f; });
    }

    return nbDepthValues;
}

void deleteDepthSimMapTiles(int rc,
                            const MultiViewParams& mp,
                            int scale,
                            int step,
                            const std::string& customSuffix)
{
  std::vector<std::string> depthMapTilePathList;
  std::vector<std::string> simMapTilePathList;

  getTilePathList(rc, mp, EFileType::depthMap, scale, step, customSuffix, depthMapTilePathList);
  getTilePathList(rc, mp, EFileType::simMap,   scale, step, customSuffix, simMapTilePathList);

  if(depthMapTilePathList.empty()) // depth map can be empty
    ALICEVISION_LOG_INFO("Cannot find any depth map tile file to delete (rc: " << rc << ").");

  if(simMapTilePathList.empty()) // sim map can be empty
    ALICEVISION_LOG_INFO("Cannot find any similarity map tile file to delete (rc: " << rc << ").");

  // delete depth map tile files
  for(const std::string& depthMapTilePath : depthMapTilePathList)
  {
    try
    {
      fs::remove(depthMapTilePath);
    }
    catch (const std::exception& e)
    {
      ALICEVISION_LOG_WARNING("Cannot delete depth map tile file (rc: " << rc << "): " << fs::path(depthMapTilePath).filename().string() << std::endl);
    }
  }

  // delete similarity map tile files
  for(const std::string& simMapTilePath : simMapTilePathList)
  {
    try
    {
      fs::remove(simMapTilePath);
    }
    catch (const std::exception& e)
    {
      ALICEVISION_LOG_WARNING("Cannot delete similarity map tile file (rc: " << rc << "): " << fs::path(simMapTilePath).filename().string() << std::endl);
    }
  }
}

} // namespace mvsUtils
} // namespace aliceVision
