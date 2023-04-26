// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
// Copyright (c) 2016 openMVG contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include <aliceVision/numeric/numeric.hpp>
#include <aliceVision/image/io.hpp>
#include <aliceVision/image/Sampler.hpp>
#include <aliceVision/system/Logger.hpp>
#include <aliceVision/cmdline/cmdline.hpp>
#include <aliceVision/system/main.hpp>

#include <dependencies/vectorGraphics/svgDrawer.hpp>
#include <aliceVision/panorama/sphericalMapping.hpp>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/math/constants/constants.hpp>

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>

#include <string>
#include <iostream>
#include <iterator>
#include <fstream>
#include <vector>

// These constants define the current software version.
// They must be updated when the command line is changed.
#define ALICEVISION_SOFTWARE_VERSION_MAJOR 2
#define ALICEVISION_SOFTWARE_VERSION_MINOR 0

using namespace aliceVision;

namespace fs = boost::filesystem;
namespace po = boost::program_options;
namespace oiio = OIIO;

/**
 * @brief A pinhole camera with its associated rotation
 * Used to sample the spherical image
 */
class PinholeCameraR
{
public:

    PinholeCameraR(int focal, int width, int height, const Mat3& R)
        : _R(R)
    {
    _K << focal,   0,  width/2.0,
            0, focal, height/2.0,
            0,     0,          1;
    }

    Vec3 getLocalRay(double x, double y) const
    {
        return (_K.inverse() * Vec3(x, y, 1.0)).normalized();
    }

    Vec3 getRay(double x, double y) const
    {
        return _R * getLocalRay(x, y);
    }

private:

    /// Rotation matrix
    Mat3 _R;
    /// Intrinsic matrix
    Mat3 _K;

};

/**
 * @brief Compute a rectilinear camera focal from an angular FoV
 * @param h
 * @param thetaMax camera FoV
 * @return
 */
double focalFromPinholeHeight(int height, double thetaMax = degreeToRadian(60.0))
{
    float f = 1.f;
    while (thetaMax < atan2(height / (2 * f) , 1))
    {
        ++f;
    }
    return f;
}

bool splitDualFisheye(const std::string& imagePath, const std::string& outputFolder, const std::string& extension,
                      const std::string& splitPreset)
{
    // Load source image from disk
    image::Image<image::RGBfColor> imageSource;
    image::readImage(imagePath, imageSource, image::EImageColorSpace::LINEAR);

    // Make sure image is horizontal
    if(imageSource.Height() > imageSource.Width())
    {
        ALICEVISION_THROW_ERROR("Cannot split dual fisheye from the vertical image " << imagePath);
    }

    // Retrieve useful dimensions for cropping
    const int outSide = std::min(imageSource.Height(), imageSource.Width() / 2);
    const int offset = std::abs((imageSource.Width() / 2) - imageSource.Height());
    const int halfOffset = offset / 2;

    // Make sure rig folder exists
    std::string rigFolder = outputFolder + "/rig";
    fs::create_directory(rigFolder);

    for (std::size_t i = 0; i < 2; ++i)
    {
        // Retrieve corner position of cropping area
        const int xbegin = i * outSide;
        int ybegin = 0;

        if (splitPreset == "bottom")
        {
            ybegin += offset;
        }
        else if (splitPreset == "center")
        {
            ybegin += halfOffset;
        }

        // Create new image containing the cropped area
        image::Image<image::RGBfColor> imageOut(imageSource.block(ybegin, xbegin, outSide, outSide));

        // Make sure sub-folder exists for complete rig structure
        std::string subFolder = rigFolder + std::string("/") + std::to_string(i);
        fs::create_directory(subFolder);

        // Save new image on disk
        fs::path path(imagePath);
        std::string filename = extension.empty() ?
            path.filename().string() : path.stem().string() + "." + extension;
        image::writeImage(subFolder + std::string("/") + filename,
                          imageOut, image::ImageWriteOptions(), image::readImageMetadata(imagePath));
    }

    // Success
    ALICEVISION_LOG_INFO(imagePath + " successfully split");
    return true;
}

bool splitEquirectangular(const std::string& imagePath, const std::string& outputFolder, const std::string& extension,
                          std::size_t nbSplits, std::size_t splitResolution, double fovDegree)
{
    // Load source image from disk
    image::Image<image::RGBColor> imageSource;
    image::readImage(imagePath, imageSource, image::EImageColorSpace::LINEAR);

    const int inWidth = imageSource.Width();
    const int inHeight = imageSource.Height();

    std::vector<PinholeCameraR> cameras;

    const double twoPi = boost::math::constants::pi<double>() * 2.0;
    const double alpha = twoPi / static_cast<double>(nbSplits);

    const double fov = degreeToRadian(fovDegree);
    const double focal_px = (splitResolution / 2.0) / tan(fov / 2.0);

    double angle = 0.0;
    for(std::size_t i = 0; i < nbSplits; ++i)
    {
        cameras.emplace_back(focal_px, splitResolution, splitResolution, RotationAroundY(angle));
        angle += alpha;
    }

    const image::Sampler2d<image::SamplerLinear> sampler;
    image::Image<image::RGBColor> imaOut(splitResolution, splitResolution, image::BLACK);

    // Make sure rig folder exists
    std::string rigFolder = outputFolder + "/rig";
    fs::create_directory(rigFolder);

    size_t index = 0;
    for(const PinholeCameraR& camera : cameras)
    {
        imaOut.fill(image::BLACK);

        // Backward mapping:
        // - Find for each pixels of the pinhole image where it comes from the panoramic image
        for(int j = 0; j < splitResolution; ++j)
        {
            for(int i = 0; i < splitResolution; ++i)
            {
                const Vec3 ray = camera.getRay(i, j);
                const Vec2 x = SphericalMapping::toEquirectangular(ray, inWidth, inHeight);
                imaOut(j,i) = sampler(imageSource, x(1), x(0));
            }
        }

        // Retrieve image specs and metadata
        oiio::ImageBuf bufferOut;
        image::getBufferFromImage(imageSource, bufferOut);

        oiio::ImageSpec& outMetadataSpec = bufferOut.specmod();

        outMetadataSpec.extra_attribs = image::readImageMetadata(imagePath);

        // Override make and model in order to force camera model in SfM
        outMetadataSpec.attribute("Make",  "Custom");
        outMetadataSpec.attribute("Model", "Pinhole");
        const float focal_mm = focal_px / splitResolution; // muliplied by sensorWidth (which is 1 for "Custom")
        outMetadataSpec.attribute("Exif:FocalLength", focal_mm);

        // Make sure sub-folder exists for complete rig structure
        std::string subFolder = rigFolder + std::string("/") + std::to_string(index);
        fs::create_directory(subFolder);

        // Save new image on disk
        fs::path path(imagePath);
        std::string filename = extension.empty() ?
            path.filename().string() : path.stem().string() + "." + extension;
        image::writeImage(subFolder + std::string("/") + filename,
                          imaOut, image::ImageWriteOptions(), outMetadataSpec.extra_attribs);

        ++index;
    }
    ALICEVISION_LOG_INFO(imagePath + " successfully split");
    return true;
}


bool splitEquirectangularPreview(const std::string& imagePath, const std::string& outputFolder,
                                 std::size_t nbSplits, std::size_t splitResolution, double fovDegree)
{
    // Load source image from disk
    image::Image<image::RGBColor> imageSource;
    image::readImage(imagePath, imageSource, image::EImageColorSpace::LINEAR);

    const int inWidth = imageSource.Width();
    const int inHeight = imageSource.Height();

    std::vector<PinholeCameraR> cameras;

    const double twoPi = boost::math::constants::pi<double>() * 2.0;
    const double alpha = twoPi / static_cast<double>(nbSplits);

    const double fov = degreeToRadian(fovDegree);
    const double focal = (splitResolution / 2.0) / tan(fov / 2.0);

    double angle = 0.0;
    for(std::size_t i = 0; i < nbSplits; ++i)
    {
        cameras.emplace_back(focal, splitResolution, splitResolution, RotationAroundY(angle));
        angle += alpha;
    }

    svg::svgDrawer svgStream(inWidth, inHeight);
    svgStream.drawRectangle(0, 0, inWidth, inHeight, svg::svgStyle().fill("black"));
    svgStream.drawImage(imagePath, inWidth, inHeight, 0, 0, 0.7f);
    svgStream.drawLine(0,0,inWidth, inHeight, svg::svgStyle().stroke("white"));
    svgStream.drawLine(inWidth,0, 0, inHeight, svg::svgStyle().stroke("white"));

    // For each cam, reproject the image borders onto the panoramic image

    for (const PinholeCameraR& camera : cameras)
    {
        // Draw the shot border with the given step
        const int step = 10;
        Vec3 ray;

        // Vertical rectilinear image border
        for (double j = 0; j <= splitResolution; j += splitResolution/(double)step)
        {
            Vec2 pt(0.,j);
            ray = camera.getRay(pt(0), pt(1));
            Vec2 x = SphericalMapping::toEquirectangular( ray, inWidth, inHeight);
            svgStream.drawCircle(x(0), x(1), 8, svg::svgStyle().fill("magenta").stroke("white", 4));

            pt[0] = splitResolution;
            ray = camera.getRay(pt(0), pt(1));
            x = SphericalMapping::toEquirectangular( ray, inWidth, inHeight);
            svgStream.drawCircle(x(0), x(1), 8, svg::svgStyle().fill("magenta").stroke("white", 4));
        }

        // Horizontal rectilinear image border
        for (double j = 0; j <= splitResolution; j += splitResolution/(double)step)
        {
            Vec2 pt(j,0.);
            ray = camera.getRay(pt(0), pt(1));
            Vec2 x = SphericalMapping::toEquirectangular( ray, inWidth, inHeight);
            svgStream.drawCircle(x(0), x(1), 8, svg::svgStyle().fill("lime").stroke("white", 4));

            pt[1] = splitResolution;
            ray = camera.getRay(pt(0), pt(1));
            x = SphericalMapping::toEquirectangular( ray, inWidth, inHeight);
            svgStream.drawCircle(x(0), x(1), 8, svg::svgStyle().fill("lime").stroke("white", 4));
        }
    }

    fs::path path(imagePath);
    std::ofstream svgFile(outputFolder + std::string("/") + path.stem().string() + std::string(".svg"));
    svgFile << svgStream.closeSvgFile().str();
    return true;
}

int aliceVision_main(int argc, char** argv)
{
    // command-line parameters
    std::string inputPath;                      // media file path list
    std::string outputFolder;                   // output folder for splited images
    std::string splitMode;                      // split mode (exif, dualfisheye, equirectangular)
    std::string dualFisheyeSplitPreset;         // dual-fisheye split type preset
    std::size_t equirectangularNbSplits;        // nb splits for equirectangular image
    std::size_t equirectangularSplitResolution; // split resolution for equirectangular image
    bool equirectangularPreviewMode = false;
    double fov = 110.0;                         // Field of View in degree
    int nbThreads = 3;
    std::string extension;

    po::options_description requiredParams("Required parameters");
    requiredParams.add_options()
        ("input,i", po::value<std::string>(&inputPath)->required(),
        "Input image file or image folder.")
        ("output,o", po::value<std::string>(&outputFolder)->required(),
        "Output keyframes folder for .jpg");

    po::options_description optionalParams("Optional parameters");
    optionalParams.add_options()
        ("splitMode,m", po::value<std::string>(&splitMode)->default_value("equirectangular"),
        "Split mode (equirectangular, dualfisheye)")
        ("dualFisheyeSplitPreset", po::value<std::string>(&dualFisheyeSplitPreset)->default_value("center"),
        "Dual-Fisheye split type preset (center, top, bottom)")
        ("equirectangularNbSplits", po::value<std::size_t>(&equirectangularNbSplits)->default_value(2),
        "Equirectangular number of splits")
        ("equirectangularSplitResolution", po::value<std::size_t>(&equirectangularSplitResolution)->default_value(1200),
        "Equirectangular split resolution")
        ("equirectangularPreviewMode", po::value<bool>(&equirectangularPreviewMode)->default_value(equirectangularPreviewMode),
        "Export a SVG file that simulate the split")
        ("fov", po::value<double>(&fov)->default_value(fov),
        "Field of View to extract (in degree).")
        ("nbThreads", po::value<int>(&nbThreads)->default_value(nbThreads),
        "Number of threads.")
        ("extension", po::value<std::string>(&extension)->default_value(extension),
         "Output image extension (empty to keep the source file format).");

    CmdLine cmdline("This program is used to extract multiple images from equirectangular or dualfisheye images or image folder.\n"
                    "AliceVision split360Images");
    cmdline.add(requiredParams);
    cmdline.add(optionalParams);
    if (!cmdline.execute(argc, argv))
    {
        return EXIT_FAILURE;
    }

    // check output folder and update to its absolute path
    {
        const fs::path outDir = fs::absolute(outputFolder);
        outputFolder = outDir.string();
        if (!fs::is_directory(outDir))
        {
            ALICEVISION_LOG_ERROR("Can't find folder " << outputFolder);
            return EXIT_FAILURE;
        }
    }

    //check split mode
    {
        //splitMode to lower
        std::transform(splitMode.begin(), splitMode.end(), splitMode.begin(), ::tolower);

        if (splitMode != "exif" &&
            splitMode != "equirectangular" &&
            splitMode != "dualfisheye")
        {
            ALICEVISION_LOG_ERROR("Invalid split mode : " << splitMode);
            return EXIT_FAILURE;
        }
    }

    //check dual-fisheye split preset
    {
        //dualFisheyeSplitPreset to lower
        std::transform(dualFisheyeSplitPreset.begin(), dualFisheyeSplitPreset.end(), dualFisheyeSplitPreset.begin(), ::tolower);

        if (dualFisheyeSplitPreset != "top" &&
            dualFisheyeSplitPreset != "bottom" &&
            dualFisheyeSplitPreset != "center")
        {
            ALICEVISION_LOG_ERROR("Invalid dual-fisheye split preset : " << dualFisheyeSplitPreset);
            return EXIT_FAILURE;
        }
    }

    std::vector<std::string> imagePaths;
    std::vector<std::string> badPaths;

    {
        const fs::path path = fs::absolute(inputPath);
        if (fs::exists(path) && fs::is_directory(path))
        {
            for (fs::directory_entry& entry : boost::make_iterator_range(fs::directory_iterator(path), {}))
                imagePaths.push_back(entry.path().string());

            ALICEVISION_LOG_INFO("Find " << imagePaths.size() << " file paths.");
        }
        else if (fs::exists(path))
        {
            imagePaths.push_back(path.string());
        }
        else
        {
            ALICEVISION_LOG_ERROR("Can't find file or folder " << inputPath);
            return EXIT_FAILURE;
        }
    }

    #pragma omp parallel for num_threads(nbThreads)
    for (int i = 0; i < imagePaths.size(); ++i)
    {
        const std::string& imagePath = imagePaths[i];
        bool hasCorrectPath = true;

        if (splitMode == "equirectangular")
        {
            if(equirectangularPreviewMode)
            {
                hasCorrectPath =
                    splitEquirectangularPreview(imagePath, outputFolder,
                                                equirectangularNbSplits, equirectangularSplitResolution, fov);
            }
            else
            {
                hasCorrectPath =
                    splitEquirectangular(imagePath, outputFolder, extension,
                                         equirectangularNbSplits, equirectangularSplitResolution, fov);
            }
        }
        else if(splitMode == "dualfisheye")
        {
            hasCorrectPath =
                splitDualFisheye(imagePath, outputFolder, extension,
                                 dualFisheyeSplitPreset);
        }
        else //exif
        {
            ALICEVISION_LOG_ERROR("Exif mode not implemented yet !");
        }

        if (!hasCorrectPath)
        {
            #pragma omp critical
            badPaths.push_back(imagePath);
        }
    }

    if (!badPaths.empty())
    {
        ALICEVISION_LOG_ERROR("Error: Can't open image file(s) below");
        for (const std::string& imagePath : imagePaths)
            ALICEVISION_LOG_ERROR("\t - " << imagePath);
    }

    return EXIT_SUCCESS;
}
