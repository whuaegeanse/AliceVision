// This file is part of the AliceVision project.
// Copyright (c) 2017 AliceVision contributors.
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file,
// You can obtain one at https://mozilla.org/MPL/2.0/.

#include "viewIO.hpp"

#include <aliceVision/numeric/numeric.hpp>
#include <aliceVision/sfmData/uid.hpp>
#include <aliceVision/camera/camera.hpp>
#include <aliceVision/image/io.hpp>
#include "aliceVision/utils/filesIO.hpp"

#include <stdexcept>
#include <regex>

namespace fs = std::filesystem;

namespace aliceVision {
namespace sfmDataIO {

void updateIncompleteView(sfmData::View& view, EViewIdMethod viewIdMethod, const std::string& viewIdRegex)
{
    // check if the view is complete
    if (view.getViewId() != UndefinedIndexT && view.getIntrinsicId() != UndefinedIndexT && view.getPoseId() == view.getViewId() &&
        view.getImage().getHeight() > 0 && view.getImage().getWidth() > 0)
        return;

    int width, height;
    const auto metadata = image::readImageMetadata(view.getImage().getImagePath(), width, height);

    view.getImage().setWidth(width);
    view.getImage().setHeight(height);

    // reset metadata
    if (view.getImage().getMetadata().empty())
        view.getImage().setMetadata(image::getMapFromMetadata(metadata));

    // Reset viewId
    if (view.getViewId() == UndefinedIndexT)
    {
        if (viewIdMethod == EViewIdMethod::FILENAME)
        {
            std::regex re;
            try
            {
                re = viewIdRegex;
            }
            catch (const std::regex_error& e)
            {
                throw std::invalid_argument("Invalid regex conversion, your regexfilename '" + viewIdRegex + "' may be invalid.");
            }

            // Get view image filename without extension
            const std::string filename = fs::path(view.getImage().getImagePath()).stem().string();

            std::smatch match;
            std::regex_search(filename, match, re);
            if (match.size() == 2)
            {
                try
                {
                    const IndexT id(std::stoul(match.str(1)));
                    view.setViewId(id);
                }
                catch (std::invalid_argument& e)
                {
                    ALICEVISION_LOG_ERROR("ViewId captured in the filename '" << filename
                                                                              << "' can't be converted to a number. "
                                                                                 "The regex '"
                                                                              << viewIdRegex << "' is probably incorrect.");
                    throw;
                }
            }
            else
            {
                ALICEVISION_LOG_ERROR("The Regex '" << viewIdRegex << "' must match a unique number in the filename " << filename
                                                    << "' to be used as viewId.");
                throw std::invalid_argument("The Regex '" + viewIdRegex + "' must match a unique number in the filename " + filename +
                                            "' to be used as viewId.");
            }
        }
        else
        {
            // Use metadata
            view.setViewId(sfmData::computeViewUID(view));
        }
    }

    if (view.getPoseId() == UndefinedIndexT)
    {
        // check if the rig poseId id is defined
        if (view.isPartOfRig())
        {
            ALICEVISION_LOG_ERROR("Error: Can't find poseId for'" << fs::path(view.getImage().getImagePath()).filename().string()
                                                                  << "' marked as part of a rig." << std::endl);
            throw std::invalid_argument("Error: Can't find poseId for'" + fs::path(view.getImage().getImagePath()).filename().string() +
                                        "' marked as part of a rig.");
        }
        else
            view.setPoseId(view.getViewId());
    }
    else if ((!view.isPartOfRig()) && (view.getPoseId() != view.getViewId()))
    {
        ALICEVISION_LOG_WARNING("PoseId and viewId are different for image '" << fs::path(view.getImage().getImagePath()).filename().string() << "'."
                                                                              << std::endl);
    }
}

std::shared_ptr<camera::IntrinsicBase> getViewIntrinsic(const sfmData::View& view,
                                                        double mmFocalLength,
                                                        double sensorWidth,
                                                        double defaultFocalLength,
                                                        double defaultFieldOfView,
                                                        double defaultFocalRatio,
                                                        double defaultOffsetX,
                                                        double defaultOffsetY,
                                                        LensParam* lensParam,
                                                        camera::EINTRINSIC defaultIntrinsicType,
                                                        camera::EDISTORTION defaultDistortionType)
{
    // can't combine defaultFocalLengthPx and defaultFieldOfView
    assert(defaultFocalLength < 0 || defaultFieldOfView < 0);

    // get view informations
    const std::string& cameraBrand = view.getImage().getMetadataMake();
    const std::string& cameraModel = view.getImage().getMetadataModel();
    const std::string& bodySerialNumber = view.getImage().getMetadataBodySerialNumber();
    const std::string& lensSerialNumber = view.getImage().getMetadataLensSerialNumber();

    double focalLength{-1.0};
    bool hasFocalLengthInput = false;

    if (sensorWidth < 0)
    {
        ALICEVISION_LOG_WARNING("Sensor size is unknown");
        ALICEVISION_LOG_WARNING("Use default sensor size (36 mm)");
        sensorWidth = 36.0;
    }

    if (defaultFocalLength > 0.0)
    {
        focalLength = defaultFocalLength;
    }

    if (defaultFieldOfView > 0.0)
    {
        const double focalRatio = 0.5 / std::tan(0.5 * degreeToRadian(defaultFieldOfView));
        focalLength = focalRatio * sensorWidth;
    }

    camera::EINTRINSIC intrinsicType = defaultIntrinsicType;
    camera::EDISTORTION distortionType = defaultDistortionType;

    bool isResized = false;

    if (view.getImage().hasMetadata({"Exif:PixelXDimension", "PixelXDimension"}) &&
        view.getImage().hasMetadata({"Exif:PixelYDimension", "PixelYDimension"}))  // has dimension metadata
    {
        // check if the image is resized
        int exifWidth = std::stoi(view.getImage().getMetadata({"Exif:PixelXDimension", "PixelXDimension"}));
        int exifHeight = std::stoi(view.getImage().getMetadata({"Exif:PixelYDimension", "PixelXDimension"}));

        // if metadata is rotated
        if (exifWidth == view.getImage().getHeight() && exifHeight == view.getImage().getWidth())
            std::swap(exifWidth, exifHeight);

        if (exifWidth > 0 && exifHeight > 0 && (exifWidth != view.getImage().getWidth() || exifHeight != view.getImage().getHeight()))
        {
            ALICEVISION_LOG_WARNING("Resized image detected: "
                                    << fs::path(view.getImage().getImagePath()).filename().string() << std::endl
                                    << "\t- real image size: " << view.getImage().getWidth() << "x" << view.getImage().getHeight() << std::endl
                                    << "\t- image size from exif metadata is: " << exifWidth << "x" << exifHeight << std::endl);
            isResized = true;
        }
    }

    // handle case where focal length (mm) is unset or false
    if (mmFocalLength <= 0.0)
    {
        ALICEVISION_LOG_WARNING("Image '" << fs::path(view.getImage().getImagePath()).filename().string()
                                          << "' focal length (in mm) metadata is missing." << std::endl
                                          << "Can't compute focal length, use default." << std::endl);
    }
    else
    {
        // Retrieve the focal from the metadata in mm and convert to pixel.
        focalLength = mmFocalLength;
        hasFocalLengthInput = true;
    }

    double focalLengthIn35mm = 36.0 * focalLength;
    double pxFocalLength = (focalLength / sensorWidth) * std::max(view.getImage().getWidth(), view.getImage().getHeight());

    // retrieve pixel aspect ratio
    double pixelAspectRatio = 1.0 / defaultFocalRatio;
    view.getImage().getDoubleMetadata({"PixelAspectRatio"}, pixelAspectRatio);
    const double focalRatio = 1.0 / pixelAspectRatio;

    bool hasFisheyeCompatibleParameters = ((focalLengthIn35mm > 0.0 && focalLengthIn35mm < 18.0) || (defaultFieldOfView > 100.0));
    
    // choose intrinsic type

    //If lcp exists and is valid
    camera::EINTRINSIC lcpIntrinsicType = camera::EINTRINSIC::UNKNOWN;
    camera::EDISTORTION lcpDistortionType = camera::EDISTORTION::DISTORTION_NONE;
    if (!(lensParam != nullptr || lensParam->isEmpty()))
    {
        lcpIntrinsicType = camera::EINTRINSIC::PINHOLE_CAMERA;

        if (lensParam->isFisheye())
        {
            lcpDistortionType = camera::EDISTORTION::DISTORTION_FISHEYE;
        }
        else
        {
            lcpDistortionType = camera::EDISTORTION::DISTORTION_RADIALK3;
        }
    }

    if (cameraBrand == "Custom")
    {
        intrinsicType = camera::EINTRINSIC_stringToEnum(cameraModel);
    }
    else if ((lcpIntrinsicType != camera::EINTRINSIC::UNKNOWN))
    {
        intrinsicType = lcpIntrinsicType;
    }
    else if (hasFisheyeCompatibleParameters)
    {
        // If the focal lens is short, the fisheye model should fit better.
        intrinsicType = camera::EINTRINSIC::PINHOLE_CAMERA;
        distortionType = camera::EDISTORTION::DISTORTION_FISHEYE;
    }
    else if (intrinsicType == camera::EINTRINSIC::UNKNOWN)
    {
        intrinsicType = camera::EINTRINSIC::PINHOLE_CAMERA;
        distortionType = camera::EDISTORTION::DISTORTION_RADIALK3;
    }

    // create the desired intrinsic
    std::shared_ptr<camera::IntrinsicBase> intrinsic = camera::createIntrinsic(
      /*camera*/ intrinsicType,
      distortionType,
      camera::EUNDISTORTION::UNDISTORTION_NONE,
      /*dimensions*/ view.getImage().getWidth(),
      view.getImage().getHeight(),
      /*focal length*/ pxFocalLength,
      pxFocalLength / focalRatio,
      /*offset*/ 0,
      0);

    if (hasFocalLengthInput)
    {
        std::shared_ptr<camera::IntrinsicScaleOffset> intrinsicScaleOffset = std::dynamic_pointer_cast<camera::IntrinsicScaleOffset>(intrinsic);

        if (intrinsicScaleOffset)
        {
            intrinsicScaleOffset->setInitialScale({pxFocalLength, (pxFocalLength > 0) ? pxFocalLength / focalRatio : -1});
            intrinsicScaleOffset->setOffset({defaultOffsetX, defaultOffsetY});
        }
    }

    // initialize distortion parameters
    if (intrinsicType == camera::EINTRINSIC::PINHOLE_CAMERA)
    {
        switch (distortionType)
        {
            case camera::EDISTORTION::DISTORTION_FISHEYE:
            {
                if (cameraBrand == "GoPro")
                {
                    intrinsic->updateFromParams({pxFocalLength, pxFocalLength, 0, 0, 0.0524, 0.0094, -0.0037, -0.0004});
                }
                else if (lensParam && (!lensParam->isEmpty()) && (cameraBrand != "Custom"))
                {
                    std::vector<double> p = {pxFocalLength, pxFocalLength, 0, 0};
                    p.push_back(lensParam->fisheyeParams.RadialDistortParam1);
                    p.push_back(lensParam->fisheyeParams.RadialDistortParam2);
                    p.push_back(0.0);
                    p.push_back(0.0);
                    intrinsic->updateFromParams(p);
                }
                break;
            }
            case camera::EDISTORTION::DISTORTION_FISHEYE1:
            {
                if (cameraBrand == "GoPro")
                {
                    intrinsic->updateFromParams({pxFocalLength, pxFocalLength, 0, 0, 1.04});
                }
                break;
            }
            case camera::EDISTORTION::DISTORTION_RADIALK3:
            {
                if (lensParam && (!lensParam->isEmpty()) && (cameraBrand != "Custom"))
                {
                    std::vector<double> p = {pxFocalLength, pxFocalLength, 0, 0};
                    p.push_back(lensParam->perspParams.RadialDistortParam1);
                    p.push_back(lensParam->perspParams.RadialDistortParam2);
                    p.push_back(lensParam->perspParams.RadialDistortParam3);
                    intrinsic->updateFromParams(p);
                }
                break;
            }
            default:
                break;
        }
    }

    // create serial number
    intrinsic->setSerialNumber(bodySerialNumber + lensSerialNumber);

    return intrinsic;
}

std::vector<std::string> viewPathsFromFolders(const sfmData::View& view, const std::vector<std::string>& folders)
{
    return utils::getFilesPathsFromFolders(folders, [&view](const std::filesystem::path& path) {
        const std::filesystem::path stem = path.stem();
        return (stem == std::to_string(view.getViewId()) || stem == fs::path(view.getImage().getImagePath()).stem());
    });
}

bool extractNumberFromFileStem(const std::string& imagePathStem, IndexT& number, std::string& prefix, std::string& suffix)
{
    // check if the image stem contains a number
    // regexFrame: ^(.*\D)?([0-9]+)([\-_\.].*[[:alpha:]].*)?$
    std::regex regexFrame("^(.*\\D)?"        // the optional prefix which ends with a non digit character
                          "([0-9]+)"         // the number
                          "([\\-_\\.]"       // the suffix starts with a separator
                          ".*[[:alpha:]].*"  // at least one letter in the suffix
                          ")?$"              // suffix is optional
    );

    std::smatch matches;
    bool containsNumber = std::regex_search(imagePathStem, matches, regexFrame);

    if (containsNumber)
    {
        prefix = matches[1];
        suffix = matches[3];
        try
        {
            number = boost::lexical_cast<IndexT>(matches[2]);
        }
        catch (const boost::bad_lexical_cast&)
        {
            ALICEVISION_LOG_WARNING("Number in stem of file " << imagePathStem << " is out of range.");
            containsNumber = false;
        }
    }

    return containsNumber;
}

}  // namespace sfmDataIO
}  // namespace aliceVision
