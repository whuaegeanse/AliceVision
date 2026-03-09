"""
Collection of unit tests for the Pinhole intrinsics.
"""

import pytest
import numpy as np

from pyalicevision import camera as av
from pyalicevision import numeric as avnum


DEFAUT_PARAMETERS = (1.0, 1.0, 0.0, 0.0)

def test_pinhole_default_constructor():
    """ Test creating a default Pinhole object and checking its default values
    have been correctly set. """
    intrinsic = av.Pinhole()

    # Distortion and undistortion are not set, default type is "EINTRINSIC::PINHOLE_CAMERA"
    assert intrinsic.getType() == 2 and intrinsic.getTypeStr() == "pinhole"

    assert intrinsic.w() == 1, "The Pinhole intrinsic's default width should be 1"
    assert intrinsic.h() == 1, "The Pinhole intrinsic's default height should be 1"
    assert intrinsic.getFocalLengthPixX() == 1.0, \
        "The Pinhole intrinsic's focal length in X should be 1.0"
    assert intrinsic.getFocalLengthPixY() == 1.0, \
        "The Pinhole intrinsic's focal length in Y should be 1.0"

    offset = intrinsic.getOffset()
    assert offset[0] == 0.0 and offset[1] == 0.0

    assert intrinsic.sensorWidth() == 36.0
    assert intrinsic.sensorHeight() == 24.0

    assert intrinsic.getHorizontalFov() == 0.9272952180016122
    assert intrinsic.getVerticalFov() == 0.9272952180016122

    assert not intrinsic.hasDistortion()
    assert intrinsic.isValid()



def test_pinhole_constructor():
    """ Test creating a Pinhole object using the full-on constructor and checking its set
    values are correct. """
    intrinsic = av.Pinhole(1000, 800, 900, 700, 0, 0)

    # Distortion and undistortion are not set, default type is "EINTRINSIC::PINHOLE_CAMERA"
    assert intrinsic.getType() == 2 and intrinsic.getTypeStr() == "pinhole"

    assert intrinsic.w() == 1000
    assert intrinsic.h() == 800
    assert intrinsic.getFocalLengthPixX() == 900
    assert intrinsic.getFocalLengthPixY() == 700

    assert intrinsic.sensorWidth() == 36.0
    assert intrinsic.sensorHeight() == 24.0

    assert intrinsic.getHorizontalFov() == 1.014197008784674
    assert intrinsic.getVerticalFov() == 1.0382922284930458

    assert intrinsic.isValid()

    # TODO: test constructor with shared_ptr of distortion models


def test_pinhole_clone():
    """ Test creating a Pinhole object, cloning it, and checking the values
    of the cloned object are correct. """
    intrinsic1 = av.Pinhole()
    intrinsic2 = intrinsic1.clone()

    assert intrinsic1.isValid() and intrinsic2.isValid()
    assert intrinsic1.w() == intrinsic2.w()
    assert intrinsic1.h() == intrinsic2.h()
    assert intrinsic1.sensorWidth() == intrinsic2.sensorWidth()
    assert intrinsic1.sensorHeight() == intrinsic2.sensorHeight()

    intrinsic1.setWidth(1000)
    intrinsic1.setHeight(800)
    intrinsic1.setSensorWidth(17.0)
    intrinsic1.setSensorHeight(13.0)
    assert intrinsic1.w() != intrinsic2.w()
    assert intrinsic1.h() != intrinsic2.h()
    assert intrinsic1.sensorWidth() != intrinsic2.sensorWidth()
    assert intrinsic1.sensorHeight() != intrinsic2.sensorHeight()


def test_pinhole_is_valid():
    """ Test creating valid and invalid Pinhole objects and checking whether they are
    correct. """
    # For the default constructor, the width and height are set to 1
    intrinsic1 = av.Pinhole()
    assert intrinsic1.isValid()

    # Width and height are custom, but different from 0
    intrinsic2 = av.Pinhole(1000, 800, 900, 700, 0, 0)
    assert intrinsic2.isValid()

    # Width and height are forcibly set to 0, which should make the model invalid
    intrinsic3 = av.Pinhole(0, 0, 0, 0, 0, 0)
    assert not intrinsic3.isValid()


def test_pinhole_get_set_params():
    """ Test creating a Pinhole object, getting and setting its parameters with the
    parent's class getters and setters. """
    intrinsic = av.Pinhole()
    params = intrinsic.getParameters()

    assert len(params) == intrinsic.getParametersSize()
    assert params == DEFAUT_PARAMETERS

    params = (2.0, 2.0, 1.0, 1.0)

    assert params != intrinsic.getParameters()
    intrinsic.updateFromParams(params)
    assert params == intrinsic.getParameters()


def test_pinhole_lock_unlock():
    """ Test creating a Pinhole object and getting/updating its lock status. """
    intrinsic = av.Pinhole()
    assert not intrinsic.isLocked()

    intrinsic.lock()
    assert intrinsic.isLocked()
    intrinsic.unlock()
    assert not intrinsic.isLocked()


def test_pinhole_ratio_lock_unlock():
    """ Test creating a Pinhole object and getting/updating the lock status of its ratio. """
    intrinsic = av.Pinhole()
    assert intrinsic.isRatioLocked()

    intrinsic.setRatioLocked(False)
    assert not intrinsic.isRatioLocked()
    intrinsic.setRatioLocked(True)
    assert intrinsic.isRatioLocked()


def test_pinhole_get_set_serial_number():
    """ Test creating a Pinhole object and getting/updating its serial number. """
    intrinsic = av.Pinhole()
    assert intrinsic.serialNumber() == ""

    serialNumber = "0123456"
    intrinsic.setSerialNumber(serialNumber)
    assert intrinsic.serialNumber() == serialNumber


def test_pinhole_get_set_state():
    """" Test creating Pinhole objects, initializing their state, and getting/updating
    it with the getters and setters. """
    intrinsic1 = av.Pinhole()
    assert intrinsic1.getState() == av.EEstimatorParameterState_REFINED
    assert not intrinsic1.isLocked()

    # If the intrinsic is not locked, the state should be initialized to "REFINED"
    intrinsic1.initializeState()
    assert intrinsic1.getState() == av.EEstimatorParameterState_REFINED
    intrinsic1.setState(av.EEstimatorParameterState_IGNORED)
    assert intrinsic1.getState() == av.EEstimatorParameterState_IGNORED

    intrinsic2 = av.Pinhole()
    assert intrinsic2.getState() == av.EEstimatorParameterState_REFINED
    intrinsic2.lock()
    assert intrinsic2.isLocked()

    # If the intrinsic is locked, the state should be initialized to "CONSTANT"
    intrinsic2.initializeState()
    assert intrinsic2.getState() == av.EEstimatorParameterState_CONSTANT
    intrinsic2.setState(av.EEstimatorParameterState_REFINED)
    assert intrinsic2.getState() == av.EEstimatorParameterState_REFINED


def test_pinhole_get_set_initialization_mode():
    """ Test creating an Pinhole object and getting/updating its initialization mode
    with the dedicated getters and setters. """
    intrinsic = av.Pinhole()
    assert intrinsic.getInitializationMode() == av.EInitMode_NONE

    intrinsic.setInitializationMode(av.EInitMode_ESTIMATED)
    assert intrinsic.getInitializationMode() == av.EInitMode_ESTIMATED


# =====================================================================
# cam2ima / ima2cam
# =====================================================================

def test_pinhole_cam2ima_default():
    """ Test cam2ima with default Pinhole (scale=(1,1), pp=(0.5,0.5)).
    cam2ima(p) = p * scale + pp. """
    intrinsic = av.Pinhole()
    # pp = (0 + 0.5, 0 + 0.5) = (0.5, 0.5)
    p = np.array([0.0, 0.0])
    result = intrinsic.cam2ima(p)
    assert result[0] == pytest.approx(0.5, abs=1e-12)
    assert result[1] == pytest.approx(0.5, abs=1e-12)

    p2 = np.array([1.0, -1.0])
    result2 = intrinsic.cam2ima(p2)
    assert result2[0] == pytest.approx(1.5, abs=1e-12)
    assert result2[1] == pytest.approx(-0.5, abs=1e-12)


def test_pinhole_cam2ima_configured():
    """ Test cam2ima with configured Pinhole (w=1000, h=800, fx=900, fy=700).
    pp = (0 + 500, 0 + 400) = (500, 400).
    cam2ima(p) = (p[0]*900 + 500, p[1]*700 + 400). """
    intrinsic = av.Pinhole(1000, 800, 900, 700, 0, 0)
    p = np.array([0.0, 0.0])
    result = intrinsic.cam2ima(p)
    assert result[0] == pytest.approx(500.0, abs=1e-10)
    assert result[1] == pytest.approx(400.0, abs=1e-10)

    p2 = np.array([0.1, 0.2])
    result2 = intrinsic.cam2ima(p2)
    assert result2[0] == pytest.approx(0.1 * 900 + 500.0, abs=1e-10)
    assert result2[1] == pytest.approx(0.2 * 700 + 400.0, abs=1e-10)


def test_pinhole_ima2cam_default():
    """ Test ima2cam with default Pinhole.
    ima2cam(p) = (p - pp) / scale. """
    intrinsic = av.Pinhole()
    p = np.array([0.5, 0.5])
    result = intrinsic.ima2cam(p)
    assert result[0] == pytest.approx(0.0, abs=1e-12)
    assert result[1] == pytest.approx(0.0, abs=1e-12)

    p2 = np.array([1.5, -0.5])
    result2 = intrinsic.ima2cam(p2)
    assert result2[0] == pytest.approx(1.0, abs=1e-12)
    assert result2[1] == pytest.approx(-1.0, abs=1e-12)


def test_pinhole_ima2cam_configured():
    """ Test ima2cam with configured Pinhole. """
    intrinsic = av.Pinhole(1000, 800, 900, 700, 0, 0)
    # ima2cam(p) = ((p[0] - 500) / 900, (p[1] - 400) / 700)
    p = np.array([590.0, 540.0])
    result = intrinsic.ima2cam(p)
    assert result[0] == pytest.approx(90.0 / 900.0, abs=1e-10)
    assert result[1] == pytest.approx(140.0 / 700.0, abs=1e-10)


def test_pinhole_cam2ima_ima2cam_round_trip():
    """ Test that cam2ima and ima2cam are inverse operations. """
    intrinsic = av.Pinhole(1000, 800, 900, 700, 5.0, -3.0)
    pts_cam = [np.array([0.0, 0.0]), np.array([0.1, -0.2]),
               np.array([-0.3, 0.15]), np.array([0.5, 0.5])]
    for p_cam in pts_cam:
        p_ima = intrinsic.cam2ima(p_cam)
        p_cam_back = intrinsic.ima2cam(p_ima)
        assert p_cam_back[0] == pytest.approx(p_cam[0], abs=1e-10)
        assert p_cam_back[1] == pytest.approx(p_cam[1], abs=1e-10)

    pts_ima = [np.array([500.0, 400.0]), np.array([600.0, 300.0]),
               np.array([450.0, 450.0])]
    for p_ima in pts_ima:
        p_cam = intrinsic.ima2cam(p_ima)
        p_ima_back = intrinsic.cam2ima(p_cam)
        assert p_ima_back[0] == pytest.approx(p_ima[0], abs=1e-10)
        assert p_ima_back[1] == pytest.approx(p_ima[1], abs=1e-10)


def test_pinhole_cam2ima_with_offset():
    """ Test cam2ima with non-zero offset. """
    intrinsic = av.Pinhole(1000, 800, 900, 700, 10.0, -5.0)
    # pp = (10 + 500, -5 + 400) = (510, 395)
    p = np.array([0.0, 0.0])
    result = intrinsic.cam2ima(p)
    assert result[0] == pytest.approx(510.0, abs=1e-10)
    assert result[1] == pytest.approx(395.0, abs=1e-10)


# =====================================================================
# addDistortion / removeDistortion (without distortion)
# =====================================================================

def test_pinhole_add_distortion_no_disto():
    """ Test that addDistortion returns the point unchanged when no distortion is set. """
    intrinsic = av.Pinhole(1000, 800, 900, 700, 0, 0)
    p = np.array([0.1, 0.2])
    result = intrinsic.addDistortion(p)
    assert result[0] == pytest.approx(p[0], abs=1e-12)
    assert result[1] == pytest.approx(p[1], abs=1e-12)


def test_pinhole_remove_distortion_no_disto():
    """ Test that removeDistortion returns the point unchanged when no distortion is set. """
    intrinsic = av.Pinhole(1000, 800, 900, 700, 0, 0)
    p = np.array([0.1, 0.2])
    result = intrinsic.removeDistortion(p)
    assert result[0] == pytest.approx(p[0], abs=1e-12)
    assert result[1] == pytest.approx(p[1], abs=1e-12)


def test_pinhole_add_remove_distortion_round_trip_no_disto():
    """ Test that addDistortion and removeDistortion are inverses when no distortion. """
    intrinsic = av.Pinhole(1000, 800, 900, 700, 0, 0)
    pts = [np.array([0.0, 0.0]), np.array([0.15, -0.1]),
           np.array([-0.3, 0.25]), np.array([0.5, 0.5])]
    for p in pts:
        p_add = intrinsic.addDistortion(p)
        p_back = intrinsic.removeDistortion(p_add)
        assert p_back[0] == pytest.approx(p[0], abs=1e-12)
        assert p_back[1] == pytest.approx(p[1], abs=1e-12)


# =====================================================================
# getUndistortedPixel / getDistortedPixel
# =====================================================================

def test_pinhole_get_undistorted_pixel_no_disto():
    """ Test that getUndistortedPixel returns the input when no distortion is set. """
    intrinsic = av.Pinhole(1000, 800, 900, 700, 0, 0)
    pts = [np.array([550.0, 420.0]), np.array([500.0, 400.0]),
           np.array([300.0, 600.0])]
    for p in pts:
        result = intrinsic.getUndistortedPixel(p)
        assert result[0] == pytest.approx(p[0], abs=1e-10)
        assert result[1] == pytest.approx(p[1], abs=1e-10)


def test_pinhole_get_distorted_pixel_no_disto():
    """ Test that getDistortedPixel returns the input when no distortion is set. """
    intrinsic = av.Pinhole(1000, 800, 900, 700, 0, 0)
    pts = [np.array([550.0, 420.0]), np.array([500.0, 400.0]),
           np.array([300.0, 600.0])]
    for p in pts:
        result = intrinsic.getDistortedPixel(p)
        assert result[0] == pytest.approx(p[0], abs=1e-10)
        assert result[1] == pytest.approx(p[1], abs=1e-10)


def test_pinhole_undistorted_distorted_pixel_round_trip():
    """ Test that getUndistortedPixel and getDistortedPixel are inverses (no disto). """
    intrinsic = av.Pinhole(1000, 800, 900, 700, 5.0, -3.0)
    pts = [np.array([510.0, 395.0]), np.array([600.0, 300.0]),
           np.array([400.0, 500.0])]
    for p in pts:
        p_undist = intrinsic.getUndistortedPixel(p)
        p_back = intrinsic.getDistortedPixel(p_undist)
        assert p_back[0] == pytest.approx(p[0], abs=1e-10)
        assert p_back[1] == pytest.approx(p[1], abs=1e-10)


# =====================================================================
# imagePlaneToCameraPlaneError / pixelProbability
# =====================================================================

def test_pinhole_image_plane_to_camera_plane_error():
    """ Test imagePlaneToCameraPlaneError: returns value / scale(0). """
    intrinsic = av.Pinhole(1000, 800, 900, 700, 0, 0)
    assert intrinsic.imagePlaneToCameraPlaneError(1.0) == pytest.approx(1.0 / 900.0, abs=1e-12)
    assert intrinsic.imagePlaneToCameraPlaneError(2.5) == pytest.approx(2.5 / 900.0, abs=1e-12)
    assert intrinsic.imagePlaneToCameraPlaneError(0.0) == pytest.approx(0.0, abs=1e-12)


def test_pinhole_pixel_probability():
    """ Test pixelProbability returns a positive value. """
    intrinsic = av.Pinhole(1000, 800, 900, 700, 0, 0)
    prob = intrinsic.pixelProbability()
    assert prob > 0.0


# =====================================================================
# getPrincipalPoint
# =====================================================================

def test_pinhole_get_principal_point():
    """ Test getPrincipalPoint: returns (offset_x + w/2, offset_y + h/2). """
    intrinsic = av.Pinhole(1000, 800, 900, 700, 10.0, -5.0)
    pp = intrinsic.getPrincipalPoint()
    assert pp[0] == pytest.approx(10.0 + 500.0, abs=1e-12)
    assert pp[1] == pytest.approx(-5.0 + 400.0, abs=1e-12)

    intrinsic_default = av.Pinhole()
    pp_default = intrinsic_default.getPrincipalPoint()
    assert pp_default[0] == pytest.approx(0.5, abs=1e-12)
    assert pp_default[1] == pytest.approx(0.5, abs=1e-12)


# =====================================================================
# setScale / getScale / setOffset / getOffset
# =====================================================================

def test_pinhole_set_get_scale():
    """ Test setting and getting the scale (focal length in pixels). """
    intrinsic = av.Pinhole(1000, 800, 900, 700, 0, 0)
    scale = intrinsic.getScale()
    assert scale[0] == pytest.approx(900.0, abs=1e-12)
    assert scale[1] == pytest.approx(700.0, abs=1e-12)

    new_scale = np.array([1200.0, 1100.0])
    intrinsic.setScale(new_scale)
    scale2 = intrinsic.getScale()
    assert scale2[0] == pytest.approx(1200.0, abs=1e-12)
    assert scale2[1] == pytest.approx(1100.0, abs=1e-12)


def test_pinhole_set_get_offset():
    """ Test setting and getting the offset. """
    intrinsic = av.Pinhole(1000, 800, 900, 700, 5.0, -3.0)
    offset = intrinsic.getOffset()
    assert offset[0] == pytest.approx(5.0, abs=1e-12)
    assert offset[1] == pytest.approx(-3.0, abs=1e-12)

    new_offset = np.array([10.0, 20.0])
    intrinsic.setOffset(new_offset)
    offset2 = intrinsic.getOffset()
    assert offset2[0] == pytest.approx(10.0, abs=1e-12)
    assert offset2[1] == pytest.approx(20.0, abs=1e-12)

    # Principal point should also update
    pp = intrinsic.getPrincipalPoint()
    assert pp[0] == pytest.approx(10.0 + 500.0, abs=1e-12)
    assert pp[1] == pytest.approx(20.0 + 400.0, abs=1e-12)


# =====================================================================
# offset lock / scale lock
# =====================================================================

def test_pinhole_offset_lock_unlock():
    """ Test getting/updating the lock status of the offset. """
    intrinsic = av.Pinhole()
    assert not intrinsic.isOffsetLocked()

    intrinsic.setOffsetLocked(True)
    assert intrinsic.isOffsetLocked()
    intrinsic.setOffsetLocked(False)
    assert not intrinsic.isOffsetLocked()


def test_pinhole_scale_lock_unlock():
    """ Test getting/updating the lock status of the scale. """
    intrinsic = av.Pinhole()
    assert not intrinsic.isScaleLocked()

    intrinsic.setScaleLocked(True)
    assert intrinsic.isScaleLocked()
    intrinsic.setScaleLocked(False)
    assert not intrinsic.isScaleLocked()


# =====================================================================
# K matrix / setK
# =====================================================================

def test_pinhole_k_matrix():
    """ Test that K() returns the correct intrinsic matrix.
    K = [[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]]. """
    intrinsic = av.Pinhole(1000, 800, 900, 700, 10.0, -5.0)
    K = intrinsic.K()
    # pp = (10 + 500, -5 + 400) = (510, 395)
    assert K[0, 0] == pytest.approx(900.0, abs=1e-12)
    assert K[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert K[0, 2] == pytest.approx(510.0, abs=1e-12)
    assert K[1, 0] == pytest.approx(0.0, abs=1e-12)
    assert K[1, 1] == pytest.approx(700.0, abs=1e-12)
    assert K[1, 2] == pytest.approx(395.0, abs=1e-12)
    assert K[2, 0] == pytest.approx(0.0, abs=1e-12)
    assert K[2, 1] == pytest.approx(0.0, abs=1e-12)
    assert K[2, 2] == pytest.approx(1.0, abs=1e-12)


def test_pinhole_k_matrix_default():
    """ Test K() for default Pinhole: fx=fy=1, pp=(0.5, 0.5). """
    intrinsic = av.Pinhole()
    K = intrinsic.K()
    assert K[0, 0] == pytest.approx(1.0, abs=1e-12)
    assert K[1, 1] == pytest.approx(1.0, abs=1e-12)
    assert K[0, 2] == pytest.approx(0.5, abs=1e-12)
    assert K[1, 2] == pytest.approx(0.5, abs=1e-12)
    assert K[2, 2] == pytest.approx(1.0, abs=1e-12)


def test_pinhole_set_k_from_values():
    """ Test setK(fx, fy, ppx, ppy) and verify via K(). """
    intrinsic = av.Pinhole(1000, 800, 900, 700, 0, 0)
    intrinsic.setK(1200.0, 1000.0, 520.0, 410.0)

    K = intrinsic.K()
    assert K[0, 0] == pytest.approx(1200.0, abs=1e-12)
    assert K[1, 1] == pytest.approx(1000.0, abs=1e-12)
    assert K[0, 2] == pytest.approx(520.0, abs=1e-12)
    assert K[1, 2] == pytest.approx(410.0, abs=1e-12)

    # offset = ppx - w/2, ppy - h/2
    offset = intrinsic.getOffset()
    assert offset[0] == pytest.approx(520.0 - 500.0, abs=1e-12)
    assert offset[1] == pytest.approx(410.0 - 400.0, abs=1e-12)


def test_pinhole_set_k_preserves_focal():
    """ Test setK and verify getFocalLengthPixX/Y. """
    intrinsic = av.Pinhole(1000, 800, 900, 700, 0, 0)
    intrinsic.setK(1500.0, 1300.0, 500.0, 400.0)
    assert intrinsic.getFocalLengthPixX() == pytest.approx(1500.0, abs=1e-12)
    assert intrinsic.getFocalLengthPixY() == pytest.approx(1300.0, abs=1e-12)


# =====================================================================
# getFocalLengthPixX / getFocalLengthPixY
# =====================================================================

def test_pinhole_get_focal_length_pix():
    """ Test getFocalLengthPixX/Y return the scale values. """
    intrinsic = av.Pinhole(1000, 800, 900, 700, 0, 0)
    assert intrinsic.getFocalLengthPixX() == pytest.approx(900.0, abs=1e-12)
    assert intrinsic.getFocalLengthPixY() == pytest.approx(700.0, abs=1e-12)

    intrinsic.setScale(np.array([1200.0, 1100.0]))
    assert intrinsic.getFocalLengthPixX() == pytest.approx(1200.0, abs=1e-12)
    assert intrinsic.getFocalLengthPixY() == pytest.approx(1100.0, abs=1e-12)


# =====================================================================
# getFocalLength / getPixelAspectRatio / setFocalLength
# =====================================================================

def test_pinhole_get_focal_length_mm():
    """ Test getFocalLength in mm. """
    intrinsic = av.Pinhole(1000, 800, 900, 700, 0, 0)
    # fx=900, fy=700, sensorWidth=36.0, max(w,h)=1000
    # focalInMM = fx * sensorWidth / max(w,h) = 900 * 36 / 1000 = 32.4
    # pixelAspectRatio = 1 / (fx/fy) = 700/900
    # result = 32.4 * (700/900)
    expected = 32.4 * (700.0 / 900.0)
    assert intrinsic.getFocalLength() == pytest.approx(expected, abs=1e-10)


def test_pinhole_get_pixel_aspect_ratio():
    """ Test getPixelAspectRatio: 1 / (fx/fy) = fy/fx. """
    intrinsic = av.Pinhole(1000, 800, 900, 700, 0, 0)
    expected = 700.0 / 900.0
    assert intrinsic.getPixelAspectRatio() == pytest.approx(expected, abs=1e-12)

    # Equal focal lengths -> ratio = 1
    intrinsic2 = av.Pinhole(1000, 800, 900, 900, 0, 0)
    assert intrinsic2.getPixelAspectRatio() == pytest.approx(1.0, abs=1e-12)


def test_pinhole_set_focal_length():
    """ Test setFocalLength and verifying the focal length is correctly retrieved. """
    intrinsic = av.Pinhole(1000, 800, 900, 900, 0, 0)
    intrinsic.setFocalLength(50.0, 1.0)
    assert intrinsic.getFocalLength() == pytest.approx(50.0, abs=1e-10)

    intrinsic.setFocalLength(35.0, 1.0)
    assert intrinsic.getFocalLength() == pytest.approx(35.0, abs=1e-10)


# =====================================================================
# rescale
# =====================================================================

def test_pinhole_rescale():
    """ Test rescaling the intrinsic. Width, height, scale, and offset are updated. """
    intrinsic = av.Pinhole(1000, 800, 900, 700, 10.0, -5.0)

    intrinsic.rescale(0.5, 0.5)

    assert intrinsic.w() == 500
    assert intrinsic.h() == 400

    scale = intrinsic.getScale()
    assert scale[0] == pytest.approx(450.0, abs=1e-10)
    assert scale[1] == pytest.approx(350.0, abs=1e-10)

    offset = intrinsic.getOffset()
    assert offset[0] == pytest.approx(5.0, abs=1e-10)
    assert offset[1] == pytest.approx(-2.5, abs=1e-10)


def test_pinhole_rescale_up():
    """ Test rescaling up. """
    intrinsic = av.Pinhole(500, 400, 450, 350, 0.0, 0.0)

    intrinsic.rescale(2.0, 2.0)

    assert intrinsic.w() == 1000
    assert intrinsic.h() == 800

    scale = intrinsic.getScale()
    assert scale[0] == pytest.approx(900.0, abs=1e-10)
    assert scale[1] == pytest.approx(700.0, abs=1e-10)


# =====================================================================
# hasDistortion / getDistortionParams / distortionInitializationMode
# =====================================================================

def test_pinhole_has_distortion():
    """ Test that hasDistortion returns False when no distortion is set. """
    intrinsic = av.Pinhole()
    assert not intrinsic.hasDistortion()


def test_pinhole_get_distortion_params_no_disto():
    """ Test getDistortionParams returns empty when no distortion is set. """
    intrinsic = av.Pinhole(1000, 800, 900, 700, 0, 0)
    assert intrinsic.getDistortionParamsSize() == 0
    params = intrinsic.getDistortionParams()
    assert len(params) == 0


def test_pinhole_distortion_initialization_mode():
    """ Test getting/setting the distortion initialization mode. """
    intrinsic = av.Pinhole()
    assert intrinsic.getDistortionInitializationMode() == av.EInitMode_NONE

    intrinsic.setDistortionInitializationMode(av.EInitMode_ESTIMATED)
    assert intrinsic.getDistortionInitializationMode() == av.EInitMode_ESTIMATED

    intrinsic.setDistortionInitializationMode(av.EInitMode_CALIBRATED)
    assert intrinsic.getDistortionInitializationMode() == av.EInitMode_CALIBRATED


# =====================================================================
# toUnitSphere
# =====================================================================

def test_pinhole_to_unit_sphere():
    """ Test toUnitSphere: returns pt.homogeneous().normalized(). """
    intrinsic = av.Pinhole(1000, 800, 900, 700, 0, 0)
    p = np.array([0.0, 0.0])
    result = intrinsic.toUnitSphere(p)
    # (0, 0).homogeneous() = (0, 0, 1), normalized = (0, 0, 1)
    assert result[0] == pytest.approx(0.0, abs=1e-12)
    assert result[1] == pytest.approx(0.0, abs=1e-12)
    assert result[2] == pytest.approx(1.0, abs=1e-12)


def test_pinhole_to_unit_sphere_nonzero():
    """ Test toUnitSphere with non-zero point. """
    intrinsic = av.Pinhole(1000, 800, 900, 700, 0, 0)
    p = np.array([1.0, 0.0])
    result = intrinsic.toUnitSphere(p)
    # (1, 0).homogeneous() = (1, 0, 1), norm = sqrt(2)
    norm = np.sqrt(2.0)
    assert result[0] == pytest.approx(1.0 / norm, abs=1e-12)
    assert result[1] == pytest.approx(0.0, abs=1e-12)
    assert result[2] == pytest.approx(1.0 / norm, abs=1e-12)


def test_pinhole_to_unit_sphere_is_unit():
    """ Test that toUnitSphere always returns a unit-length vector. """
    intrinsic = av.Pinhole(1000, 800, 900, 700, 0, 0)
    pts = [np.array([0.0, 0.0]), np.array([0.5, -0.3]),
           np.array([-1.0, 2.0]), np.array([3.0, 4.0])]
    for p in pts:
        result = intrinsic.toUnitSphere(p)
        norm = np.sqrt(result[0]**2 + result[1]**2 + result[2]**2)
        assert norm == pytest.approx(1.0, abs=1e-12)


# =====================================================================
# FOV
# =====================================================================

def test_pinhole_fov():
    """ Test FOV values for a known configuration. """
    intrinsic = av.Pinhole(1000, 800, 900, 700, 0, 0)
    h_fov = intrinsic.getHorizontalFov()
    v_fov = intrinsic.getVerticalFov()

    # hFov = 2 * atan2(sensorWidth/2, focalLengthMM_x)
    # focalLengthMM_x = sensorWidth * fx / w = 36 * 900 / 1000 = 32.4
    import math
    expected_h_fov = 2.0 * math.atan2(36.0 / 2.0, 32.4)
    assert h_fov == pytest.approx(expected_h_fov, abs=1e-10)

    # vFov = 2 * atan2(sensorHeight/2, focalLengthMM_y)
    # focalLengthMM_y = sensorHeight * fy / h = 24 * 700 / 800 = 21.0
    expected_v_fov = 2.0 * math.atan2(24.0 / 2.0, 21.0)
    assert v_fov == pytest.approx(expected_v_fov, abs=1e-10)


def test_pinhole_fov_equal_with_equal_focal():
    """ Test that hFov == vFov when focal lengths are equal and aspect ratio matches. """
    # fx = fy = 1000, w = 1000, h = 1000, sensorWidth = sensorHeight
    intrinsic = av.Pinhole(1000, 1000, 1000, 1000, 0, 0)
    intrinsic.setSensorWidth(36.0)
    intrinsic.setSensorHeight(36.0)
    assert intrinsic.getHorizontalFov() == pytest.approx(intrinsic.getVerticalFov(), abs=1e-12)


def test_pinhole_fov_changes_with_focal():
    """ Test that FOV changes after modifying the focal length. """
    intrinsic = av.Pinhole(1000, 800, 900, 900, 0, 0)
    fov_before = intrinsic.getHorizontalFov()

    intrinsic.setFocalLength(50.0, 1.0)
    fov_after = intrinsic.getHorizontalFov()

    # Larger focal => smaller FOV
    assert fov_after < fov_before


# =====================================================================
# setWidth / setHeight / setSensorWidth / setSensorHeight
# =====================================================================

def test_pinhole_set_width_height():
    """ Test setting width and height independently. """
    intrinsic = av.Pinhole()
    assert intrinsic.w() == 1 and intrinsic.h() == 1

    intrinsic.setWidth(2000)
    intrinsic.setHeight(1500)
    assert intrinsic.w() == 2000
    assert intrinsic.h() == 1500


def test_pinhole_set_sensor_dimensions():
    """ Test setting sensor width and height. """
    intrinsic = av.Pinhole()
    assert intrinsic.sensorWidth() == 36.0
    assert intrinsic.sensorHeight() == 24.0

    intrinsic.setSensorWidth(17.3)
    intrinsic.setSensorHeight(13.0)
    assert intrinsic.sensorWidth() == pytest.approx(17.3, abs=1e-12)
    assert intrinsic.sensorHeight() == pytest.approx(13.0, abs=1e-12)


# =====================================================================
# Asymmetric focal (fx != fy)
# =====================================================================

def test_pinhole_asymmetric_focal_cam2ima():
    """ Test cam2ima handles different fx and fy correctly. """
    intrinsic = av.Pinhole(1000, 800, 900, 700, 0, 0)
    p = np.array([1.0, 1.0])
    result = intrinsic.cam2ima(p)
    # cam2ima(p) = (1.0*900 + 500, 1.0*700 + 400) = (1400, 1100)
    assert result[0] == pytest.approx(1400.0, abs=1e-10)
    assert result[1] == pytest.approx(1100.0, abs=1e-10)


def test_pinhole_asymmetric_focal_ima2cam():
    """ Test ima2cam handles different fx and fy correctly. """
    intrinsic = av.Pinhole(1000, 800, 900, 700, 0, 0)
    p = np.array([1400.0, 1100.0])
    result = intrinsic.ima2cam(p)
    assert result[0] == pytest.approx(1.0, abs=1e-10)
    assert result[1] == pytest.approx(1.0, abs=1e-10)
