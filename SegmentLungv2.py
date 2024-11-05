import logging
from typing import Optional

import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
)

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    slicer.util.pip_install("matplotlib")

try:
    from sklearn import cluster
except ModuleNotFoundError:
    slicer.util.pip_install("scikit-learn")

try:
    from scipy.ndimage import distance_transform_edt
except ModuleNotFoundError:
    slicer.util.pip_install("scipy")

try:
    import cv2
except ModuleNotFoundError:
    slicer.util.pip_install("opencv-python")

try:
    from skimage import measure
    from skimage import morphology
    from skimage.segmentation import watershed
except ModuleNotFoundError:
    slicer.util.pip_install("scikit-image")

from slicer import vtkMRMLScalarVolumeNode, vtkMRMLSegmentationNode


#
# SegmentLungv2
#


class SegmentLungv2(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Segment Lung")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []
        self.parent.contributors = [
            "Konrad Reczko (AGH)", "Joanna Wojcicka (AGH)"]
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is a simple lung segmentation module.
""")
        self.parent.acknowledgementText = _("""
Thanks mom :D
""")


#
# SegmentLungv2ParameterNode
#


@parameterNodeWrapper
class SegmentLungv2ParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    totalSegmentator: vtkMRMLSegmentationNode
    groundTruthVolume: vtkMRMLScalarVolumeNode


#
# SegmentLungv2Widget
#


class SegmentLungv2Widget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SegmentLungv2.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = SegmentLungv2Logic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[SegmentLungv2ParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume:
            self.ui.applyButton.toolTip = _("Compute output volume")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input volume node")
            self.ui.applyButton.enabled = False

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            self.logic.process(
                self.ui.inputSelector.currentNode(),
                self.ui.totalSegmentatorSelector.currentNode(),
                self.ui.groundTruthSelector.currentNode()
            )


#
# SegmentLungv2Logic
#

def __kmeans_clusterization(img: np.ndarray, n_clusters: int):
    img = np.where(img > 0, 1, 0)
    markers = np.stack(np.where(img > 0), axis=-1)

    clusters = cluster.KMeans(n_clusters=n_clusters, random_state=0)
    points = clusters.fit_predict(X=markers)

    return points, clusters.cluster_centers_


def watershed_segmentation(img: np.ndarray, kmeans_centers):
    img = np.where(img > 0, 1, 0)
    markers = np.zeros(img.shape, dtype=np.int32)

    for i, center in enumerate(kmeans_centers):
        cnter_int = tuple(np.round(center).astype(int))

        if 0 <= cnter_int[0] < img.shape[0] and 0 <= cnter_int[1] < img.shape[1] and 0 <= cnter_int[2] < img.shape[2]:
            markers[cnter_int] = i + 1

    distance = distance_transform_edt(img)
    return watershed(-distance, markers, mask=img)


def get_lung_mask(foo):
    lung_mask = np.array(foo)
    binary_image = np.where(foo < -320, 0, 1)

    for i in range(0, foo.shape[2]):
        # Remove small objects (air bubbles and noise) using morphological opening
        cleaned_image = morphology.remove_small_objects(binary_image[:, :, i].astype(bool), min_size=500)

        # Invert image
        cleaned_image = np.where(cleaned_image == 0, 1, 0)
        cleaned_image = cv2.erode(cleaned_image.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
        labels = measure.label(cleaned_image)

        # Find the largest two components (the lungs)
        props = measure.regionprops(labels)
        props = sorted(props, key=lambda x: x.area, reverse=True)

        # Create an empty mask and fill it with the two largest regions (lungs)
        lung_mask[:, :, i] = np.zeros_like(labels)
        for prop in props[1:3]:
            lung_mask[:, :, i][labels == prop.label] = 1
            lung_mask[:, :, i] = cv2.dilate(lung_mask[:, :, i].astype(np.uint8),
                                            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
            lung_mask[:, :, i] = cv2.morphologyEx(lung_mask[:, :, i], cv2.MORPH_CLOSE,
                                                  cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

        # label the right and left lung
        right_lung = np.zeros_like(labels)
        left_lung = np.zeros_like(labels)
        for prop in props[1:3]:
            if prop.centroid[1] > foo.shape[1] / 2:
                right_lung[labels == prop.label] = 2
            else:
                left_lung[labels == prop.label] = 1

        lung_mask[:, :, i] = right_lung + left_lung

    _, centroids = __kmeans_clusterization(lung_mask, 2)
    watersh = watershed_segmentation(lung_mask, centroids)
    if np.unique(watersh).shape[0] > 2:
        lung_mask = watersh

    return lung_mask


def compute_dice_coefficient(mask_gt, mask_pred):
    """Computes soerensen-dice coefficient.

  compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
  and the predicted mask `mask_pred`.

  Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.

  Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN.
  """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum


class SegmentLungv2Logic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return SegmentLungv2ParameterNode(super().getParameterNode())

    def process(self, inputVolume: vtkMRMLScalarVolumeNode, totalSegmentator: vtkMRMLSegmentationNode = None,
                groundTruthVolume: vtkMRMLScalarVolumeNode = None) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be segmented
        """

        import time

        startTime = time.time()
        logging.info("Processing started")

        # Create segmentation
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentationNode.CreateDefaultDisplayNodes()  # only needed for display
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(inputVolume)

        # Create segment for each lung
        segmentL = segmentationNode.GetSegmentation().AddEmptySegment()
        segmentR = segmentationNode.GetSegmentation().AddEmptySegment()
        foo = slicer.util.arrayFromVolume(inputVolume)

        # Swap the x axis with the z axis
        foo = np.swapaxes(foo, 0, 2)
        # Iterate over the slices in the volume
        lung_mask = get_lung_mask(foo)
        lung_mask = np.swapaxes(lung_mask, 0, 2)

        # Separate the left and right lung (based on label)
        left_lung = np.where(lung_mask == 1, 1, 0).astype(np.uint8)
        right_lung = np.where(lung_mask == 2, 1, 0).astype(np.uint8)

        slicer.util.updateSegmentBinaryLabelmapFromArray(left_lung, segmentationNode, segmentL, inputVolume)
        slicer.util.updateSegmentBinaryLabelmapFromArray(right_lung, segmentationNode, segmentR, inputVolume)

        segmentationNode.CreateClosedSurfaceRepresentation()

        if groundTruthVolume is not None:
            gt = slicer.util.arrayFromVolume(groundTruthVolume)

            left_lung_dice = compute_dice_coefficient(np.where(gt == 2, 1, 0), left_lung.astype(bool))
            right_lung_dice = compute_dice_coefficient(np.where(gt == 3, 1, 0), right_lung.astype(bool))
            print("Predicted segmentations")
            print(f"Left lung dice coefficient: {left_lung_dice}")
            print(f"Right lung dice coefficient: {right_lung_dice}")

            if totalSegmentator is not None:
                segment_ids = totalSegmentator.GetSegmentation().GetSegmentIDs()
                left_lung_ids = ['lung_upper_lobe_left', 'lung_middle_lobe_left', 'lung_lower_lobe_left']
                right_lung_ids = ['lung_upper_lobe_right', 'lung_middle_lobe_right', 'lung_lower_lobe_right']

                left_lung_total = np.zeros_like(left_lung).astype(bool)
                right_lung_total = np.zeros_like(right_lung).astype(bool)

                for segment_id in segment_ids:
                    if segment_id in left_lung_ids:
                        left_lung_total = np.logical_or(left_lung_total, slicer.util.arrayFromSegmentBinaryLabelmap(totalSegmentator, segment_id, inputVolume).astype(bool))
                    elif segment_id in right_lung_ids:
                        right_lung_total = np.logical_or(right_lung_total, slicer.util.arrayFromSegmentBinaryLabelmap(totalSegmentator, segment_id, inputVolume).astype(bool))

                left_lung_dice = compute_dice_coefficient(np.where(gt == 2, 1, 0).astype(bool), left_lung_total)
                right_lung_dice = compute_dice_coefficient(np.where(gt == 3, 1, 0).astype(bool), right_lung_total)
                print("Total segmentator")
                print(f"Left lung dice coefficient: {left_lung_dice}")
                print(f"Right lung dice coefficient: {right_lung_dice}")

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime - startTime:.2f} seconds")
