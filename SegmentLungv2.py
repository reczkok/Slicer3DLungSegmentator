import logging
import os
from typing import Annotated, Optional

import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

import numpy as np

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

from slicer import vtkMRMLScalarVolumeNode


#
# SegmentLungv2
#


class SegmentLungv2(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("SegmentLungv2")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = [
            "John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#SegmentLungv2">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # SegmentLungv21
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="SegmentLungv2",
        sampleName="SegmentLungv21",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "SegmentLungv21.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="SegmentLungv21.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="SegmentLungv21",
    )

    # SegmentLungv22
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="SegmentLungv2",
        sampleName="SegmentLungv22",
        thumbnailFileName=os.path.join(iconsPath, "SegmentLungv22.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="SegmentLungv22.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="SegmentLungv22",
    )


#
# SegmentLungv2ParameterNode
#


@parameterNodeWrapper
class SegmentLungv2ParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode


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
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.thresholdedVolume:
            self.ui.applyButton.toolTip = _("Compute output volume")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input and output volume nodes")
            self.ui.applyButton.enabled = False

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(),
                               self.ui.imageThresholdSliderWidget.value, self.ui.invertOutputCheckBox.checked)


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

        #label the right and left lung
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

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputVolume: vtkMRMLScalarVolumeNode = None,
                imageThreshold: float = 100,
                invert: bool = False,
                showResult: bool = True) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
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

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime - startTime:.2f} seconds")


#
# SegmentLungv2Test
#


class SegmentLungv2Test(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_SegmentLungv21()

    def test_SegmentLungv21(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("SegmentLungv21")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = SegmentLungv2Logic()

        logic.process(inputVolume, outputVolume, threshold, True)

        self.delayDisplay("Test passed")
