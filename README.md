**Real-Time Object Tracking and Classification in CCTV Footage**

**Overview**

This project integrates a custom hybrid deep learning model (FusionResNet) with state-of-the-art object detection and tracking techniques to perform real-time object tracking and classification in CCTV footage. The pipeline utilizes YOLO for object detection, DeepSORT for tracking, and the FusionResNet model for object classification.

**Features**

Real-Time Processing: Tracks and classifies objects in live CCTV feeds.

Custom Hybrid Model: FusionResNet combines CNN and ResNet for better feature extraction and classification.

State-of-the-Art Detection and Tracking: YOLO for fast detection and DeepSORT for robust tracking.

Handles Class Imbalance: Custom loss function with class weights.

**System Requirements**

**Hardware**

GPU (Recommended): NVIDIA GPU with CUDA support for efficient processing.

CPU: Minimum quad-core processor (performance may vary).

Memory: At least 8 GB RAM.

**Software**

Python 3.8+

PyTorch 1.9+

OpenCV 4.5+

**Pipeline Description**

Object Detection: YOLO detects objects in each frame of the CCTV footage.

Object Tracking: DeepSORT tracks detected objects across frames.

Classification: FusionResNet classifies objects into predefined categories.
