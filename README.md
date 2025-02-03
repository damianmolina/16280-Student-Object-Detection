# 16280-Student-Object-Detection

## Setup Instructions

1. **Fill in the necessary code** for the following scripts:  
   - `bbox_predictor.py`  
   - `bbox_visualizer.py`  

2. **Build the required packages** using:  
   ```sh
   colcon build --packages-select image_processing_pkg object_detection_interfaces
   ```  

3. **Source the setup file**:  
   ```sh
   source install/setup.bash
   ```  

## Running Object Detection

To perform object detection using TurtleBot images:  

1. **Start the camera node** on your TurtleBot:  
   _(Ensure the TurtleBot is powered on and the camera node is running in a separate terminal.)_  

2. **Launch the detection pipeline** on your VM:  
   ```sh
   ros2 launch image_processing_pkg detection_pipeline_launch.py
   ```  

This will initiate the object detection process using images captured by the TurtleBot.
