import os
import cv2
import torch
import numpy as np
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from object_detection_interfaces.msg import DetectionArray, Detection


from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn

def load_model(checkpoint_path, image_size, device='cuda'):
    model = fasterrcnn_resnet50_fpn(pretrained=False, max_size=image_size, min_size=image_size, num_classes=4)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

class BBoxPredictor(Node):
    def __init__(self):
        super().__init__('bbox_predictor')
        # TODO: Initialize self.subscription to subscribe to Image messages from /image_raw
        self.subscription = None

        # TODO: Initialize self.detection_publisher that publishes messages of type DetectionArray, use topic "/detections"
        self.detection_publisher = None
        
        self.bridge = CvBridge()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # TODO: Load the model into self.model
        model_path = "YOUR_PATH_HERE/16280-Object-Detection/src/image_processing_pkg/image_processing_pkg/model.pth"
        image_size = None # Change this to image size your model uses!
        self.model = None # Hint: Use the load_model helper function!

        # TODO: Call self.get_logger() to log a string indicating that the predictor node was initialized
        your_string = None
        self.get_logger().info(your_string)

    def image_callback(self, msg):
        # TODO: Call self.get_logger() to log a string inidicating that an image was received for prediction

        # TODO: Use bridge.imgmsg_to_cv2 to return an image using msg as input 
        # Hint: use bgr8 as the desired_encoding
        cv_image = None
        
        image_tensor = torch.from_numpy(cv_image).permute(2, 0, 1).float()
        image_tensor = image_tensor.to(self.device)
        images = [image_tensor]

        with torch.no_grad():
            predictions = self.model(images)

        detection_msg = DetectionArray()

        # Sync timestamp with image
        detection_msg.header.stamp = msg.header.stamp  

        for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
            if score < 0.9:
                continue
            detection = Detection()
            detection.bbox = [box[0].item(), box[1].item(), box[2].item(), box[3].item()]
            detection.label = int(label.item())
            detection.score = float(score.item())
            detection_msg.detections.append(detection)

        # TODO: Have self.detect_publisher publish your detection message

        # TODO: Log a message indicating how many detections were just published

def main(args=None):
    # TODO: Call rclpy.init to initialize ROS2

    # TODO: Create the BBoxPredictor node 

    # TODO: Call rclpy.spin on the node to keep it running

    # TODO: Call rclpy.shutdown() to shutdown ROS2 

if __name__ == '__main__':
    main()