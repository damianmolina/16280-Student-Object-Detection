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
        self.subscription = self.create_subscription(Image, '/image_raw', self.image_callback, 10)

        # TODO: Initialize self.detection_publisher that publishes messages of type DetectionArray, use topic "/detections"
        self.detection_publisher = self.create_publisher(DetectionArray, '/detections', 10)
        
        self.bridge = CvBridge()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # TODO: Load the model into self.model
        model_path = "/Users/damianmolina/Downloads/16280-Student-Object-Detection/src/image_processing_pkg/image_processing_pkg/model.pth"
        image_size = 244 # Change this to image size your model uses!
        self.model = load_model(model_path, image_size, self.device) # Hint: Use the load_model helper function!

        # TODO: Call self.get_logger() to log a string indicating that the predictor node was initialized
        your_string = "Predictor Node Initialized"
        self.get_logger().info(your_string)

    def image_callback(self, msg):
        # TODO: Call self.get_logger() to log a string inidicating that an image was received for prediction
        your_string = "Image Received for Prediction"
        self.get_logger().info(your_string)

        # TODO: Use bridge.imgmsg_to_cv2 to return an image using msg as input 
        # Hint: use bgr8 as the desired_encoding
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
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
        self.detection_publisher.publish(detection_msg)

        # TODO: Log a message indicating how many detections were just published
        your_string = f"{len(detection_msg.detections)} detections published"
        self.get_logger().info(your_string)

def main(args=None):
    # TODO: Call rclpy.init to initialize ROS2
    rclpy.init(args=args)

    # TODO: Create the BBoxPredictor node 
    bbox_predictor = BBoxPredictor()

    # TODO: Call rclpy.spin on the node to keep it running
    rclpy.spin(bbox_predictor)

    # TODO: Call rclpy.shutdown() to shutdown ROS2 
    rclpy.shutdown()

if __name__ == '__main__':
    main()