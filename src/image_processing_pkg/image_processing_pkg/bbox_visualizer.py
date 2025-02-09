import cv2
import numpy as np
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from object_detection_interfaces.msg import DetectionArray
import message_filters # For Sync!

class BBoxVisualizer(Node):
    def __init__(self):
        super().__init__('bbox_visualizer')
        
        self.bridge = CvBridge()
        
        # TODO: Fill in the message types and topics for the following two Subscribers 
        # Note: we are using special Subscribers for message synchronization between Images and Detections
        # Hint: See the diagram in the writeup for what the BBoxVisualizer node should subscribe to! 
        self.image_sub = message_filters.Subscriber(self, Image, '/image_raw')
        self.detection_sub = message_filters.Subscriber(self, DetectionArray, '/detections')
        
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.detection_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.callback)

        # TODO: create a publisher that publishes Image type messages to topic /image_with_bboxes
        self.publisher = self.create_publisher(Image, '/image_with_bboxes', 10)

        self.labels_dict = {1: "blue", 2: "green", 3: "red"}
        self.label_colors = {
            "blue": (255, 0, 0),
            "green": (0, 255, 0),
            "red": (0, 0, 255)
        }

        # TODO: Log information that the BBox Visualizer Node was Initialized
        self.get_logger().info("BBox Visualizer node started")


    def callback(self, image_msg, detections_msg):
        # TODO: Use bridge.imgmsg_to_cv2() to convert the Image message to cv2 format 
        # Hint: use bgr8 as the desired encoding
        cv_image = CvBridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        
        for detection in detections_msg.detections:
            x_min, y_min, x_max, y_max = map(int, detection.bbox)
            class_label = self.labels_dict.get(detection.label, "unknown")
            color = self.label_colors.get(class_label, (255, 255, 255))
            cv2.rectangle(cv_image, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(cv_image, f"{class_label}: {detection.score:.2f}", (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # TODO: Using self.bridge.cv2_to_imgmsg(), convert the processed image (cv_image) back to an Image message
        # Hint: Use "bgr8" for the encoding
        processed_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')

        # TODO: Publish your processed image message
        self.publisher.publish(processed_msg)

        # TODO: Log information stating that the image with bboxes has been published
        self.get_logger().info("Image with BBoxes Published")

def main(args=None):
    # TODO: Call rclpy.init to initialize ROS2
    rclpy.init(args=args)

    # TODO: Create the BBoxVisualizer node 
    bbox_visualizer = BBoxVisualizer()

    # TODO: Call rclpy.spin on the node to keep it running
    rclpy.spin(bbox_visualizer)

    # TODO: Call rclpy.shutdown() to shutdown ROS2 properly
    rclpy.shutdown()

if __name__ == '__main__':
    main()