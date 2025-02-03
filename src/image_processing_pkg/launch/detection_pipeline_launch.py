from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='image_processing_pkg',
            executable='bbox_predictor',
            name='bbox_predictor',
            output='screen'
        ),
        Node(
            package='image_processing_pkg',
            executable='bbox_visualizer',
            name='bbox_visualizer',
            output='screen'
        ),
    ])
