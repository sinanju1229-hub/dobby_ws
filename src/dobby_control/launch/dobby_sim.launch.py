import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        
        # 1. 영상 연결 다리 (Gazebo -> ROS 2)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=['/camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image'],
            output='screen'
        ),

        # 2. 비전 모듈 (눈: YOLO + Logic)
        Node(
            package='dobby_control',
            executable='dobby_vision',
            name='dobby_vision',
            output='screen'
        ),

        # 3. 제어 모듈 (다리: PX4 Control)
        Node(
            package='dobby_control',
            executable='dobby_control',
            name='dobby_control',
            output='screen'
        ),
    ])
