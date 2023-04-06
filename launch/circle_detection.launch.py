import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    node1 = Node(
        package='circle_detection',
        executable='circle_detection',
        name='circle_detection',
        output='screen',
        emulate_tty=True
    )

    sick_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(get_package_share_directory('sick_scan2'), 'launch'),
             '/sick_tim_5xx.launch.py']
        ),
    )
    return LaunchDescription([node1, sick_node])
