import rclpy
import numpy as np
import math
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from tf2_ros import transform_broadcaster
from tf2_ros.buffer import Buffer
from tf2_ros import transform_listener

import circle_detection.submodules.circle_analyze as ca


class CircleDetector(Node):
    def __init__(self):
        super().__init__('circle_detection')
        self.set_handles()
        self.set_arguments()
        self.timer = self.create_timer(0.05, self.update)
        pass

    def set_handles(self):
        '''set publisher and subscriber'''
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.scan_sub
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.odom_sub

    def set_arguments(self):
        self.cartesian = []
        pass

    def scan_callback(self, msg):
        self.cartesian = []
        for i in range(len(msg.ranges)):
            angle = msg.angle_min+i*msg.angle_increment
            if msg.ranges[i] < 6 and -np.arccos(42.5/188) < angle < np.arccos(42.5/188):
                # 射程範囲内の点群を抽出・座標変換
                point = [msg.ranges[i]*np.cos(msg.angle_min+i*msg.angle_increment),
                         msg.ranges[i]*np.sin(msg.angle_min+i*msg.angle_increment)]
                self.cartesian = np.append(self.cartesian, point)
                self.cartesian = np.reshape(self.cartesian, (-1, 2))

    def odom_callback(self, msg):
        pass

    def figure_detect(self, local_cartesian):
        '''円と直線の検出'''
        clustor = np.empty([0, 0])
        circles = np.empty([0, 0, 0])
        lines = np.empty([0, 0, 0, 0])

        # 点群をクラスタ化しながら円検出
        for i in range(len(local_cartesian)-1):
            if ca.calc_distance(local_cartesian[i:i+2]) < 0.1:
                # 隣接点との距離が近かったらクラスタに含める
                clustor = np.append(clustor, local_cartesian[i])
                clustor = np.reshape(clustor, (-1, 2))
            else:
                # 隣接点との距離が遠かったらクラスタの切れ目と判断
                if len(clustor) > 2:
                    # クラスタが3点以上なら円・直線検出
                    circle = ca.circle_fitting(clustor[:, 0], clustor[:, 1])
                    line, r = ca.line_fitting(clustor[:, 0], clustor[:, 1])
                    if 0.03 < circle[2] < 0.10:
                        # 半径が4cm~8cmならポール円と判断
                        circles = np.append(circles, circle)
                        circles = np.reshape(circles, (-1, 3))
                    elif r > 0.9 and ca.calc_distance(np.reshape(line, [2, 2])) > 0.1:
                        # 決定係数が0.9以上で長さが10cm以上のものを直線と判断
                        lines = np.append(lines, line)
                        lines = np.reshape(lines, [-1, 2, 2])
                clustor = np.empty([0, 0])

        return circles, lines

    def get_current_pose(self):
        '''tf2から現在のロボットの座標を取得'''
        
        pass
        
    def update(self):
        circles, lines = self.figure_detect(self.cartesian)
        
        print("circle: ")
        print(circles)
        print("lines :")
        print(lines)


def main(args=None):
    rclpy.init(args=args)
    circledetector = CircleDetector()
    rclpy.spin(circledetector)
    circledetector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
