import rclpy
import numpy as np
import math
import yaml
from rclpy.node import Node
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Joy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry


from tf2_ros import transform_broadcaster
from tf2_ros.buffer import Buffer
from tf2_ros import transform_listener
import math


import circle_detection.submodules.circle_analyze as ca
from dataclasses import dataclass


@dataclass
class pose:
    x: float
    y: float
    theta: float


def quaternion_from_euler(ai, aj, ak):
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((4, ))
    q[0] = cj*sc - sj*cs
    q[1] = cj*ss + sj*cc
    q[2] = cj*cs - sj*sc
    q[3] = cj*cc + sj*ss

    return q


class CircleDetector(Node):
    def __init__(self):
        super().__init__('circle_detection')
        self.set_handles()
        self.set_arguments()
        self.read_yaml()
        self.tf_broadcaster = TransformStamped()
        self.robot_pose = pose(0, 0, 0)
        self.flag = False
        self.timer = self.create_timer(0.05, self.update)

    def set_handles(self):
        '''set publisher and subscriber'''
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.scan_sub
        self.flag_sub = self.create_subscription(Bool, 'flag', self.flag_callback, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = transform_listener.TransformListener(self.tf_buffer, self)

    def set_arguments(self):
        self.cartesian = []  # 点群の座標(ロボット視点)
        self.circle_place = dict()  # マップ上の円の座標

    def read_yaml(self):
        self.circle_place = dict()
        with open('/home/yugoosada/ros2_ws/src/circle_detection/map/map.yaml') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            self.circle_place = config['circles']

    def flag_callback(self, msg):
        self.flag = msg.data

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

    def get_current_pose(self):
        '''tf2から現在のロボットの座標を取得'''
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            self.robot_pose.x = t.transform.translation.x
            self.robot_pose.y = t.transform.translation.y
            self.robot_pose.theta = math.atan2(2.0 * (t.transform.rotation.x*t.transform.rotation.y+t.transform.rotation.w*t.transform.rotation.z),
                                               t.transform.rotation.w*t.transform.rotation.w+t.transform.rotation.x*t.transform.rotation.x -
                                               t.transform.rotation.y*t.transform.rotation.y-t.transform.rotation.z*t.transform.rotation.z)

        except Exception as e:
            self.get_logger().info('tf2 error: {}'.format(e))

    def coordinate_transform(self, x, y):
        '''座標変換'''
        X = self.robot_pose.x
        Y = self.robot_pose.y
        W = self.robot_pose.theta
        return [X+x*np.cos(W)-y*np.sin(W), Y+x*np.sin(W)+y*np.cos(W)]

    def figure_detect(self, local_cartesian):
        '''円と直線の検出'''
        clustor = np.empty([0, 0])
        circles = np.empty([0, 0, 0])
        circles_map = np.empty([0, 0, 0])
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
                        X = self.coordinate_transform(circle[0], circle[1])
                        circles_map = np.append(circles_map, [X[0], X[1], circle[2]])
                        circles = np.reshape(circles, (-1, 3))
                        circles_map = np.reshape(circles_map, (-1, 3))
                    elif r > 0.9 and ca.calc_distance(np.reshape(line, [2, 2])) > 0.1:
                        # 決定係数が0.9以上で長さが10cm以上のものを直線と判断
                        lines = np.append(lines, line)
                        lines = np.reshape(lines, [-1, 2, 2])
                clustor = np.empty([0, 0])

        return circles, circles_map, lines

    def triangluration(self, map_circles, lidar_circles):
        map_vector = map_circles[1]-map_circles[0]
        lidar_vector = lidar_circles[1]-lidar_circles[0]
        map_arg = np.arctan2(map_vector[1], map_vector[0])
        lidar_arg = np.arctan2(lidar_vector[1], lidar_vector[0])
        arg = map_arg-lidar_arg
        J = np.array([[np.cos(arg), -np.sin(arg)], [np.sin(arg), np.cos(arg)]])
        actual_place = map_circles[0].reshape(-1, 1) + np.dot(J, -lidar_circles[0].reshape(-1, 1))
        return actual_place.reshape(1, -1), arg

    def dist(self, A, B):
        return math.sqrt((A[0]-B[0])**2+(A[1]-B[1])**2)

    def fitting(self, map, data, nya):
        ans = np.empty((0, 2, 2))
        index = list()
        for i in range(len(data)):
            for j in range(len(map)):
                if self.dist(data[i], map[j]['place']) < 0.5:
                    a = [[map[j]['place'][0], map[j]['place'][1]], nya[i][0:2]]
                    ans = np.append(ans, a)
                    ans = np.reshape(ans, (-1, 2, 2))
                    index.append(map[j]['name'])

        return ans, index

    def write_tf(self, true_pose):
        t = TransformStamped()

        # Read message content and assign it to
        # corresponding tf variables
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'lidar_link'

        # Turtle only exists in 2D, thus we get x and y translation
        # coordinates from the message and set the z coordinate to 0
        t.transform.translation.x = true_pose[0]
        t.transform.translation.y = true_pose[1]
        t.transform.translation.z = 0.0

        # For the same reason, turtle can only rotate around one axis
        # and this why we set rotation in x and y to 0 and obtain
        # rotation in z axis from the message
        q = quaternion_from_euler(0, 0, true_pose[2])
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)

    def update(self):
        # 現在位置を取得
        self.get_current_pose()
        # self.robot_pose.x = -1.3
        # self.robot_pose.y = 0
        # self.robot_pose.theta = 0

        # 点群は常に取得
        # 点群から円を検出
        circles, circles_map, lines = self.figure_detect(self.cartesian)

        # print('circles')
        # print(circles)
        # print('circles_map')
        # print(circles_map)
        # 検出された円をマップをもとにラベリング
        fit_ans, index = self.fitting(self.circle_place, circles_map, circles)
        # print(fit_ans)

        # ラベリングされた円から機体の位置を逆算
        true_pose = [0, 0, 0]
        # print(len(fit_ans))
        if len(fit_ans) >= 2:
            for i in range(len(fit_ans)):
                if not i == 0:
                    map_circle = np.array([fit_ans[i-1][0], fit_ans[i][0]])
                    lidar_circle = np.array([fit_ans[i-1][1], fit_ans[i][1]])
                    print('map_circle')
                    print(map_circle)
                    print('lidar_circle')
                    print(lidar_circle)
                    place, arg = self.triangluration(map_circle, lidar_circle)
                    # print('place')
                    # print(place)
                    true_pose[0] += place[0][0] / (float(len(fit_ans))-1.0)
                    true_pose[1] += place[0][1] / (float(len(fit_ans))-1.0)
                    true_pose[2] += arg / (float(len(fit_ans))-1.0)

        if self.flag:
            self.write_tf(true_pose)

        print('true_pose')
        print(true_pose)


def main(args=None):
    rclpy.init(args=args)
    circledetector = CircleDetector()
    rclpy.spin(circledetector)
    circledetector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
