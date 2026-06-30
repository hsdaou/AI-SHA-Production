import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan, PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Header
import numpy as np
from scipy.spatial.transform import Rotation
import sensor_msgs_py.point_cloud2 as pc2


class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')
        
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/LiDAR/LD19', self.lidar_callback, 10)
        self.depth_sub = self.create_subscription(PointCloud2, '/camera/depth/points', self.depth_callback, 10)
        # self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        self.pose_pub = self.create_publisher(PoseStamped, '/pose_estimate', 10)
        self.state_pub = self.create_publisher(Odometry, '/robot_state', 10)
        
        self.timer = self.create_timer(0.05, self.fusion_update)
        
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self.angular_velocity = np.array([0.0, 0.0, 0.0])
        
        self.imu_data = None
        self.lidar_data = None
        self.depth_data = None
        # self.odom_data = None
        
        self.last_update_time = self.get_clock().now()
        
        self.position_covariance = np.eye(3) * 0.1
        self.velocity_covariance = np.eye(3) * 0.01
        
    def imu_callback(self, msg):
        self.imu_data = msg
        q = msg.orientation
        self.orientation = np.array([q.w, q.x, q.y, q.z])
        w = msg.angular_velocity
        self.angular_velocity = np.array([w.x, w.y, w.z])
        
    def lidar_callback(self, msg):
        self.lidar_data = msg
        
    def depth_callback(self, msg):
        self.depth_data = msg
        
    # def odom_callback(self, msg):
    #     self.odom_data = msg
        
    def fusion_update(self):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_update_time).nanoseconds / 1e9
        self.last_update_time = current_time
        
        if self.imu_data is not None:
            linear_accel = np.array([
                self.imu_data.linear_acceleration.x,
                self.imu_data.linear_acceleration.y,
                self.imu_data.linear_acceleration.z
            ])
            
            rot = Rotation.from_quat([self.orientation[1], self.orientation[2], 
                                      self.orientation[3], self.orientation[0]])
            world_accel = rot.apply(linear_accel)
            world_accel[2] -= 9.81
            
            self.velocity += world_accel * dt
            self.position += self.velocity * dt
            
        # if self.odom_data is not None:
        #     odom_pos = np.array([
        #         self.odom_data.pose.pose.position.x,
        #         self.odom_data.pose.pose.position.y,
        #         self.odom_data.pose.pose.position.z
        #     ])
        #     
        #     odom_vel = np.array([
        #         self.odom_data.twist.twist.linear.x,
        #         self.odom_data.twist.twist.linear.y,
        #         self.odom_data.twist.twist.linear.z
        #     ])
        #     
        #     alpha = 0.7
        #     self.position = alpha * odom_pos + (1 - alpha) * self.position
        #     self.velocity = alpha * odom_vel + (1 - alpha) * self.velocity
            
        if self.lidar_data is not None:
            ranges = np.array(self.lidar_data.ranges)
            valid_ranges = ranges[np.isfinite(ranges)]
            if len(valid_ranges) > 0:
                min_distance = np.min(valid_ranges)
                if min_distance < 0.3:
                    self.velocity *= 0.8
                    
        pose_msg = PoseStamped()
        pose_msg.header = Header()
        pose_msg.header.stamp = current_time.to_msg()
        pose_msg.header.frame_id = 'map'
        
        pose_msg.pose.position.x = self.position[0]
        pose_msg.pose.position.y = self.position[1]
        pose_msg.pose.position.z = self.position[2]
        
        pose_msg.pose.orientation.w = self.orientation[0]
        pose_msg.pose.orientation.x = self.orientation[1]
        pose_msg.pose.orientation.y = self.orientation[2]
        pose_msg.pose.orientation.z = self.orientation[3]
        
        self.pose_pub.publish(pose_msg)
        
        state_msg = Odometry()
        state_msg.header = pose_msg.header
        state_msg.child_frame_id = 'base_link'
        state_msg.pose.pose = pose_msg.pose
        
        state_msg.twist.twist.linear.x = self.velocity[0]
        state_msg.twist.twist.linear.y = self.velocity[1]
        state_msg.twist.twist.linear.z = self.velocity[2]
        
        state_msg.twist.twist.angular.x = self.angular_velocity[0]
        state_msg.twist.twist.angular.y = self.angular_velocity[1]
        state_msg.twist.twist.angular.z = self.angular_velocity[2]
        
        self.state_pub.publish(state_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
