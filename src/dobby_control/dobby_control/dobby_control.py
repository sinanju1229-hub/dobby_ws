import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Vector3
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleOdometry
import math

class DobbyControl(Node):
    def __init__(self):
        super().__init__('dobby_control_node')
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 1. 비전 명령 구독
        self.sub_cmd = self.create_subscription(Vector3, '/dobby/target_info', self.cmd_callback, 10)
        
        # 2. 로버의 현재 자세(Heading) 구독 [추가됨!]
        self.sub_odom = self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_callback, qos_profile)
        
        # 3. PX4 명령 퍼블리셔
        self.pub_offboard = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.pub_traj = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.pub_vehicle_command = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        self.target_vel = 0.0
        self.target_yaw_rate = 0.0
        self.current_yaw = 0.0 # 현재 로버가 보고 있는 방향
        self.offboard_setpoint_counter = 0

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("도비 제어기(V2): 방향 보정 기능 탑재 완료!")

    def cmd_callback(self, msg):
        self.target_vel = msg.x
        self.target_yaw_rate = msg.y

    def odom_callback(self, msg):
        # 쿼터니언 -> 오일러(Yaw) 변환
        q = msg.q
        # yaw (z-axis rotation)
        siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
        cosy_cosp = 1 - 2 * (q[2] * q[2] + q[3] * q[3])
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def control_loop(self):
        # Heartbeat
        msg_offboard = OffboardControlMode()
        msg_offboard.position = False
        msg_offboard.velocity = True
        msg_offboard.acceleration = False
        msg_offboard.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_offboard.publish(msg_offboard)
        
        # [핵심 수정] 몸체 기준 속도를 지도 기준 속도(NED)로 변환
        # 로버가 북쪽(0도)을 보고 있을 때: vx = speed
        # 로버가 동쪽(90도)을 보고 있을 때: vy = speed
        vx = self.target_vel * math.cos(self.current_yaw)
        vy = self.target_vel * math.sin(self.current_yaw)

        msg_traj = TrajectorySetpoint()
        msg_traj.position = [float('nan'), float('nan'), float('nan')]
        msg_traj.velocity = [vx, vy, 0.0]   # 계산된 벡터 적용
        msg_traj.yawspeed = self.target_yaw_rate # 회전 속도 적용
        
        msg_traj.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_traj.publish(msg_traj)

        # 초기화 및 시동
        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()
        if self.offboard_setpoint_counter < 11:
            self.offboard_setpoint_counter += 1

    def engage_offboard_mode(self):
        msg = VehicleCommand()
        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        msg.param1 = 1.0
        msg.param2 = 6.0
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_vehicle_command.publish(msg)

    def arm(self):
        msg = VehicleCommand()
        msg.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        msg.param1 = 1.0
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.pub_vehicle_command.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = DobbyControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()