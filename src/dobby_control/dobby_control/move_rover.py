import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

# PX4 메시지 타입들 import
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleStatus

class DobbyMove(Node):
    def __init__(self):
        super().__init__('dobby_move_node')

        # QoS 설정 (통신 신뢰성 확보)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 1. 퍼블리셔 설정 (명령 보내기)
        self.offboard_control_mode_publisher_ = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher_ = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher_ = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # 2. 서브스크라이버 설정 (상태 듣기)
        self.vehicle_status_subscriber_ = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v1', self.vehicle_status_callback, qos_profile)

        self.vehicle_status = VehicleStatus()
        self.offboard_setpoint_counter_ = 0

        # 0.1초마다 명령을 보내는 타이머 (10Hz)
        self.timer_ = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info("도비 로버 준비 완료! (Waiting for command...)")

    def vehicle_status_callback(self, msg):
        self.vehicle_status = msg

    def timer_callback(self):
        # 1. Offboard 모드 유지를 위한 심장박동 신호 (Heartbeat) 전송
        # 이 신호를 1초 이상 안 보내면 로버가 멈춥니다 (안전장치)
        self.publish_offboard_control_mode()

        # 2. 로버에게 이동 명령 (앞으로 1.0 m/s)
        self.publish_trajectory_setpoint()

        # 3. 초기화 단계: 시동 걸고 Offboard 모드로 전환
        if self.offboard_setpoint_counter_ == 10:
            self.engage_offboard_mode()
            self.arm()

        if self.offboard_setpoint_counter_ < 11:
            self.offboard_setpoint_counter_ += 1

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = True # 속도 제어를 하겠다
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher_.publish(msg)

    def publish_trajectory_setpoint(self):
        msg = TrajectorySetpoint()
        msg.position = [float('nan'), float('nan'), float('nan')]
        msg.velocity = [1.0, 0.0, 0.0] # [북쪽(x), 동쪽(y), 아래(z)] -> x=1.0m/s 앞으로 전진!
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher_.publish(msg)

    def engage_offboard_mode(self):
        # Offboard 모드로 변경 명령
        msg = VehicleCommand()
        msg.param1 = 1.0
        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.param2 = 6.0 # PX4_CUSTOM_MAIN_MODE_OFFBOARD
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher_.publish(msg)
        self.get_logger().info("Offboard 모드 전환 시도...")

    def arm(self):
        # 시동(Arming) 명령
        msg = VehicleCommand()
        msg.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        msg.param1 = 1.0 # 1=Arm, 0=Disarm
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher_.publish(msg)
        self.get_logger().info("시동(Arming) 명령 전송!")

def main(args=None):
    rclpy.init(args=args)
    dobby_move = DobbyMove()
    rclpy.spin(dobby_move)
    dobby_move.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
