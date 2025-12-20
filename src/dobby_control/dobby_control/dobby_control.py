import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Vector3, Twist
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
import time

class DobbyArduControl(Node):
    def __init__(self):
        super().__init__('dobby_ardu_control')

        # QoS 설정 (MAVROS와의 호환성 고려)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # 1. 비전 노드에서 데이터 받기 (입력)
        # dobby_vision이 보내는 {x: 전진속도, y: 회전속도}를 받습니다.
        self.sub_vision = self.create_subscription(
            Vector3, 
            '/dobby/target_info', 
            self.vision_callback, 
            10
        )

        # 2. 아두파일럿 상태 구독 (안전장치)
        # 현재 시동이 걸렸는지, 모드가 GUIDED인지 확인합니다.
        self.sub_state = self.create_subscription(
            State, 
            '/mavros/state', 
            self.state_callback, 
            qos_profile
        )

        # 3. 아두파일럿에게 명령 보내기 (출력)
        # MAVROS는 'Twist' 메시지로 속도 명령을 받습니다.
        self.pub_vel = self.create_publisher(
            Twist, 
            '/mavros/setpoint_velocity/cmd_vel_unstamped', 
            qos_profile
        )

        # 4. 서비스 클라이언트 (자동 시동 및 모드 변경용)
        self.client_arming = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.client_set_mode = self.create_client(SetMode, '/mavros/set_mode')

        # 변수 초기화
        self.target_vel = 0.0       # 목표 전진 속도 (m/s)
        self.target_yaw_rate = 0.0  # 목표 회전 속도 (rad/s)
        self.current_state = State()
        self.last_vision_time = self.get_clock().now()
        
        # 0.1초마다 제어 명령 전송 (10Hz)
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("도비(아두파일럿) 제어기: 준비 완료! GUIDED 모드 대기 중...")

    def vision_callback(self, msg):
        """비전 노드로부터 목표 속도 수신"""
        self.target_vel = msg.x
        self.target_yaw_rate = msg.y
        # 마지막으로 데이터를 받은 시간 갱신 (안전장치용)
        self.last_vision_time = self.get_clock().now()

    def state_callback(self, msg):
        """아두파일럿 상태 업데이트"""
        self.current_state = msg

    def control_loop(self):
        """주기적으로 속도 명령 전송 및 상태 관리"""
        
        # [안전장치] 비전 노드가 죽었거나 1초 이상 데이터가 안 오면 정지!
        time_diff = (self.get_clock().now() - self.last_vision_time).nanoseconds / 1e9
        if time_diff > 1.0:
            self.target_vel = 0.0
            self.target_yaw_rate = 0.0
            if time_diff > 5.0: # 5초 동안 조용하면 로그 출력
                self.get_logger().warn("비전 연결 끊김! 정지합니다.", throttle_duration_sec=5)

        # [자동 설정] 연결은 됐는데 아직 GUIDED 모드가 아니거나 시동이 안 걸렸다면?
        # (주의: 로버 스위치를 켜고 안전 버튼을 누른 상태여야 함)
        if self.current_state.connected:
            if self.current_state.mode != "GUIDED":
                self.set_mode("GUIDED")
            elif not self.current_state.armed:
                # 안전을 위해 비전에서 뭔가 감지되었을 때만 시동 걸기 (옵션)
                # 여기서는 바로 시동을 겁니다. 주의하세요!
                self.set_arming(True)

        # [명령 생성]
        # 아두파일럿(ArduRover)은 Body Frame 좌표계를 자동으로 처리해줍니다.
        # linear.x = 전진 속도 (양수: 전진, 음수: 후진)
        # angular.z = 회전 속도 (양수: 좌회전, 음수: 우회전 - 좌표계 설정에 따라 다를 수 있음)
        twist = Twist()
        twist.linear.x = float(self.target_vel)
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = float(self.target_yaw_rate)

        # 명령 전송
        self.pub_vel.publish(twist)

    def set_mode(self, mode):
        """모드 변경 요청 (예: MANUAL -> GUIDED)"""
        if self.client_set_mode.wait_for_service(timeout_sec=1.0):
            req = SetMode.Request()
            req.custom_mode = mode
            future = self.client_set_mode.call_async(req)
            self.get_logger().info(f"모드 변경 시도: {mode}")

    def set_arming(self, arm):
        """시동 걸기/끄기 요청"""
        if self.client_arming.wait_for_service(timeout_sec=1.0):
            req = CommandBool.Request()
            req.value = arm
            future = self.client_arming.call_async(req)
            self.get_logger().info(f"시동 요청: {'ON' if arm else 'OFF'}")

def main(args=None):
    rclpy.init(args=args)
    node = DobbyArduControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 종료 시 안전하게 정지 명령 보냄
        stop_twist = Twist()
        node.pub_vel.publish(stop_twist)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
