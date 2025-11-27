import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3 # 통신용 메시지 (x:전진, y:회전, z:상태)
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import mediapipe as mp
import time

class DobbyVision(Node):
    def __init__(self):
        super().__init__('dobby_vision_node')

        # === [설정] ===
        self.img_center_x = 320     # 화면 중심
        self.k_yaw = 0.002          # 회전 반응 감도
        self.k_speed = 0.5          # 전진 반응 감도
        self.stop_height = 300      # 정지 거리 기준 (픽셀 높이)

        self.master_id = None
        self.wave_time_threshold = 3.0
        self.wave_histories = {}
        self.master_last_seen = 0

        # === [통신] ===
        # 카메라 구독
        self.sub_img = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 1)
        # 명령 발행 (다리 노드에게 보냄)
        self.pub_cmd = self.create_publisher(Vector3, '/dobby/target_info', 10)

        self.bridge = CvBridge()
        self.yolo = YOLO('yolov8n.pt')

        # MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

        self.get_logger().info("눈(Vision) 모듈 가동! 주인님을 찾습니다.")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            current_time = time.time()

            # 전송할 데이터 초기화 (x=속도, y=회전, z=상태 0:대기/1:추적)
            cmd_msg = Vector3()
            cmd_msg.x = 0.0
            cmd_msg.y = 0.0
            cmd_msg.z = 0.0 

            # YOLO 트래킹
            results = self.yolo.track(frame, persist=True, verbose=False, classes=0)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # [모드 1: 주인 추적]
                if self.master_id is not None:
                    if self.master_id in track_ids:
                        idx = track_ids.index(self.master_id)
                        box = boxes[idx]
                        self.master_last_seen = current_time

                        # 이동 명령 계산
                        vel, yaw = self.calculate_command(box)
                        cmd_msg.x = float(vel)
                        cmd_msg.y = float(yaw)
                        cmd_msg.z = 1.0 # 추적 중

                        self.draw_box(frame, box, "MASTER", (0, 255, 0))
                    else:
                        # 놓침 (5초간 대기)
                        if current_time - self.master_last_seen > 5.0:
                            self.master_id = None

                # [모드 2: 주인 탐색]
                else:
                    for box, track_id in zip(boxes, track_ids):
                        # 손 흔드는지 확인
                        if self.check_waving(frame, box, track_id, current_time):
                            self.master_id = track_id
                            self.master_last_seen = current_time
                            self.get_logger().info(f"주인님 등록! ID: {track_id}")
                            break

            # 계산된 명령을 제어 모듈로 전송!
            self.pub_cmd.publish(cmd_msg)

            cv2.imshow("Dobby Vision Eye", frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"비전 에러: {e}")

    def calculate_command(self, box):
        x1, y1, x2, y2 = map(int, box)
        center_x = (x1 + x2) / 2
        height = y2 - y1

        # 중심 맞추기 (회전)
        error_x = self.img_center_x - center_x
        cmd_yaw = self.k_yaw * error_x

        # 거리 맞추기 (속도)
        if height < self.stop_height:
            cmd_vel = self.k_speed * (1.0 - (height / 480.0))
            cmd_vel = min(cmd_vel, 1.0)
        else:
            cmd_vel = 0.0

        return cmd_vel, cmd_yaw

    def check_waving(self, frame, box, track_id, current_time):
        # [테스트용] 무조건 흔드는 것으로 간주 (실제 사용 시엔 Pose 로직 적용)
        is_waving = True 

        if track_id not in self.wave_histories:
            self.wave_histories[track_id] = {'start_time': None}
        record = self.wave_histories[track_id]

        if is_waving:
            if record['start_time'] is None: record['start_time'] = current_time
            duration = current_time - record['start_time']
            self.draw_box(frame, box, f"Wait: {duration:.1f}", (0, 255, 255))
            if duration >= self.wave_time_threshold: return True
        else:
            record['start_time'] = None
            self.draw_box(frame, box, "Person", (0, 0, 255))
        return False

    def draw_box(self, frame, box, text, color):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def main(args=None):
    rclpy.init(args=args)
    node = DobbyVision()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
