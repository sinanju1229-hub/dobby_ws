import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import mediapipe as mp
import time

class DobbyVision(Node):
    def __init__(self):
        super().__init__('dobby_vision_node')
        
        # === [튜닝된 설정: 더 가까이, 더 빠르게] ===
        self.img_center_x = 320     
        self.center_tolerance = 60  # [수정] 40->60 (너무 예민하게 멈추지 않게 완화)
        
        self.k_yaw = 0.02           # [수정] 0.005->0.02 (회전 반응 4배 강화!)
        self.k_speed = 1.0          # [수정] 0.8->1.0 (최대 속도 증가)
        self.stop_height = 430      # [수정] 350->430 (화면에 꽉 찰 때까지 접근 = 약 2m)
        
        self.master_id = None
        self.wave_time_threshold = 3.0
        self.wave_histories = {}
        self.master_last_seen = 0
        
        # === [통신] ===
        self.sub_img = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 1)
        self.pub_cmd = self.create_publisher(Vector3, '/dobby/target_info', 10)
        
        self.bridge = CvBridge()
        self.yolo = YOLO('yolov8m.pt') # m 모델 유지 (눈 좋음)
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, 
            min_detection_confidence=0.5,
            model_complexity=1
        )
        
        self.get_logger().info("도비 비전(V3): 근접/고속 모드 가동!")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            current_time = time.time()
            
            cmd_msg = Vector3()
            
            results = self.yolo.track(frame, persist=True, verbose=False, classes=0)
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # [상태 1: 추적]
                if self.master_id is not None:
                    if self.master_id in track_ids:
                        idx = track_ids.index(self.master_id)
                        box = boxes[idx]
                        self.master_last_seen = current_time
                        
                        vel, yaw = self.calculate_command(box)
                        cmd_msg.x = float(vel)
                        cmd_msg.y = float(yaw)
                        cmd_msg.z = 1.0 
                        
                        # 상태 표시 (거리 감 잡기용 박스 높이 출력)
                        h = box[3] - box[1]
                        self.draw_box(frame, box, f"GO! H:{int(h)}/{self.stop_height}", (0, 255, 0))
                    else:
                        if current_time - self.master_last_seen > 5.0:
                            self.master_id = None
                
                # [상태 2: 탐색]
                else:
                    for box, track_id in zip(boxes, track_ids):
                        if self.check_waving(frame, box, track_id, current_time):
                            self.master_id = track_id
                            self.master_last_seen = current_time
                            self.get_logger().info(f"주인님 발견! 돌격 앞으로!")
                            break

            self.pub_cmd.publish(cmd_msg)
            cv2.imshow("Dobby Vision Eye", frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"비전 에러: {e}")

    def calculate_command(self, box):
        x1, y1, x2, y2 = map(int, box)
        center_x = (x1 + x2) / 2
        height = y2 - y1
        
        # 1. 회전 (중앙 맞추기)
        error_x = self.img_center_x - center_x
        cmd_yaw = -1 * self.k_yaw * error_x # 오차가 클수록 빨리 돔
        
        # 2. 속도 (가까이 붙기)
        # 중앙 오차(tolerance) 안쪽으로 들어왔을 때만 직진!
        if abs(error_x) > self.center_tolerance:
            cmd_vel = 0.0
        else:
            # 목표 높이(430)보다 작으면(멀면) 접근
            if height < self.stop_height:
                # 거리에 비례해서 속도 조절 (멀면 빠르고, 가까우면 천천히)
                # 최소 속도 0.2는 보장해서 답답함 제거
                cmd_vel = self.k_speed * (1.0 - (height / 500.0)) 
                cmd_vel = max(0.3, min(cmd_vel, 1.2)) 
            else:
                cmd_vel = 0.0
                
        return cmd_vel, cmd_yaw

    def check_waving(self, frame, box, track_id, current_time):
        # 1. ROI 추출
        h, w, _ = frame.shape
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        person_roi = frame[y1:y2, x1:x2]
        if person_roi.size == 0: return False

        # 2. Pose 인식
        roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(roi_rgb)

        is_waving = False
        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks.landmark
            # 손목이 어깨보다 높으면 OK
            l_hand_up = (lm[15].y < lm[11].y)
            r_hand_up = (lm[16].y < lm[12].y)
            
            if l_hand_up or r_hand_up:
                is_waving = True

        # 3. 시간 체크
        if track_id not in self.wave_histories:
            self.wave_histories[track_id] = {'start_time': None}
        record = self.wave_histories[track_id]

        if is_waving:
            if record['start_time'] is None: record['start_time'] = current_time
            duration = current_time - record['start_time']
            self.draw_box(frame, box, f"Check.. {duration:.1f}s", (0, 255, 255))
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