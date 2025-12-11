import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import mediapipe as mp
import time
import numpy as np

class DobbyVision(Node):
    def __init__(self):
        super().__init__('dobby_vision_node')
        
        # === [설정] ===
        self.img_center_x = 320     
        self.center_tolerance = 60  
        self.k_yaw = 0.02           
        self.k_speed = 1.0          
        self.stop_height = 430      
        
        self.master_id = None
        self.wave_time_threshold = 3.0
        self.wave_histories = {}
        self.master_last_seen = 0
        
        # 1. 카메라 영상 받기 (Raw Image)
        self.sub_img = self.create_subscription(
            Image, 
            '/camera/image_raw', 
            self.image_callback, 
            10)
            
        # 2. 제어 명령 보내기
        self.pub_cmd = self.create_publisher(Vector3, '/dobby/target_info', 10)
        
        # [NEW] 3. 결과 화면(박스 그려진 것) 보내기 -> 노트북에서 확인용!
        self.pub_debug = self.create_publisher(Image, '/dobby/debug_image', 10)
        
        self.bridge = CvBridge()
        
        # 라즈베리파이 부하 고려: Nano 모델 사용
        self.yolo = YOLO('yolov8n.pt') 
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, 
            min_detection_confidence=0.5,
            model_complexity=0
        )
        
        self.get_logger().info("도비 비전: 결과 화면 송출 중 (/dobby/debug_image)")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            current_time = time.time()
            cmd_msg = Vector3()
            
            # YOLO 트래킹
            results = self.yolo.track(frame, persist=True, verbose=False, classes=0)
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                if self.master_id is not None:
                    if self.master_id in track_ids:
                        idx = track_ids.index(self.master_id)
                        box = boxes[idx]
                        self.master_last_seen = current_time
                        
                        vel, yaw = self.calculate_command(box)
                        cmd_msg.x = float(vel)
                        cmd_msg.y = float(yaw)
                        cmd_msg.z = 1.0 
                        
                        # 화면에 정보 표시 (박스 그리기)
                        h = box[3] - box[1]
                        self.draw_box(frame, box, f"MASTER H:{int(h)}", (0, 255, 0))
                    else:
                        if current_time - self.master_last_seen > 5.0:
                            self.master_id = None
                else:
                    for box, track_id in zip(boxes, track_ids):
                        if self.check_waving(frame, box, track_id, current_time):
                            self.master_id = track_id
                            self.master_last_seen = current_time
                            self.get_logger().info(f"주인님 등록 완료!")
                            break

            self.pub_cmd.publish(cmd_msg)
            
            # [NEW] 박스가 그려진 최종 화면(frame)을 노트북으로 전송!
            debug_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.pub_debug.publish(debug_msg)

        except Exception as e:
            self.get_logger().error(f"비전 에러: {e}")

    def calculate_command(self, box):
        x1, y1, x2, y2 = map(int, box)
        center_x = (x1 + x2) / 2
        height = y2 - y1
        error_x = self.img_center_x - center_x
        cmd_yaw = -1 * self.k_yaw * error_x 
        
        if abs(error_x) > self.center_tolerance:
            cmd_vel = 0.0
        else:
            if height < self.stop_height:
                cmd_vel = self.k_speed * (1.0 - (height / 500.0))
                cmd_vel = max(0.3, min(cmd_vel, 1.2)) 
            else:
                cmd_vel = 0.0
        return cmd_vel, cmd_yaw

    def check_waving(self, frame, box, track_id, current_time):
        h, w, _ = frame.shape
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        person_roi = frame[y1:y2, x1:x2]
        if person_roi.size == 0: return False

        roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(roi_rgb)
        is_waving = False
        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks.landmark
            l_hand_up = (lm[15].y < lm[11].y)
            r_hand_up = (lm[16].y < lm[12].y)
            if l_hand_up or r_hand_up:
                is_waving = True

        if track_id not in self.wave_histories:
            self.wave_histories[track_id] = {'start_time': None}
        record = self.wave_histories[track_id]

        if is_waving:
            if record['start_time'] is None: record['start_time'] = current_time
            duration = current_time - record['start_time']
            # 인식 중일 때 박스 표시
            self.draw_box(frame, box, f"Check.. {duration:.1f}s", (0, 255, 255))
            if duration >= self.wave_time_threshold: return True
        else:
            record['start_time'] = None
            # 평소 상태 박스 표시
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

if __name__ == '__main__':
    main()