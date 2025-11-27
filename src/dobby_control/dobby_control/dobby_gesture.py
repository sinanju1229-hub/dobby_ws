import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
import time
from collections import deque

class DobbyGesture(Node):
    def __init__(self):
        super().__init__('dobby_gesture_node')
        
        # === [설정] ===
        self.master_id = None       # 주인님의 ID (None이면 탐색 모드)
        self.wave_time_threshold = 3.0  # 손 흔들기 인식 필요 시간 (초)
        self.lost_patience = 5.0    # 주인을 놓쳐도 기다려주는 시간 (초)
        
        # === [데이터 관리] ===
        self.wave_histories = {}    # ID별 손 흔들기 시작 시간 및 상태 저장
        self.master_last_seen = 0   # 주인을 마지막으로 본 시간
        
        # === [ROS 통신] ===
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 1)
        self.bridge = CvBridge()
        
        # === [AI 모델] ===
        # 1. YOLO (사람 추적용)
        self.yolo = YOLO('yolov8n.pt')
        
        # 2. MediaPipe Pose (어깨-팔꿈치-손목 관절 인식용)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        self.get_logger().info("도비: 주인님을 찾고 있습니다! (손을 3초간 흔들어주세요)")

    def image_callback(self, msg):
        try:
            # 1. 이미지 변환
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            current_time = time.time()
            
            # 2. YOLO 트래킹 실행 (persist=True로 ID 유지)
            # 사람(class=0)만 추적
            results = self.yolo.track(frame, persist=True, verbose=False, classes=0)
            
            if results[0].boxes.id is None:
                self.display_feed(frame)
                return

            # 탐지된 객체들의 정보 추출
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # === [모드 1: 주인님이 정해졌을 때 (추적 모드)] ===
            if self.master_id is not None:
                if self.master_id in track_ids:
                    # 주인을 찾음!
                    idx = track_ids.index(self.master_id)
                    box = boxes[idx]
                    self.master_last_seen = current_time
                    
                    # 시각화 (초록색 박스)
                    self.draw_box(frame, box, "MASTER (Tracking)", (0, 255, 0))
                    self.get_logger().info(f"주인님 추적 중... ID: {self.master_id}")
                    
                    # TODO: 여기서 '종료 조건' 제스처(예: X자 표시)를 검사할 수도 있음
                    
                else:
                    # 주인이 화면에서 사라짐
                    time_lost = current_time - self.master_last_seen
                    if time_lost > self.lost_patience:
                        self.get_logger().warn(f"주인님을 잃어버렸습니다... 다시 찾습니다.")
                        self.master_id = None # 주인 해제 (재탐색)
                    else:
                        cv2.putText(frame, f"Searching Master... ({int(self.lost_patience - time_lost)}s)", 
                                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

            # === [모드 2: 주인님을 찾는 중 (탐색 모드)] ===
            else:
                for box, track_id in zip(boxes, track_ids):
                    # 각 사람에 대해 포즈 추정 (손 흔드는지 확인)
                    if self.check_waving(frame, box, track_id, current_time):
                        # 3초 이상 흔들었으면 주인으로 등록!
                        self.master_id = track_id
                        self.master_last_seen = current_time
                        self.get_logger().info(f"새로운 주인님 등록 완료! ID: {track_id}")
                        break # 즉시 루프 종료하고 섬기기 시작

            self.display_feed(frame)

        except Exception as e:
            self.get_logger().error(f"에러 발생: {e}")

    def check_waving(self, frame, box, track_id, current_time):
        """
        특정 사람(box)이 손을 흔들고 있는지 판별하는 함수
        어깨, 팔꿈치, 손목의 좌표를 사용
        """
        # 사람 영역 자르기 (ROI)
        h, w, _ = frame.shape
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        person_roi = frame[y1:y2, x1:x2]
        if person_roi.size == 0: return False

        # MediaPipe Pose 실행
        roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(roi_rgb)

        is_waving = False
        
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            
            # 주요 관절 인덱스 (MediaPipe Pose 기준)
            # 11: 왼쪽 어깨, 12: 오른쪽 어깨
            # 13: 왼쪽 팔꿈치, 14: 오른쪽 팔꿈치
            # 15: 왼쪽 손목, 16: 오른쪽 손목
            
            # 왼쪽 팔 검사
            l_sh = landmarks[11]
            l_el = landmarks[13]
            l_wr = landmarks[15]
            
            # 오른쪽 팔 검사
            r_sh = landmarks[12]
            r_el = landmarks[14]
            r_wr = landmarks[16]

            # 조건 1: 손목이 팔꿈치보다 높이 있는가? (y좌표는 위로 갈수록 작음)
            l_hand_up = l_wr.y < l_el.y and l_wr.y < l_sh.y
            r_hand_up = r_wr.y < r_el.y and r_wr.y < r_sh.y
            
            # 조건 2: 손이 움직이고 있는가? (단순화: 손을 들고 있으면 흔든다고 가정하거나, 좌표 변화량 체크)
            # 여기서는 '손을 어깨 높이 위로 들고 있는 상태'를 감지하면 타이머를 증가시킵니다.
            if l_hand_up or r_hand_up:
                is_waving = True

        # === 시간 체크 로직 ===
        if track_id not in self.wave_histories:
            self.wave_histories[track_id] = {'start_time': None, 'last_detected': 0}

        record = self.wave_histories[track_id]

        if is_waving:
            if record['start_time'] is None:
                record['start_time'] = current_time
            
            duration = current_time - record['start_time']
            
            # 시각화: 진행률 표시 바
            self.draw_box(frame, box, f"Waving: {duration:.1f}s", (0, 255, 255))
            
            if duration >= self.wave_time_threshold:
                return True # 3초 달성!
        else:
            # 손을 내리면 타이머 리셋
            record['start_time'] = None
            self.draw_box(frame, box, "Person", (0, 0, 255))

        return False

    def draw_box(self, frame, box, text, color):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def display_feed(self, frame):
        cv2.imshow("Dobby Smart Eye", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = DobbyGesture()
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
