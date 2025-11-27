import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')
        
        # 1. 영상 구독 (Subscribe)
        # 큐 사이즈를 1로 해서 가장 최신 영상만 가져옵니다 (렉 방지)
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            1)
        
        self.bridge = CvBridge()
        
        # 2. YOLO v8 모델 로드 (처음 실행 시 자동으로 다운로드됨)
        # 'yolov8n.pt'는 가장 빠르고 가벼운 모델입니다.
        self.model = YOLO('yolov8n.pt')
        
        self.get_logger().info("도비의 AI 눈(YOLO)이 켜졌습니다!")

    def image_callback(self, msg):
        try:
            # 3. ROS 이미지를 OpenCV 이미지로 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 4. YOLO 추론 실행
            # classes=0 은 '사람(person)'만 찾으라는 뜻입니다.
            results = self.model(cv_image, verbose=False, classes=0)
            
            # 5. 결과 그리기
            annotated_frame = results[0].plot()
            
            # 사람이 감지되었는지 확인
            if len(results[0].boxes) > 0:
                self.get_logger().info(f"사람 발견! {len(results[0].boxes)}명")

            # 6. 화면에 띄우기
            cv2.imshow("Dobby's Eye (YOLO v8)", annotated_frame)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'에러 발생: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
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
