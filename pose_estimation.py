import os
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from enum import IntEnum
from dataclasses import dataclass

model_path = './pose_landmarker_full.task' # Chương trình này sử dụng MediaPipe Model Pose Landmarker Full để quan sát các khớp

# Cài đặt ban đầu cho mô hình
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Tuỳ chọn cho model
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO, # Chế độ chạy - được đặt thành video thời gian thực
    num_poses = 1, # Số dáng nhận diện, ở đây chỉ có thể nhận 1 người
    #min_pose_detection_confidence = 0.5, # Độ chính xác tối thiểu để có thể xác nhận dáng
    #min_pose_presence_confidence = 0.5, # Độ chính xác tối thiểu để có thể xác nhận sự hiện diện
    #min_tracking_confidence = 0.5, # Độ chính xác tối thiểu để có thể xác nhận việc nhận dạng là chính xá
)

# Config bài tập - khởi tạo
@dataclass
class ExerciseConfig:
    name: str                               # Tên bài tập
    knee_angle_up: float = None             # Góc gối để tính trạng thái UP    
    knee_angle_down_entry: float = None     # Ngưỡng gối để tính DOWN
    hip_angle_up: float = None              # Góc hông để tính trạng thái UP  
    correct_knee_max: float = None          # Ngưỡng gối để khi squat đủ sẽ tính là "Đúng"
    correct_hip_max: float = None           # Ngưỡng hông để khi squat đủ sẽ tính là "Đúng"
    correct_elbow_max: float = None         # Ngưỡng khuỷu tay để khi squhít đất đủ sẽ tính là "Đúng"
    elbow_angle_up: float = None
    elbow_angle_down_entry: float = None
    plank_hip_min: float = None
    hold_frames_required: int = None
    body_straight_min: float = None

# Góc yêu cầu của mỗi bài tập
EXERCISE_CONFIGS = {
    1: ExerciseConfig(name="Squat", knee_angle_up=160, knee_angle_down_entry=100,       # Squat
                   hip_angle_up=160, correct_knee_max=95, correct_hip_max=95),    
    2: ExerciseConfig(name="Plank", plank_hip_min=160, hold_frames_required=60),        #Plank
    3: ExerciseConfig(name="PushUp", elbow_angle_up=160, elbow_angle_down_entry=90,
                       correct_elbow_max=90, body_straight_min=160)                                            #PushUp
}

# Hàm tính góc
def calculate_angle(a, b, c):
    """Tinh goc (do) tao boi 3 diem a-b-c, voi b la dinh goc."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

# mp.pose.POSE_CONNECTION đã loại bỏ, phải tự tạo pose landmark mới
class PoseLandmark(IntEnum):
    NOSE = 0                        # Mũi
    LEFT_EYE_INNER = 1              
    LEFT_EYE = 2                    # Mắt trái
    LEFT_EYE_OUTER = 3              
    RIGHT_EYE_INNER = 4     
    RIGHT_EYE = 5                   # Mắt phải
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7                    # Tai trái
    RIGHT_EAR = 8                   # Tai phải
    MOUTH_LEFT = 9                  # Miệng trái
    MOUTH_RIGHT = 10                # Miệng phải
    LEFT_SHOULDER = 11              # Vai trái
    RIGHT_SHOULDER = 12             # Vai phải
    LEFT_ELBOW = 13                 # Khuỷu tay trái
    RIGHT_ELBOW = 14                # Khuỷu tay phải
    LEFT_WRIST = 15                 # Cổ tay trái
    RIGHT_WRIST = 16                # Cổ tay phải
    LEFT_PINKY = 17                 # Ngón út trái
    RIGHT_PINKY = 18                # Ngón út phải
    LEFT_INDEX = 19                 # Ngón trỏ trái
    RIGHT_INDEX = 20                # Ngón trỏ phải
    LEFT_THUMB = 21                 # Ngón cái trái
    RIGHT_THUMB = 22                # Ngón cái phải
    LEFT_HIP = 23                   # Hông trái
    RIGHT_HIP = 24                  # Hông phải
    LEFT_KNEE = 25                  # Đầu gối trái
    RIGHT_KNEE = 26                 # Đầu gối phải
    LEFT_ANKLE = 27                 # Cổ chân trái 
    RIGHT_ANKLE = 28                # Cổ chân phải
    LEFT_HEEL = 29                  # Gót chân trái 
    RIGHT_HEEL = 30                 # Gót chân phải
    LEFT_FOOT_INDEX = 31            # Chân trỏ trái
    RIGHT_FOOT_INDEX = 32           # Chân trỏ phải

# Phân bộ phận cơ thể
POSE_CONNECTIONS = [
    # Mặt
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), 

    # Thân 
    (11, 12), (11, 23), (12, 24), (23, 24),

    # Tay trái
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),

    # Tay phải
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),

    # Chân trái
    (23, 25), (25, 27), (27, 29), (29, 31),

    # Chân phải
    (24, 26), (26, 28), (28, 30), (30, 32),
]

# Vẽ các điểm khớp sử dụng hàm ngoài
def draw_landmarks_manual(frame, landmarks):
    h, w, _ = frame.shape
    for start_idx, end_idx in POSE_CONNECTIONS:
        x1, y1 = int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h)
        x2, y2 = int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 0, 255), -1)

def main(video_source=0):
    cap = cv2.VideoCapture(video_source)
    stage = None # "up" hoac "down"

    # Biến trạng thái, thay đổi liên tục khi chương trình chạy
    excercise_mode = 1                      # Chế độ bài tập, mặc định là Squat
    stage = None                            # Trạng thái lên xuống        
    min_knee_angle = 180                    # Góc nhỏ nhất của đầu gối - dùng để so sánh lúc ngồi xuống
    min_hip_angle = 180                     # Góc nhỏ nhất của hông
    min_elbow_angle = 180                   # Góc nhỏ nhất của khuỷu tay 
    down_frame_counter = 0                  # Đếm số frame lúc bắt đầu plank. Nếu đủ frame sẽ bắt đầu đếm thời gian, reset nếu tư thế hỏng
    timer_running = False                   # Tình trạng bộ đếm giờ plank
    plank_start_time = None                 # Thời gian bắt đầu tính giờ Plank
    results = {                             # Kết quả cho từng bài tập
        1: {"correct": 0, "wrong": 0},
        2: {"time_held": 0.0},
        3: {"correct": 0, "wrong": 0},
    }
    cfg = EXERCISE_CONFIGS[excercise_mode]  # Chọn config tuỳ theo chế độ

    with PoseLandmarker.create_from_options(options) as landmarker:

        # Khi video đang ghi hình
        while cap.isOpened():
            ret, frame = cap.read() # Đọc frame

            if not ret:
                break

            # Chuyển đổi video đầu vào thành định dạng phù hợp    
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)

            # Hàm ghi kết quả
            pose_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if pose_result.pose_landmarks:
                landmarks = pose_result.pose_landmarks[0]

                # Toạ độ bên trái
                hip = [landmarks[PoseLandmark.LEFT_HIP].x, landmarks[PoseLandmark.LEFT_HIP].y]                      # Hông
                knee = [landmarks[PoseLandmark.LEFT_KNEE].x, landmarks[PoseLandmark.LEFT_KNEE].y]                   # Đầu gối
                ankle = [landmarks[PoseLandmark.LEFT_ANKLE].x, landmarks[PoseLandmark.LEFT_ANKLE].y]                # Mắt cá chân
                shouder = [landmarks[PoseLandmark.LEFT_SHOULDER].x, landmarks[PoseLandmark.LEFT_SHOULDER].y]        # Vai
                elbow = [landmarks[PoseLandmark.LEFT_ELBOW].x, landmarks[PoseLandmark.LEFT_ELBOW].y]                # Khuỷu tay
                wrist = [landmarks[PoseLandmark.LEFT_WRIST].x, landmarks[PoseLandmark.LEFT_WRIST].y]                # Cổ tay
                foot = [landmarks[PoseLandmark.LEFT_FOOT_INDEX].x, landmarks[PoseLandmark.LEFT_FOOT_INDEX].y]       # Bàn Chân

                hip_angle = calculate_angle(knee, hip, shouder) # Góc hông
                knee_angle = calculate_angle(hip, knee, ankle) # Góc đầu gối
                elbow_angle = calculate_angle(wrist, elbow, shouder) # Góc khuỷu tay
                body_line_angle = calculate_angle(shouder, hip, ankle)  # Đo nguyên người - dành cho hít đất
                ankle_angle = calculate_angle(knee, ankle, foot) # Góc cổ chân

                # Ô vuông góc trên cùng trái
                cv2.rectangle(frame, (0, 0), (280, 100), (0, 0, 0), -1)

                # Hiển thị tên bài tập và thống kê tương ứng
                cv2.putText(frame, f"Bai tap: {cfg.name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # Logic đánh giá tuỳ theo bài tạp
                if (excercise_mode == 1):
                    cv2.putText(frame, f"So lan dung: {results[1]['correct']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"So lan sai: {results[1]['wrong']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    if knee_angle > cfg.knee_angle_up and hip_angle > cfg.hip_angle_up:                         # Nếu như góc đầu gối và hông hiện tại lớn hơn số góc định sẵn ở config
                        if stage == "down":                                                                     # Nếu trạng thái là đang xuống
                            if min_knee_angle < cfg.correct_knee_max and min_hip_angle < cfg.correct_hip_max:   # Nếu góc đầu gối và hông hiện tại nhỏ hơn yêu cầu
                                results[1]["correct"] += 1                                                      # Tính là đúng
                                print(f"Squat dung! So lan dung: {results[1]['correct']}")
                            else:
                                results[1]["wrong"] += 1                                                        
                                print(f"Squat sai! So lan sai: {results[1]['wrong']}")
                        # Reset
                        stage = "up"
                        min_knee_angle = 180
                        min_hip_angle = 180

                    if knee_angle < cfg.knee_angle_down_entry and stage == "up":
                        stage = "down"

                    # Cách tính đúng/sai của bài tập này là tính xuyên suốt rep, không phải tính lúc ngồi xuống
                    if stage == "down":
                        min_knee_angle = min(min_knee_angle, knee_angle)
                        min_hip_angle = min(min_hip_angle, hip_angle)
                        if knee_angle > cfg.correct_knee_max and hip_angle > cfg.correct_hip_max:
                            cv2.putText(frame, "Tu the sai!", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        else:
                            cv2.putText(frame, "Tu the dung!", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if (excercise_mode == 2):
                    current_hold = results[2]["time_held"] + (time.time() - plank_start_time) if timer_running else results[2]["time_held"]       # Tinh thoi gian de hien thi: neu dang giu tu the thi cong them thoi gian dang chay vao tong da tich luy, neu khong thi chi hien tong da tich luy
                    cv2.putText(frame, f"Thoi gian giu: {current_hold:.1f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
                    if (excercise_mode == 2):
                        if hip_angle > cfg.plank_hip_min:
                            down_frame_counter += 1
                            # Chỉ bắt đầu đếm giờ sau khi giữ tư thế đủ lâu (60 frame / 30fps = 2s)
                            if down_frame_counter >= cfg.hold_frames_required and not timer_running:
                                timer_running = True
                                plank_start_time = time.time()
                        else: # Nếu như tư thế bị hỏng và đang bấm giờ thì cộng thời gian vừa giữ vào tổng rồi dừng bấm giờ
                            if timer_running:
                                results[2]["time_held"] += time.time() - plank_start_time
                                timer_running = False
                            down_frame_counter = 0      # Reset lại bộ đếm giữ tư thế

                    # Hiển thị thời gian giữ tư thế
                    if timer_running:
                        current_hold = results[2]["time_held"] + (time.time() - plank_start_time)
                    else:
                        current_hold = results[2]["time_held"]

                # Cách theo dõi của hít đất gần giống với squat nhưng tính góc khác
                if (excercise_mode == 3):
                    cv2.putText(frame, f"So lan dung: {results[3]['correct']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"So lan sai: {results[3]['wrong']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    is_body_straight = body_line_angle > cfg.body_straight_min # Người có đang thẳng không?

                    if is_body_straight:
                        if elbow_angle > cfg.elbow_angle_up:
                            if stage == "down":
                                if min_elbow_angle < cfg.correct_elbow_max:
                                    results[3]["correct"] += 1
                                    print(f"Hit dat dung! So lan dung: {results[3]['correct']}")
                                else:
                                    results[3]["wrong"] += 1
                                    print(f"Hit dat sai! So lan sai: {results[3]['wrong']}")
                            stage = "up"
                            min_elbow_angle = 180

                        if elbow_angle < cfg.elbow_angle_down_entry and stage == "up":
                            stage = "down"

                        if stage == "down":
                            min_elbow_angle = min(min_elbow_angle, elbow_angle)
                            if elbow_angle > cfg.correct_elbow_max:
                                cv2.putText(frame, "Tu the sai!", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            else:
                                cv2.putText(frame, "Tu the dung!", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Vui long vao tu the hit dat!", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)   # Không thẳng = không tính logic đúng sai          
                        
                draw_landmarks_manual(frame, landmarks)

                # Thông tin bài tập hiện tại + các góc cơ thể liên quan
                if (excercise_mode == 1):
                    cv2.putText(frame, f"Goc goi: {int(knee_angle)}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(frame, f"Goc hong: {int(hip_angle)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                elif (excercise_mode == 3):
                    cv2.putText(frame, f"Goc hong: {int(hip_angle)}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(frame, f"Goc khuyu tay: {int(elbow_angle)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(frame, f"Goc mat ca chan: {int(ankle_angle)}", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                else:
                    cv2.putText(frame, f"Goc hong: {int(hip_angle)}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(frame, f"Goc khuyu tay: {int(elbow_angle)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Chọn chế độ
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):         # Q = thoát
                break
            
            new_mode = None
            if key == ord("1"):         # Chế độ 1 - Squat
                new_mode = 1
            elif key == ord("2"):       # Chế độ 2 - Plank
                new_mode = 2
            elif key == ord("3"):       # Chế độ 3 - Hít đất
                new_mode = 3

            if new_mode is not None and new_mode != excercise_mode:             # Check xem đảm bảo có đang chạy một trong các chế độ và chế độ đó có phải là có trong chương trình không
                if excercise_mode == 2 and timer_running:                       # Nếu chế độ 2 và bộ đếm đang chạy
                    results[2]["time_held"] += time.time() - plank_start_time   # Ghi kết quả = thời gian giữ - thời gian sẵn sàng
                    timer_running = False                                       # Xong hủ bộ đếm, không cho chạy lúc đang ở bài tập khác
                excercise_mode = new_mode                                       # Đổi sang bài tập theo nút nhấn
                cfg = EXERCISE_CONFIGS[excercise_mode]                          # Lựa chọn config bài tập

                # Reset lại các giá trị - không reset thì sẽ tính sai, đặc biệt là khi đổi trong khi đang đếm
                stage = None
                min_knee_angle = 180
                min_hip_angle = 180
                min_elbow_angle = 180 
                down_frame_counter = 0
                print(f"Chuyen sang bai tap: {cfg.name}")
                
            cv2.imshow("Phan tich dong tac the duc - Nhan Q de thoat", frame)
                

    cap.release()

    if excercise_mode == 2 and timer_running:
        results[2]["time_held"] += time.time() - plank_start_time

    # Đổi đường dẫn và tên folder đầu ra ở đâu
    output_dir = "workout_results"
    os.makedirs(output_dir, exist_ok=True)  # Nếu chưa có thì sẽ tự tạo thêm folder mới

    filename = os.path.join(output_dir, f"workout_results_{time.strftime('%Y-%m-%d_%H-%M-%S')}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=== KET QUA BUOI TAP ===\n\n")
        f.write(f"Squat:\n  So lan dung: {results[1]['correct']}\n  So lan sai: {results[1]['wrong']}\n\n")
        f.write(f"Plank:\n  Thoi gian giu tu the: {results[2]['time_held']:.1f} giay\n\n")
        f.write(f"Push-up:\n  So lan dung: {results[3]['correct']}\n  So lan sai: {results[3]['wrong']}\n")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(video_source=0)