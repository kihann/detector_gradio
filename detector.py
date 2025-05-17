from ultralytics import YOLO

import os
import time
import cv2
import numpy as np
import gradio as gr

MODEL_PATH = "yolov10m.pt"  # 사용할 모델 경로 또는 이름. n은 "nano"를 의미함.
CONFIDENCE_THRESHOLD = .3 # 객체 탐지 최소 신뢰도. 모델이 가벼울수록 신뢰도가 낮아질 수 있으므로 조절 필요
PREDICT_IMG_SIZE = 360 # 성능 최적화를 위한 추론 이미지 사이즈 줄이기
model = None  # 전역 모델 변수 초기화
prev_time = 0 # FPS 계산을 위한 이전 프레임 처리 시간

# 1. 모델 로드 함수
def load_yolo_model():
    """
    지정된 경로의 YOLO 모델을 로드하여 전역 변수 'model'에 할당합니다.
    모델 로드 성공 또는 실패 메시지를 콘솔에 출력합니다.
    """
    global model # 전역변수 model = None(초기화)

    try:
        model = YOLO(MODEL_PATH)
        print(f"Succeed: YOLO 모델 로드 완료 ('{MODEL_PATH}')")
        print(f"탐지 클래스: {model.names}")
    except Exception as e:
        print(f"Error: YOLO 모델 로드 실패 ('{MODEL_PATH}'): {e}")
        model = None # model = None 유지

# 2. 웹캠 객체 탐지 함수
def webcam_detector(frame: np.ndarray):
    """
    웹캠으로부터 입력받은 단일 프레임에 대해 객체 탐지를 수행합니다.
    탐지 결과(바운딩 박스, 클래스, 신뢰도)가 그려진 프레임과 탐지 정보를 반환합니다.
    실시간 FPS 정보도 프레임에 함께 표시합니다.

    Args:
        frame (np.ndarray): 웹캠에서 입력된 이미지 프레임

    Returns:
        tuple:
            - np.ndarray: 객체 탐지 결과 및 FPS가 표시된 이미지 프레임
            - str: 탐지된 객체들의 정보 (클래스, 신뢰도) 또는 오류 메시지
    """
    global model, prev_time # 전역변수 사용

    # 웹캠 프레임 유효성 검사
    if frame is None:
        print("Error: 웹캠으로부터 프레임을 받지 못했습니다.")
        error_img = np.zeros((480, 640, 3), dtype=np.uint8) # 검은색 배경의 오류 이미지
        cv2.putText(error_img, "Webcam feed NOT available", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(error_img, "Check webcam & browser permission.", (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return error_img, "웹캠 프레임 없음."
    
    # 모델 로드 상태 확인
    if model is None:
        # 모델이 로드되지 않았다면, 원본 프레임에 오류 메시지를 표시합니다.
        cv2.putText(frame, "Error: Model not loaded!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Check console for details. Path: {MODEL_PATH}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame, "YOLO 모델이 로드되지 않았습니다."
    
    # 추론(객체 탐지) 수행 시작 시간 기록
    process_start_time = time.time()

    # 추론(객체 탐지) 수행
    results = model.predict(
        source = frame, 
        conf = CONFIDENCE_THRESHOLD, 
        imgsz = PREDICT_IMG_SIZE,
        verbose=False
    ) # verbose=False로 Ultralytics의 상세 로그를 미출력

    # 추론 결과 시각화
    annotated_frame = results[0].plot() # 바운딩 박스와 레이블이 그려진 이미지 반환

    # 탐지된 객체 정보 문자열 생성
    detected_info_list = []
    # results는 단일 이미지에 대한 결과이므로 results[0]을 사용 (기존 코드에서는 r for results 불필요)
    for box in results[0].boxes: # results[0]에서 boxes를 바로 가져옴
        try:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            detected_info_list.append(f"종류: {class_name}, 신뢰도: {confidence:.2f}")
        except Exception as e:
            print(f"Error processing detection result: {e}")
            continue
    detected_info_str = "\n".join(detected_info_list) if detected_info_list else "탐지된 객체 없음"

    # 단일 프레임 처리 시간 및 FPS 계산
    process_end_time = time.time()
    processing_time = process_end_time - process_start_time
    fps = 1 / processing_time if processing_time > 0 else 0

    # FPS 정보를 초록색으로 annotated_frame 좌상단에 표시
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Model: {os.path.basename(MODEL_PATH)} @{PREDICT_IMG_SIZE}px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)

    return annotated_frame, detected_info_str

# 3. Gradio 인터페이스 설정
def create_gradio_interface():
    """
    Gradio 웹 인터페이스를 생성하고 반환합니다.
    """
    input_webcam = gr.Image(
        sources=["webcam"], # 웹캠 입력 받음
        streaming=True, # 실시간 스트리밍을 활성화
        type="numpy", # OpenCV 처리용으로 numpy 배열 형태로 반환
        label="Webcam Input"
    )

    # 출력 컴포넌트 1: 객체 탐지 결과 이미지
    # type="numpy"로 설정하여 OpenCV로 처리된 NumPy 배열 이미지를 출력합니다.
    output_annotated_image = gr.Image(
        type="numpy",
        label="Detection Result"
    )

    # 출력 컴포넌트 2: 탐지된 객체 정보 텍스트
    output_detected_info = gr.Textbox(
        label="Detected Objects Info",
        lines=10, # 텍스트 박스의 기본 줄 수
        show_copy_button=True # 복사 버튼 표시
    )
    
    # 정보 표시용 Markdown
    model_info_md = gr.Markdown(
        f"""
        ## YOLOv10 실시간 웹캠 객체 탐지 데모
        - 사용 모델: {os.path.basename(MODEL_PATH)}
        - 추론 이미지 크기: {PREDICT_IMG_SIZE} px
        - 최소 신뢰도: {CONFIDENCE_THRESHOLD}
        - 오픈소스 사용: Ultralytics YOLO (출처: [https://www.google.com/url?sa=E&source=gmail&q=https://ultralytics.com/yolo](https://www.google.com/url?sa=E&source=gmail&q=https://ultralytics.com/yolo))
        
        **실행 방법: 웹캠 접근을 허용한 후, 잠시 기다리면 실시간으로 객체 탐지가 시작됩니다.**
        """
    )
    
    # gr.Blocks()를 사용하여 좀 더 유연한 레이아웃 구성
    with gr.Blocks() as blocks:
        model_info_md.render() # 상단에 정보 표시
        with gr.Row():
            input_webcam.render()
            with gr.Column(scale=1):
                output_annotated_image.render()
                output_detected_info.render()
        
        # 실제 이벤트 처리기 연결 (=웹캠 스트림이 업데이트 될 때마다 webcam_detector 호출)
        input_webcam.stream(
            fn=webcam_detector,
            inputs=[input_webcam],
            outputs=[output_annotated_image, output_detected_info]
        )
        
    return blocks

##########################################################################################

if __name__ == "__main__":
    # 1. YOLO 모델 로드 시도
    load_yolo_model()

    # 2. 모델 로드 성공 여부 확인 후 Gradio 인터페이스 실행
    if model is not None:
        print("YOLO 모델이 성공적으로 로드되었습니다. Gradio 인터페이스를 시작합니다.")
        gr_ = create_gradio_interface()
        gr_.launch(debug=True, share=True)
    else:
        print("Error: YOLO 모델 로드에 실패, Gradio 인터페이스를 시작할 수 없습니다.")
        print(f"모델 경로를 확인하거나 ('{MODEL_PATH}'), Ultralytics 설치 상태를 점검하세요.")