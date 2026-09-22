# Auto Label Generator

YOLO 객체 탐지 결과를 이용해 도로 영상에서 차량 및 운전자 데이터를 자동으로 저장하고, YOLO 형식의 라벨 파일을 생성하는 프로젝트입니다.

현재 구현은 다음 객체를 탐지합니다.

- `person`
- `bicycle`
- `car`
- `motorcycle`
- `bus`
- `truck`

`person`은 차량 또는 이륜차 영역과 IoU가 `0.3` 이상일 때만 저장되므로, 차량 주변의 운전자 후보를 추출하는 용도로 사용할 수 있습니다.

## 동작 개요

1. 입력 영상 또는 RTSP 스트림을 읽습니다.
2. Ultralytics YOLO 모델로 객체를 탐지합니다.
3. 탐지 결과를 정규화된 YOLO 라벨 형식으로 변환합니다.
4. 객체가 탐지된 프레임만 이미지, 라벨, 예측 시각화 이미지로 저장합니다.
5. 탐지 객체에 따라 `bus`, `truck`, `bi`, `else` 폴더로 분류합니다.

## 요구 사항

- Python 3.9 이상 권장
- YOLO 추론이 가능한 CPU 또는 CUDA 환경
- 입력 동영상 파일 또는 RTSP 카메라 주소

필수 패키지는 다음과 같습니다.

```bash
pip install ultralytics opencv-python pillow
```

Ultralytics가 사용하는 PyTorch는 실행 환경에 맞는 버전으로 설치하는 것을 권장합니다. CUDA를 사용할 경우 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)에 따라 먼저 설치하세요.

## 실행 방법

### 동영상 파일 모드

기본 모드는 `VIDEO_MODE`입니다. `main.py`의 `run()` 함수에서 입력 파일 경로를 지정합니다.

```python
mode = VIDEO_MODE
path = "path/to/input.mp4"
```

그 후 프로젝트 루트에서 실행합니다.

```bash
python main.py
```

### RTSP 모드

`config.py`에서 RTSP 주소를 설정합니다.

```python
RTSP_ADDRESS = "rtsp://username:password@host:port/stream"
```

그리고 `main.py`의 `run()` 함수에서 모드를 변경합니다.

```python
mode = RTSP_MODE
```

RTSP 모드에서는 프레임 수집, YOLO 추론, FPS 측정이 별도 스레드로 실행됩니다. 실행 중 콘솔에서 빈 줄을 입력하면 프로그램을 종료하고 FPS 정보가 저장됩니다.

## 설정

주요 설정은 [`config.py`](config.py)에 있습니다.

| 설정 | 설명 | 기본값 |
| --- | --- | --- |
| `YOLO_MODEL` | 사용할 Ultralytics YOLO 모델 파일 | `yolo11x.pt` |
| `RTSP_ADDRESS` | RTSP 입력 주소 | 빈 문자열 |
| `RTSP_MODE` | RTSP 모드 식별값 | `0` |
| `VIDEO_MODE` | 동영상 파일 모드 식별값 | `1` |

`yolo11x.pt`가 로컬에 없으면 Ultralytics가 모델을 다운로드할 수 있습니다. 네트워크가 제한된 환경에서는 모델 파일을 미리 준비하고 `YOLO_MODEL`에 경로를 지정하세요.

## 결과 폴더 구조

실행 후 탐지 결과는 프로젝트 루트의 `saved` 폴더에 날짜별로 저장됩니다.

```text
saved/
└── YYYYMMDD/
    ├── bus/
    │   ├── images/   # 원본 프레임
    │   ├── labels/   # YOLO 형식 라벨
    │   └── predict/  # 탐지 결과 시각화 이미지
    ├── truck/
    ├── bi/           # bicycle 또는 motorcycle 포함 결과
    └── else/         # 그 외 결과
```

RTSP 모드로 종료하면 날짜 폴더에 다음 파일도 생성됩니다.

```text
saved/YYYYMMDD/fps.txt
```

## 라벨 형식

각 라벨 파일의 한 줄은 다음 형식입니다.

```text
class_id x_center y_center width height
```

좌표와 크기는 이미지 전체를 기준으로 `0.0~1.0` 범위로 정규화된 값이며, YOLO 모델의 클래스 ID를 사용합니다. 신뢰도(confidence)는 현재 라벨 파일에 저장하지 않습니다.

## 주요 파일

- [`main.py`](main.py): 영상/RTSP 입력, YOLO 추론, 스레드 처리, 결과 저장
- [`config.py`](config.py): 모델·입력 설정, 날짜 처리, IoU 및 운전자 필터링

## 현재 구현상의 주의점

- 동영상 파일 경로는 `main.py`의 `run()` 함수 안에서 직접 지정해야 합니다.
- VIDEO/RTSP 모드 전환도 현재는 `run()` 함수의 `mode` 값을 수정하는 방식입니다.
- RTSP 입력이 열리지 않거나 프레임을 읽지 못하는 경우를 별도로 처리하지 않으므로, 주소와 네트워크 연결을 먼저 확인해야 합니다.
- `saved` 폴더는 자동으로 생성되며, 같은 날짜에 동일한 타임스탬프 이름이 생성되면 파일이 덮어써질 수 있습니다.
- 대형 모델인 `yolo11x.pt`는 높은 정확도를 제공하지만 GPU 메모리와 추론 시간이 많이 필요할 수 있습니다.

## 라이선스

이 저장소에는 현재 별도의 라이선스 파일이 포함되어 있지 않습니다. 배포 또는 상업적 사용 전 프로젝트 코드와 사용 모델의 라이선스를 확인하세요.
