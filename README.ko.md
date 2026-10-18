# avmosaic

이미지와 영상에서 노출 부위를 찾아 모자이크로 가리는 도구입니다. 명령줄(CLI)과 파이썬에서 쓸 수 있고, 윈도우·맥·리눅스에서 똑같이 동작합니다.

![원본 · 검출 박스(--draw-boxes) · 모자이크](docs/demo.jpg)

*데모는 공개 도메인 NASA 사진에 `faces` 프리셋을 적용한 것입니다. 기본 프리셋은 성기와 항문을 가립니다.*

[English](README.md) · **한국어** · [日本語](README.ja.md)

> **자동 탐지는 보조 수단이지 보장이 아닙니다.** 특이한 각도, 클로즈업, 가려짐, 조명 때문에 탐지기가 부위를 놓칠 수 있고, 엉뚱한 곳을 가릴 때도 있습니다. 공개하기 전에 결과물을 반드시 사람이 확인하세요. 콘텐츠에 적용되는 법규를 지키는 책임은 사용자에게 있으며, 처리할 권리가 있는 자료에만 사용하세요.

## 기능

- 오픈소스 [NudeNet](https://github.com/notAI-tech/NudeNet) v3 ONNX 모델로 탐지합니다. `320n`은 패키지에 포함돼 있고, 더 정확한 `640m`은 필요할 때 내려받습니다. ONNX Runtime을 통해 CPU, CUDA, DirectML, CoreML에서 돌아갑니다.
- 모자이크 타일을 고정 격자에 맞춰서 프레임 사이에 블록이 흔들리지 않습니다. 기본 타일 크기는 흔히 인용되는 기준인 '이미지 긴 변의 1/100 이상, 최소 4px'을 따릅니다. 블러와 단색 채우기도 고를 수 있습니다.
- 영상에서는 부위를 광학 흐름으로 추적해서, 모자이크가 이동 경로를 따라 번지지 않고 부위를 따라 움직입니다. 탐지 사이 프레임, 탐지기가 잠깐 놓친 구간, 처음 탐지되기 직전 몇 프레임까지 덮습니다. 장면이 바뀌면 추적을 끊고, 오디오는 유지합니다.
- 클래스, 문턱값, 모델, 여백, 타일 크기, 모양, 유지 프레임, 코덱, 화질 등 주요 파라미터를 전부 옵션으로 조절할 수 있습니다.
- 폴더 일괄 처리, 전체 검출 결과 JSON 리포트, 튜닝용 `--draw-boxes` 모드를 지원합니다.
- 한글·일본어 파일명, 알파 채널, 홀수 해상도, NTSC 프레임레이트(29.97 등)를 처리합니다.

## 설치

파이썬 3.10 이상이 필요합니다. ffmpeg는 `imageio-ffmpeg`에 들어 있어서 따로 설치할 필요가 없습니다. 인텔 맥이라면 주의하세요. 최신 ONNX Runtime은 인텔 맥용 휠을 내지 않아서, 사용 중인 파이썬 버전에 맞는 예전 ONNX Runtime이 있을 때만 설치됩니다(인텔 맥의 파이썬 3.14용은 없음).

```bash
git clone https://github.com/fattaillabs/avmosaic && cd avmosaic
pip install .            # 개발용: pip install -e ".[dev]"
```

GPU(선택): `onnxruntime` 대신 `onnxruntime-gpu`(NVIDIA CUDA)나 `onnxruntime-directml`(윈도우, 모든 GPU)을 설치한 뒤 `--device cuda` 또는 `--device directml`을 주면 됩니다. 기본값 `--device auto`는 CUDA → DirectML → CoreML(애플) → CPU 순서로 고릅니다.

## 빠른 시작

```bash
avmosaic video.mp4                    # -> video_censored.mp4
avmosaic photo.jpg -o out.jpg
avmosaic ./scenes -o ./censored       # 폴더 전체, 하위 구조 그대로
avmosaic video.mp4 --model 640m --detect-every 2 --hold 8
avmosaic photo.jpg --draw-boxes       # 무엇을 몇 점으로 잡았는지 확인
avmosaic detect video.mp4 -o detections.json   # 탐지만, 결과 영상 없음
avmosaic classes                      # 클래스·프리셋 목록
avmosaic models download 640m         # 큰 모델 미리 받기
avmosaic frames sample.mp4 -o ./frames --every 2   # 영상 → 라벨용 이미지
avmosaic label ./frames                # 브라우저에서 모자이크가 갈 곳 표시
avmosaic eval regions ./frames         # 모자이크가 표시한 곳에 있는지 확인
avmosaic eval run ./NPDI -o npdi.jsonl  # 라벨 있는 데이터셋으로 성능 측정(아래 참고)
avmosaic eval summary npdi.jsonl
```

`avmosaic 파일 …`은 `avmosaic censor 파일 …`의 줄임입니다.

## 파라미터

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--classes` | `genitals` | 프리셋이나 클래스 이름(쉼표 구분). 프리셋: `genitals`(여성·남성 성기, 항문), `breasts`, `buttocks`, `nudity`, `faces`, `all` |
| `--threshold` | `0.25` | 최소 탐지 점수. 낮출수록 더 많이 가림 |
| `--model` | `320n` | `320n`, `640m`, 또는 `.onnx` 파일 경로 |
| `--resolution` | 모델 기본 | 모델 입력 크기(32의 배수) |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `tensorrt`, `directml`, `coreml`, `openvino` |
| `--channels` | `rgb` | 모델에 넣는 채널 순서(`bgr`이면 원본 NudeNet과 완전히 같음) |
| `--method` | `mosaic` | `mosaic`, `blur`, `fill` |
| `--block-size` | `auto` | 모자이크 타일: `auto`, 픽셀 수(`24`), 긴 변 대비 비율(`2%`) |
| `--shape` | `rect` | `rect` 또는 `ellipse`(타원이어도 박스 전체를 덮음) |
| `--padding` | `0.15` | 박스를 각 변마다 크기의 이 비율만큼 넓힘 |
| `--fill-color` | `black` | `--method fill`일 때 색: `black`, `white`, `#rrggbb`, `r,g,b` |
| `--detect-every` | `1` | 영상: N프레임마다 탐지(장면 전환 때도 탐지). 사이 프레임은 추적으로 채움 |
| `--tracking` | `on` | 영상: 광학 흐름으로 부위 추적. `off`는 `--hold` 프레임 안의 박스 합집합으로 가림(단순하지만 빠른 움직임에서 번짐) |
| `--hold` | `5` | 추적 켬: 처음 탐지되기 전 N프레임까지 거슬러 따라감. 끔: 앞뒤 N프레임 안의 박스도 가림 |
| `--max-lost` | `15` | 추적 켬: 탐지기가 놓친 뒤에도 N프레임 동안 계속 따라감 |
| `--smooth` | `2` | 추적 켬: 박스 위치를 앞뒤 N프레임으로 평균(원래 박스도 항상 덮음) |
| `--codec` / `--crf` / `--preset` | `libx264` / `18` / `medium` | 영상 인코더 설정. 추가 인자는 `--ffmpeg-args "…"` |
| `--no-audio` | 끔 | 오디오 제거. 기본은 복사하고, MP4에 맞지 않는 코덱(PCM 등)이면 AAC로 변환 |
| `--report 파일` | — | 모든 검출 결과(영상은 프레임별)를 JSON으로 저장 |
| `--draw-boxes` | 끔 | 결과에 박스·클래스·점수 표시 |
| `--progress` | `auto` | 진행 표시: 터미널에서는 막대(`auto`), `always`면 로그 파일에도 10초마다 한 줄 |
| `-o`, `--suffix`, `-y` | — | 출력 파일/폴더, `-o`가 없을 때 붙는 접미사, 덮어쓰기 |

### 튜닝 순서

1. `avmosaic sample.mp4 --draw-boxes --report r.json`으로 돌려서 결과를 봅니다.
2. 놓친 부위가 있으면 `--threshold`를 낮추고(예: `0.15`), `--model 640m`을 써 보고, `--hold`와 `--padding`을 키웁니다.
3. 너무 많이 가리면 `--threshold`를 올리고, 리포트에서 어떤 클래스가 잡혔는지 확인합니다.

## 추적 방식

탐지기가 찾은 부위마다 트랙을 만들고, median-flow 광학 흐름으로 프레임마다 따라갑니다. 박스 안 격자 점을 루카스–카나데 방식으로 추적하고, 앞뒤 방향 오차로 잘못 추적한 점을 걸러 냅니다. 덜 가리기보다 더 가리는 쪽으로 설계했습니다.

- **탐지 사이**(`--detect-every N`): 다음 탐지가 오면 쌓인 흐름 오차를 사이 프레임에 나눠 보정합니다. 흐름 경로와 보정 경로를 둘 다 덮습니다.
- **탐지기가 놓친 구간**: 최대 `--max-lost` 프레임까지 광학 흐름으로 계속 따라갑니다. 흐름도 따라가지 못하면(가려짐, 흔들림, 무늬 없음) '멈췄다'와 '최근 속도로 계속 움직였다'를 둘 다 덮습니다. 그 안에 다시 탐지되면, 끊긴 구간 프레임이 아직 버퍼에 남아 있어 두 탐지를 잇는 직선 경로도 함께 덮습니다. 탐지 없이 지나는 프레임마다 박스를 1%씩(최대 +50%) 키웁니다.
- **처음 탐지되기 전**: 새 트랙은 `--hold` 프레임만큼 거꾸로 따라가 덮습니다.
- **장면 전환**: 모든 트랙을 끝내고 그 프레임에서 탐지를 다시 합니다. 이전 장면의 박스를 넘겨 오지 않습니다.

`max(--max-lost, --hold, --smooth, --detect-every − 1)` 프레임만큼 미리 읽은 뒤 출력합니다. 기본값이면 15프레임이고, 1080p 기준 약 100MB입니다.

## 모자이크가 제자리에 있는지 확인

검열 도구에 중요한 건 '이 파일이 포르노인가'가 아니라 '모자이크가 가려야 할 곳에 있는가'입니다. 사람이 샘플 몇 장에 한 번만 표시해 두면, 그 뒤로는 코드나 설정을 바꿀 때마다 자동으로 확인할 수 있습니다.

1. `avmosaic frames sample.mp4 -o ./frames --every 2`로 2초마다 프레임을 이미지로 저장합니다.
2. `avmosaic label ./frames`로 로컬 페이지를 엽니다(127.0.0.1 전용). 가려야 할 곳마다 박스를 드래그하고, 가릴 곳이 없는 이미지는 **C**를 누릅니다. 박스는 이미지 옆에 `<이름>.json`으로 저장됩니다. 처음에는 빈 화면으로 시작합니다. `--suggest`를 주면 탐지기 박스를 미리 채워 빠르지만, 라벨이 탐지기가 이미 찾는 쪽으로 쏠립니다.
3. `avmosaic eval regions ./frames`는 텍스트 보고서를 출력합니다.
   - 표시한 부위가 몇 % 덮였는지(나쁜 경우부터 목록)
   - 모자이크가 표시 밖으로 얼마나 넘쳤는지
   - 깨끗한 이미지를 잘못 가렸는지

   표시 부위가 90%(`--min-coverage`) 미만으로 덮이거나 깨끗한 이미지가 2%(`--max-false-area`) 넘게 가려지면 종료 코드 1을 냅니다. `censor`와 같은 탐지·검열 옵션을 받으므로 설정끼리 바로 비교할 수 있습니다. `AVMOSAIC_SAMPLES=./frames pytest tests/test_samples.py`는 같은 검사를 테스트로 돌립니다.

## 실제 영상으로 시험해 보기

avmosaic은 노출 자료를 포함하거나 호스팅하거나 배포하지 않으며, 테스트에도 쓰지 않습니다. 실제 영상으로 시험해 보려면 figshare에 공개된 [The Image and video dataset for adult content detection in image and video](https://figshare.com/articles/dataset/The_Image_and_video_dataset_for_adult_content_detection_in_image_and_video/14495367?file=27771210) 같은 연구용 데이터셋을 쓸 수 있습니다.

> **18세 이상, 본인 책임.** 이런 데이터셋에는 노골적인 성인 콘텐츠가 들어 있습니다. 내려받기 전에 데이터셋 페이지의 라이선스와 면책 문구를 읽어 보세요. 본인의 책임 하에, 합법인 곳에서만 쓰시고, 그 파일을 이 저장소에 커밋하거나 이슈에 첨부하지 마세요.

```bash
avmosaic "영상.mp4" --draw-boxes --report report.json   # 결과를 직접 보며 확인
avmosaic frames "영상.mp4" -o ./frames --every 2         # 라벨링 + 위치 판정
```

## NPDI·Pornography-2k로 성능 측정

이 연구용 데이터셋들은 공개 다운로드가 아닙니다. 저작권 있는 성인물이라, 정식 신청과 책임 동의서 서명을 거쳐야 받을 수 있습니다.

- **NPDI**(영상 800개, 약 80시간. 절반은 비포르노이고, 그중 200개는 해변·레슬링 같은 '어려운' 영상): 캄피나스 주립대 Sandra Avila 교수
- **Pornography-2k**(영상 2,000개, 약 140시간): 캄피나스 주립대 Anderson Rocha 교수

두 데이터셋 모두 영상 단위 라벨만 있고 부위 박스는 없어서, `avmosaic eval`은 그 라벨로 잴 수 있는 것을 측정합니다. 이미지 폴더도 받으며 이미지 한 장을 샘플 하나로 셉니다(예: 포르노 이미지 대 어려운 오답인 수영복 사진). 라벨은 지정한 폴더(그 이름 포함) 아래 경로로 정합니다. `no[nt][-_ ]?porn`(`nonporn`, `non_porn`, `not_porn` 등)에 맞으면 비포르노, 그다음 `porn`에 맞으면 포르노입니다(`--positive`, `--negative`로 바꿀 수 있음).

```bash
avmosaic eval run ./NPDI -o npdi-320n.jsonl --sample-fps 1             # 중단해도 이어서 실행
avmosaic eval run ./NPDI -o npdi-640m.jsonl --model 640m
avmosaic eval summary npdi-320n.jsonl npdi-640m.jsonl --thresholds 0.15,0.25,0.35
```

요약은 텍스트로만 다음을 보여 줍니다.

- 파일 단위 AUC
- 문턱값별로 포르노·비포르노 영상 중 한 번이라도 가려진 비율과, 샘플 프레임 중 가려진 평균 비율(경로에 easy/difficult가 있으면 나눠서)
- 점수가 가장 높았던 비포르노 파일과 시각(오탐 확인용)
- 아무것도 잡히지 않은 포르노 파일

## 속도

애플 M1(CoreML)에서 1080p 30fps 20초(600프레임) 영상으로 잰 값입니다.

| 설정 | 시간 | 처리 속도 |
|---|---|---|
| 기본(`320n`) | 22.0초 | 27fps |
| `--detect-every 2` | 16.1초 | 37fps |
| `--model 640m --detect-every 3` | 75.6초 | 7.9fps |

1080p 프레임 하나를 탐지하는 데 `320n`은 15.7ms(CPU 33.1ms), `640m`은 346ms(CPU 469ms)가 걸립니다. 같은 영상을 ffmpeg로 재인코딩만 하면 8.3초입니다.

같은 영상을 따로 돌렸을 때 추적은 약 10%를 더했습니다(31.8초 → 34.8초). 이 영상엔 따라갈 부위가 없었으므로 장면 전환 감지 비용입니다. 추적을 켜면 `--detect-every 3`으로 탐지를 줄여도 박스가 정확해서, 20.0초가 걸렸습니다.

## 파이썬 API

```python
from avmosaic import CensorOptions, VideoOptions, censor_image, censor_video, create_detector

detector = create_detector("320n", classes=["FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED"],
                           threshold=0.2)
censor_image("in.jpg", "out.jpg", detector, CensorOptions(block_size="2%"))
censor_video("in.mp4", "out.mp4", detector, CensorOptions(), VideoOptions(hold=8))
```

`detect(bgr_image) -> list[Detection]` 메서드만 있으면 어떤 객체든 탐지기로 쓸 수 있습니다. 직접 학습한 모델로 바꿔 끼울 수도 있습니다.

## 테스트

```bash
pip install -e ".[dev]"
pytest                    # 전체(약 10초)
pytest -m "not model"     # 실제 모델을 돌리는 테스트 제외
```

테스트에는 음란물을 전혀 쓰지 않습니다.

- **파이프라인 테스트**는 모델 대신 색 탐지기를 씁니다. 합성한 빨강·검정 체커 무늬를 가릴 부위로 간주합니다. 모자이크 기하, 프레임 수, 오디오와 길이를 확인하고, 프레임 유지 로직도 확인합니다(일부러 놓친 프레임이 `hold=0`이면 노출되고 `hold=2`면 가려지는지). 홀수 해상도, NTSC, 유니코드 경로도 검사합니다.
- **추적 테스트**는 메모리에서 만든 프레임으로 플래너를 돌려, 매 프레임 실제 물체 박스가 몇 % 덮였는지 정확히 잽니다. 빠른 움직임, 15프레임 연속 미탐지, `--detect-every 6`의 곡선 경로, 거슬러 덮기, 장면 전환, 반대로 움직이는 두 물체, 광학 흐름 강제 실패를 검사합니다. 빠른 움직임 테스트에서 40px 물체는 두 방식 모두 100% 덮입니다. 하지만 가리는 면적은 추적이 물체의 1.03배(프레임당 8px)·1.14배(25px)인 데 비해, `--tracking off`는 2.70배·5.48배입니다.
- **모델 테스트**는 실제 NudeNet 모델을 돌립니다. BGR 모드에서 원본 `nudenet`과 결과가 완전히 같은지 확인하고, 같은 모델의 *얼굴* 클래스와 공개 도메인 사진으로 처음부터 끝까지 검증합니다.
- **내 콘텐츠로 정확도 확인:** 라벨을 붙인 이미지를 비공개 폴더에 두고(절대 커밋하지 마세요) `AVMOSAIC_SAMPLES=/경로 pytest tests/test_samples.py -v`를 실행합니다. 라벨 형식은 [tests/test_samples.py](tests/test_samples.py)에 있습니다.

CI는 리눅스·맥·윈도우에서 파이썬 3.10, 3.12, 3.14로 테스트를 돌립니다.

## 한계

- 오류의 주된 원인은 탐지기입니다. NudeNet은 일반적인 노출 이미지로 학습돼서, 큰 화면 속 작은 부위, 극단적인 클로즈업, 일부가 가려진 경우에 약합니다.
- 가변 프레임레이트(VFR) 영상은 고정 프레임레이트로 바뀝니다(오디오 싱크 유지를 위해).
- 자막·데이터 스트림은 복사되지 않습니다. 홀수 해상도는 H.264 때문에 1px 덧붙입니다.
- 이미지 메타데이터(EXIF)는 출력에 복사되지 않습니다. EXIF 회전 정보는 먼저 반영합니다.

## 라이선스

이 프로젝트 코드는 MIT입니다([LICENSE](LICENSE)). NudeNet 라이선스에 관한 중요한 안내를 포함한 서드파티 구성 요소는 [NOTICE](NOTICE)에 있습니다.
