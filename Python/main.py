## 파이썬 메인 기능 
## GUI와 연동하여 스마트홈 제어, whisper를 이용한 음성인식(STT), 얼굴인식, LLM 질의응답 수행 및 구글 TTS 출력
## 주요 라이브러리: faster-whisper, face_recognition, google-generativeai, sounddevice, playsound, opencv-python
## 부득이하게 성능 문제로 인해 CPU 모드로 동작 (Whisper int8, Gemini-2.5-flash 모델 사용)
import socket
import time
import threading
import os
import sys
import urllib.parse
import numpy as np
import sounddevice as sd
import requests
import cv2
import face_recognition
import google.generativeai as genai
from pathlib import Path
from playsound import playsound
from faster_whisper import WhisperModel
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# ---------------------------------------------------------
# [설정] API 키 및 모델 설정
# ---------------------------------------------------------
## -----------------------------------------
## Gemini API 키 안 털리게 조심!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!(이미 한 번 털림)
## 커밋 시 API 키 부분 꼭 지우기!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! -- env 파일로 분리하여 해결
## -----------------------------------------

# 현재 파일 위치 기준으로 key.env 찾기 (절대 경로)
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "key.env")
load_dotenv(env_path)

# 키 확인
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print(f"[ERROR] API 키를 찾을 수 없습니다!")
    print(f"파이썬이 찾은 경로: {env_path}")
    print("key.env 파일 안에 'GEMINI_API_KEY=...' 라고 적혀있는지 확인해주세요.")
else:
    print("[System] API Key Loaded.")

# Gemini 모델 설정
genai.configure(api_key=GEMINI_API_KEY)
GEN_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 1000,
}

try:
    # 요청하신 대로 2.5-flash 모델 유지
    gemini_model = genai.GenerativeModel(
        model_name="gemini-2.5-flash", 
        generation_config=GEN_CONFIG
    )
except Exception as e:
    print(f"[Error] 모델 초기화 실패: {e}")

# ---------------------------------------------------------
# [설정] 네트워크 및 시스템 상수
# ---------------------------------------------------------
JAVA_SERVER_IP = "127.0.0.1"
CMD_PORT = 39186          # 제어 명령 포트
VOICE_SERVER_PORT = 40191 # 음성 트리거 포트
DOOR_EVENT_PORT = 39189   # 도어락 이벤트 포트

SAMPLE_RATE = 16000       # 마이크 샘플링 레이트

# 상태 플래그
g_is_registering = False   # 얼굴 등록 모드 여부
g_is_recognizing = False   # 얼굴 인식 활성화 여부
g_command_lock = False     # 명령 중복 방지 락

# 중복 실행 방지용 타임스탬프 (스마트 케어 실행 시간)
g_last_smart_care_time = 0 
g_indoor_temp = None       # 자바에서 받아온 실내 온도

# ---------------------------------------------------------
# [Data] 명령어 및 매핑 데이터
# ---------------------------------------------------------
# 음성 명령 - 제어 코드 매핑
COMMANDS = [
    # 한국어
    {"kws": ["불 켜", "조명 켜", "전등 켜"], "msg": "조명을 켭니다.", "lang": "ko", "cmd": "LED_ON"},
    {"kws": ["불 꺼", "조명 꺼", "전등 꺼"], "msg": "조명을 끕니다.", "lang": "ko", "cmd": "LED_OFF"},
    {"kws": ["선풍기 켜", "팬 켜"], "msg": "선풍기를 가동합니다.", "lang": "ko", "cmd": "FAN_ON"},
    {"kws": ["선풍기 꺼", "팬 꺼"], "msg": "선풍기를 정지합니다.", "lang": "ko", "cmd": "FAN_OFF"},
    {"kws": ["문 열어", "문 열어줘"], "msg": "네, 문을 열어드릴게요.", "lang": "ko", "cmd": "UNLOCK"},
    
    # 영어
    {"kws": ["turn on light", "lights on"], "msg": "Turning on lights.", "lang": "en", "cmd": "LED_ON"},
    {"kws": ["turn off light", "lights off"], "msg": "Turning off lights.", "lang": "en", "cmd": "LED_OFF"},
    {"kws": ["turn on fan", "fan on"], "msg": "Fan started.", "lang": "en", "cmd": "FAN_ON"},
    {"kws": ["turn off fan", "fan off"], "msg": "Fan stopped.", "lang": "en", "cmd": "FAN_OFF"},
    {"kws": ["open the door", "open door"], "msg": "Unlocking door.", "lang": "en", "cmd": "UNLOCK"},

    # 일본어
    {"kws": ["電気つけて", "ライトオン"], "msg": "電気をつけます。", "lang": "ja", "cmd": "LED_ON"},
    {"kws": ["電気消して", "ライトオフ"], "msg": "電気を消します。", "lang": "ja", "cmd": "LED_OFF"},
    {"kws": ["扇風機つけて", "ファンオン"], "msg": "扇風機をつけます。", "lang": "ja", "cmd": "FAN_ON"},
    {"kws": ["扇風機消して", "ファンオフ"], "msg": "扇風機を止めます。", "lang": "ja", "cmd": "FAN_OFF"},
    {"kws": ["ドア開けて", "ドアオープン"], "msg": "ドアを開けます。", "lang": "ja", "cmd": "UNLOCK"},
]

# Whisper 프롬프트용 키워드 조합
ALL_KEYWORDS = ", ".join([kw for cmd in COMMANDS for kw in cmd['kws']])

# 영문 도시명 -> 한글 변환 매핑 (IP Geolocation 대응)
CITY_MAP = {
    "Seoul": "서울", "Busan": "부산", "Incheon": "인천", "Daegu": "대구",
    "Daejeon": "대전", "Gwangju": "광주", "Suwon": "수원", "Ulsan": "울산",
    "Jeonju": "전주", "Jeju": "제주", "Seongnam": "성남", "Goyang": "고양",
    "Yongin": "용인", "Cheongju": "청주", "Cheonan": "천안", "Pohang": "포항",
    # 필요 시 추가
}

# ---------------------------------------------------------
# [Util] 유틸리티 함수
# ---------------------------------------------------------

def get_current_location():
    """IP 기반 현재 위치(도시) 조회"""
    try:
        res = requests.get("https://ipinfo.io/json", timeout=3)
        city_eng = res.json().get("city", "Seoul")
        return CITY_MAP.get(city_eng, city_eng)
    except:
        return "서울" # Fallback

## gemini 자체로는 날씨 정보를 받아오지 못하여 크롤링으로 대체(네이버 날씨)
def get_realtime_weather():
    """네이버 날씨 크롤링 (현재 위치 기준)"""
    try:
        city = get_current_location()
        url = f"https://search.naver.com/search.naver?query={city}+날씨"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        
        temp = soup.find('div', {'class': 'temperature_text'}).text.strip().replace("현재 온도", "")
        status = soup.find('span', {'class': 'weather before_slash'}).text
        
        return f"{city} 날씨: 기온 {temp}, 상태 {status}"
    except Exception as e:
        print(f"[Weather] 조회 실패: {e}")
        return "날씨 정보 조회 불가"

def ask_gemini(text, lang="ko"):
    """Gemini API 호출 및 응답 생성"""
    try:
        print(f"[Gemini] Query: {text} (Lang: {lang})")
        
        context_info = ""
        weather_kws = ["날씨", "weather", "天気", "tenki"]
        
        # 날씨 키워드 감지 시 컨텍스트 주입
        if any(w in text.lower() for w in weather_kws):
            weather_data = get_realtime_weather()
            print(f"[Gemini] Context Injected: {weather_data}")
            context_info = f"참고 정보: {weather_data}"

        # 언어 설정 정의 위치 수정
        lang_instruction = {
            "ko": "한국어로 간결하게 답변.",
            "en": "Answer briefly in English.",
            "ja": "日本語で簡潔に答えて。"
        }.get(lang, "한국어로 답변.")

        prompt = f"""
        역할: 스마트홈 AI 비서.
        지침: 서론 없이 핵심만 1~2문장으로 답변할 것.
        언어설정: {lang_instruction}
        {context_info}
        사용자 질문: {text}
        """
        
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[Gemini] Error: {e}")
        return "죄송합니다. 처리 중 오류가 발생했습니다."

def speak_answer(text, lang="ko"):
    """Google TTS를 이용한 음성 출력"""
    try:
        enc_text = urllib.parse.quote(text)
        url = f"https://translate.google.com/translate_tts?ie=UTF-8&q={enc_text}&tl={lang}&client=tw-ob"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        
        filename = "temp_voice.mp3"
        with open(filename, 'wb') as f:
            f.write(res.content)
            
        playsound(filename)
        if os.path.exists(filename):
            os.remove(filename)
    except Exception as e:
        print(f"[TTS] Error: {e}")

## java 서버로 TCP 명령 전송 함수
def send_command_to_java(cmd):
    """Java 서버로 TCP 명령 전송"""
    global g_command_lock
    g_command_lock = True
    
    # 락 해제 타이머
    threading.Timer(1.5, lambda: globals().update(g_command_lock=False)).start()

    for _ in range(3): # 최대 3회 재시도
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                s.connect((JAVA_SERVER_IP, CMD_PORT))
                s.sendall((cmd + "\n").encode())
            print(f"[TCP] Sent: {cmd}")
            return True
        except:
            time.sleep(0.5)
    print(f"[TCP] Failed to send: {cmd}")
    return False

# ---------------------------------------------------------
# [New Feature] 스마트 케어 루틴 (온도 제어)
# ---------------------------------------------------------
def run_smart_care_routine():
    """문이 열릴 때 실내 온도를 확인하고 적절한 조치를 취함"""
    global g_indoor_temp, g_last_smart_care_time
    
    # 🔥 [수정] 쿨타임을 10초로 늘려 무한 반복 방지
    if time.time() - g_last_smart_care_time < 10.0:
        print("[SmartCare] 최근 실행되어 건너뜁니다.")
        return

    g_last_smart_care_time = time.time()
    
    print("[SmartCare] 온도 체크 시작... 자바에게 요청 전송")
    
    # 1. 자바에게 온도 물어보기
    g_indoor_temp = None
    if not send_command_to_java("REQ_TEMP"):
        print("[SmartCare] 자바 서버 연결 실패 (명령 전송 불가)")
        return
    
    # 2. 답변 올 때까지 대기 (0.2초 * 15회 = 3초 대기)
    for i in range(15):
        time.sleep(0.2)
        if g_indoor_temp is not None:
            print(f"[SmartCare] {i*0.2:.1f}초 만에 온도 수신 성공!")
            break
    
    # 3. 온도에 따른 판단 및 제어
    if g_indoor_temp is not None:
        print(f"[SmartCare] 측정된 실내 온도: {g_indoor_temp}°C")
        
        if g_indoor_temp <= 18.0:
            msg = f"실내 온도가 {g_indoor_temp}도입니다. 춥네요. 난방기를 켜드릴게요."
            speak_answer(msg, "ko")
            send_command_to_java("FAN_ON")
            
        elif g_indoor_temp >= 26.0:
            msg = f"실내 온도가 {g_indoor_temp}도입니다. 덥네요. 에어컨을 켜드릴게요."
            speak_answer(msg, "ko")
            send_command_to_java("FAN_ON")
        else:
            print("[SmartCare] 온도가 적당함")
            
    else:
        # 끝까지 온도가 안 들어왔을 때
        print("[SmartCare] ❌ 온도 수신 실패 (타임아웃)")
        print("   👉 팁: 자바 프로그램을 껐다가 다시 켰는지 확인해주세요.")

# ---------------------------------------------------------
# [Logic] 얼굴 인식 및 등록
# ---------------------------------------------------------
def start_face_registration():
    global g_is_registering
    g_is_registering = True 
    print("[Face] 등록 모드 시작")
    speak_answer("얼굴 등록을 시작합니다.", "ko")
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        speak_answer("카메라를 찾을 수 없습니다.", "ko")
        g_is_registering = False
        return

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        cv2.putText(frame, "Press 's' to Save, 'q' to Quit", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Face Registration', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'): 
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes = face_recognition.face_locations(rgb)
                if boxes:
                    enc = face_recognition.face_encodings(rgb, boxes)[0]
                    np.save("owner_face.npy", enc)
                    print("[Face] 데이터 저장 완료")
                    speak_answer("얼굴이 등록되었습니다.", "ko")
                    break
                else: 
                    speak_answer("얼굴이 감지되지 않았습니다.", "ko")
            except Exception as e:
                print(f"[Face] 등록 에러: {e}")
        elif key == ord('q'):
            speak_answer("취소되었습니다.", "ko")
            break
            
    cap.release()
    cv2.destroyAllWindows()
    g_is_registering = False

# 얼굴 인식 스레드 함수
def run_face_recognition():
    global g_is_recognizing
    print("[Face] 인식 스레드 대기 중...")
    
    video_capture = None
    last_unlock_ts = 0
    
    while True:
        now = time.time()
        cooldown_active = (now - last_unlock_ts < 10)
        
        # 인식 비활성화 상태거나 쿨타임 중이면 대기
        if not g_is_recognizing or g_is_registering or cooldown_active:
            if video_capture:
                video_capture.release()
                video_capture = None
            time.sleep(1)
            continue
            
        # 카메라 초기화
        if video_capture is None:
            video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not video_capture.isOpened():
                print("[Face] 카메라 오픈 실패")
                g_is_recognizing = False
                continue
            print("[Face] 인식 시작")
            
        # 등록된 얼굴 데이터 로드
        try:
            owner_encoding = np.load("owner_face.npy")
        except:
            speak_answer("등록된 얼굴 데이터가 없습니다.", "ko")
            g_is_recognizing = False
            continue

        ret, frame = video_capture.read()
        if not ret: continue

        # 성능 최적화를 위한 리사이징
        small_frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locs = face_recognition.face_locations(rgb_frame)
        if face_locs:
            encodings = face_recognition.face_encodings(rgb_frame, face_locs)
            for enc in encodings:
                match = face_recognition.compare_faces([owner_encoding], enc, tolerance=0.45)
                if True in match:
                    print("[Face] 인증 성공 -> 잠금 해제")
                    
                    # 1. 환영 인사
                    speak_answer("주인님, 어서 오세요. 문을 엽니다.", "ko")
                    
                    # 2. 스마트 케어 실행 (쿨타임 적용됨)
                    run_smart_care_routine()
                    
                    # 3. 문 열기 명령
                    send_command_to_java("UNLOCK")
                    
                    last_unlock_ts = time.time()
                    g_is_recognizing = False
                    break
                    
    if video_capture: video_capture.release()

threading.Thread(target=run_face_recognition, daemon=True).start()

# ---------------------------------------------------------
# [Network] 서버 리스너
# ---------------------------------------------------------
def gui_command_listener():
    """Java GUI 명령 수신"""
    global g_is_recognizing, g_indoor_temp
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((JAVA_SERVER_IP, CMD_PORT))
                while True:
                    data = s.recv(1024).decode()
                    if not data: break
                    
                    cmds = data.split('\n')
                    for cmd in cmds:
                        cmd = cmd.strip()
                        if not cmd: continue
                        
                        print(f"[GUI Recv] {cmd}")

                        # 자바에서 보내준 온도 데이터 수신 (CURRENT_TEMP:24.5)
                        if cmd.startswith("CURRENT_TEMP:"):
                            try:
                                temp_str = cmd.split(":")[1]
                                g_indoor_temp = float(temp_str)
                            except: pass

                        elif cmd == "REQ_FACE_UNLOCK":
                            speak_answer("카메라를 봐주세요.", "ko")
                            g_is_recognizing = True
                            threading.Timer(10, lambda: globals().update(g_is_recognizing=False)).start()
                        elif cmd == "REGISTER_FACE":
                            threading.Thread(target=start_face_registration).start()
        except:
            time.sleep(3)

threading.Thread(target=gui_command_listener, daemon=True).start()

# 연결 유지 및 키패드 문 열림 감지 리스너
def door_event_listener():
    """도어락 상태 모니터링 (키패드 오픈 감지)"""
    global g_last_smart_care_time
    while True: 
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((JAVA_SERVER_IP, DOOR_EVENT_PORT))
                while True:
                    data = s.recv(1024).decode()
                    if not data: break
                    
                    # 문 열림 신호가 왔을 때
                    if "UNLOCKED" in data:
                        # 🔥 [수정] 쿨타임 10초 적용 (최근 얼굴인식/음성으로 연 게 아닐 때만 실행)
                        if time.time() - g_last_smart_care_time > 10.0:
                            print("[Door] Keypad/Manual Unlock Detected")
                            speak_answer("문이 열렸습니다.", "ko")
                            
                            # 키패드로 열었을 때도 스마트 케어(온도 체크) 실행
                            run_smart_care_routine()
        except:
            time.sleep(3)

threading.Thread(target=door_event_listener, daemon=True).start()

# ---------------------------------------------------------
# [Voice] 음성 인식 및 처리
# ---------------------------------------------------------
print("[System] Whisper 모델 로딩 중...")
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
print("[System] 준비 완료")

recording_state = {'active': False, 'stream': None, 'chunks': []}

# 녹음 스레드 함수
def record_audio_thread():
    """백그라운드 녹음 스레드"""
    global recording_state
    while recording_state['active']:
        try:
            if recording_state['stream']:
                chunk, _ = recording_state['stream'].read(SAMPLE_RATE // 10)
                if chunk is not None:
                    recording_state['chunks'].append(chunk)
        except: break

# 녹음 시작
def start_recording():
    if recording_state['active']: return
    
    print("[Voice] 녹음 시작")
    recording_state['active'] = True
    recording_state['chunks'] = []
    
    try:
        recording_state['stream'] = sd.InputStream(channels=1, samplerate=SAMPLE_RATE, dtype=np.float32)
        recording_state['stream'].start()
        threading.Thread(target=record_audio_thread, daemon=True).start()
    except Exception as e:
        print(f"[Voice] Mic Error: {e}")
        recording_state['active'] = False

# 녹음 종료 및 처리
def stop_and_process():
    if not recording_state['active']: return
    
    print("[Voice] 녹음 종료 및 분석")
    recording_state['active'] = False
    time.sleep(0.2) # 버퍼 플러시 대기
    
    if recording_state['stream']:
        recording_state['stream'].stop()
        recording_state['stream'].close()
        recording_state['stream'] = None

    if not recording_state['chunks']: return

    # 오디오 데이터 병합 및 정규화
    audio_data = np.concatenate(recording_state['chunks'], axis=0)
    max_val = np.max(np.abs(audio_data))
    if max_val > 0: audio_data = audio_data / max_val * 0.9
    
    # Whisper 처리를 위해 임시 파일 저장
    temp_wav = "temp_req.wav"
    import soundfile as sf
    sf.write(temp_wav, audio_data, SAMPLE_RATE)
    
    try:
        # STT 수행
        segments, info = whisper_model.transcribe(
            temp_wav, 
            beam_size=5, 
            vad_filter=True, 
            initial_prompt=f"Commands: {ALL_KEYWORDS}"
        )
        text = " ".join([s.text for s in segments]).strip()
        lang = info.language
        
        print(f"[STT] Result: '{text}' (Lang: {lang})")
        
        if text:
            for cmd in COMMANDS:
                if any(kw in text.lower() for kw in cmd["kws"]):
                    print(f"[Intent] Command Detected: {cmd['cmd']}")
                    speak_answer(cmd["msg"], cmd["lang"])
                    
                    # 음성으로 '문 열어' 했을 때도 스마트 케어 실행
                    if cmd["cmd"] == "UNLOCK":
                        run_smart_care_routine()
                    
                    send_command_to_java(cmd["cmd"])
                    return

            # LLM 질의 (Gemini)
            answer = ask_gemini(text, lang)
            print(f"[Gemini] Answer: {answer}")
            speak_answer(answer, lang)
        else:
            print("[Voice] 음성 미감지")
            
    except Exception as e:
        print(f"[Voice] Analysis Error: {e}")
    finally:
        if os.path.exists(temp_wav): os.remove(temp_wav)

def voice_trigger_server():
    """GUI의 음성 버튼 이벤트 수신"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", VOICE_SERVER_PORT))
            s.listen(5)
            while True:
                conn, _ = s.accept()
                data = conn.recv(1024).decode().strip()
                if data == "START_RECORDING":
                    start_recording()
                elif data == "STOP_RECORDING":
                    stop_and_process()
                conn.close()
    except Exception as e:
        print(f"[VoiceServer] Error: {e}")

threading.Thread(target=voice_trigger_server, daemon=True).start()

# ---------------------------------------------------------
# Main Loop
# ---------------------------------------------------------
print("\n" + "="*40)
print("   Smart Home AI Assistant v1.0")
print("   Modules: Whisper, Gemini, FaceRec")
print("="*40 + "\n")

try:
    while True: time.sleep(1)
except KeyboardInterrupt:
    print("\n[System] Shutting down...")