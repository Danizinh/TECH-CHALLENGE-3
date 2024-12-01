import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import face_recognition
from deepface import DeepFace
import mediapipe as mp

def analyze_emotions(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        # Verifica se o resultado é uma lista e contém pelo menos um item
        if isinstance(result, list) and len(result) > 0:
            result = result[0]
        
        # Verifica se o resultado contém a chave 'emotion'
        if 'emotion' in result:
            filtered_emotions = {emotion: value for emotion, value in result['emotion'].items() if value > 0.8}
            print(f"Filtered Emotions: {filtered_emotions}")  # Adiciona um print para depuração

            # Detecta sorrisos
            face_landmarks_list = face_recognition.face_landmarks(frame)
            for face_landmarks in face_landmarks_list:
                top_lip = face_landmarks['top_lip']
                bottom_lip = face_landmarks['bottom_lip']
                top_lip_height = np.mean([point[1] for point in top_lip])
                bottom_lip_height = np.mean([point[1] for point in bottom_lip])
                lip_distance = bottom_lip_height - top_lip_height

                if lip_distance > 5:  # Ajuste o valor conforme necessário
                    filtered_emotions['smile'] = 1.0

            return filtered_emotions
        else:
            return {}
    except Exception as e:
        print(f"Erro ao analisar emoções: {e}")
        return {}

def detect_activities(frame, pose):
    """
    Detecta atividades no frame usando MediaPipe Pose.
    Retorna o tipo de movimento detectado.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Verifica se os braços estão levantados
        left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]

        if left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y:
            return "levantar os braços"

        # Adicione outras detecções de movimento aqui

        return "normal"
    else:
        return "anomalous"

def generate_summary(total_frames, anomalies_count, activities_summary, emotions_summary, intervalo):
    try:
        print("Gerando relatório geral...")
        with open('summary_report.txt', 'w', encoding='utf-8') as report:
            report.write("=== Relatório Geral ===\n\n")
            report.write(f"Total de frames analisados: {total_frames // intervalo}\n")
            report.write(f"Número de anomalias detectadas: {anomalies_count}\n\n")
            report.write("Resumo das Atividades:\n")
            for activity in set(activities_summary):
                count = activities_summary.count(activity)
                report.write(f"- {activity}: {count} ocorrências\n")
            report.write("\nResumo das Emoções:\n")
            emotion_totals = {}
            for emotions in emotions_summary:
                for emotion, value in emotions.items():
                    if value > 0.8:
                        emotion_totals[emotion] = emotion_totals.get(emotion, 0) + 1
            for emotion, count in emotion_totals.items():
                report.write(f"- {emotion}: {count} ocorrências\n")
        print("Relatório gerado com sucesso: 'summary_report.txt'")
    except Exception as e:
        print(f"Erro ao gerar o relatório: {e}")

def main():
    video_path = 'media/video.mp4'

    if not os.path.exists(video_path):
        print(f"Erro: O arquivo de vídeo '{video_path}' não foi encontrado.")
        return

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Erro: Não foi possível abrir o arquivo de vídeo '{video_path}'.")
        return

    activities_summary = []
    emotions_summary = []
    total_frames = 0
    anomalies_count = 0
    intervalo = 20

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    print("Iniciando processamento do vídeo...")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Fim do vídeo ou erro ao carregar o frame.")
            break

        total_frames += 1
        
        if total_frames % intervalo == 0:
            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame)

            emotions = {}  # Inicializa a variável emotions

            for (top, right, bottom, left) in face_locations:
                name = "Desconhecido"

                emotions = analyze_emotions(frame[top:bottom, left:right])  # Analisa emoções na região do rosto
                # Filtra emoção com value > 50%
                emotions = {emotion: value for emotion, value in emotions.items() if value > 50}
                
                emotion_text = ', '.join([f"{key}: {int(value)}%" for key, value in emotions.items()])

                # Desenha o retângulo em volta do rosto
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Exibe o nome e as emoções no quadro
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, emotion_text, (left, bottom + 20), font, 0.4, (0, 255, 0), 1)

            activity = detect_activities(frame, pose)

            activities_summary.append(activity)
            emotions_summary.append(emotions)

            if activity == "anomalous":
                anomalies_count += 1

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_capture.release()
    cv2.destroyAllWindows()

    generate_summary(total_frames, anomalies_count, activities_summary, emotions_summary, intervalo)

if __name__ == "__main__":
    main()

