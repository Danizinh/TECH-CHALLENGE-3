import cv2
import numpy as np
import face_recognition
from deepface import DeepFace
import os
import mediapipe as mp

def load_images_from_folder(folder):
    known_face_encodings = []  
    known_face_names = []  

    # Percorre todos os arquivos na pasta fornecida
    for filename in os.listdir(folder):
        # Verifica se o arquivo é uma imagem
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Carrega a imagem
            image_path = os.path.join(folder, filename)
            # Carrega a imagem.
            image = face_recognition.load_image_file(image_path)
            # Obtem as codificações faciais (assumindo uma face por imagem)
            face_encodings = face_recognition.face_encodings(image)
            
            if face_encodings:  
                face_encoding = face_encodings[0] 
            # Extrai o nome do arquivo, removendo o sufixo numérico e a extensão
                name = os.path.splitext(filename)[0][:-1]
               # Adiciona a codificação e o nome às listas
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)

    # Retorna as listas de codificações e nomes.
    return known_face_encodings, known_face_names


def analyze_emotions(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        # Verifica se o resultado é uma lista e contém pelo menos um item
        if isinstance(result, list) and len(result) > 0:
            result = result[0]
        
        # Verifica se o resultado contém a chave 'emotion'
        if 'emotion' in result:
            filtered_emotions = {emotion: value for emotion, value in result['emotion'].items() if value > 50}
            print(f"Filtered Emotions: {filtered_emotions}")  # Adiciona um print para depuração
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

    max_frames = 3500
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
            face_encodings = []

            for face_location in face_locations:
                top, right, bottom, left = face_location
                face_image = rgb_frame[top:bottom, left:right]
                try:
                    encodings = face_recognition.face_encodings(face_image)
                    if encodings:
                        face_encodings.append(encodings[0])
                except TypeError as e:
                    # print(f"Erro ao codificar a face: {e}")
                    continue

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name = "Unknown"
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            emotions = analyze_emotions(frame)
            activity = detect_activities(frame, pose)

            emotions_summary.append(emotions)
            activities_summary.append(activity)

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

