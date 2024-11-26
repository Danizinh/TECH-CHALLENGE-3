import cv2
import numpy as np
import face_recognition
from deepface import DeepFace
import os

def load_images_from_folder(folder):
    known_face_encodings = []
    known_face_names = []

    # Percorre todos os arquivos na pasta fornecida
    for filename in os.listdir(folder):
        # Verifica se o arquivo é uma imagem
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Carrega a imagem
            image_path = os.path.join(folder, filename)
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

    return known_face_encodings, known_face_names

def analyze_emotions(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        return result['emotion']
    except Exception as e:
        print(f"Erro ao analisar emoções: {e}")
        return {}

def detect_activities(frame, background_subtractor):
    """
    Detecta atividades no frame usando subtração de fundo.
    Retorna 'normal' ou 'anomalous' com base na detecção de movimento.
    """
    # Aplica a subtração de fundo para obter a máscara de movimento
    fg_mask = background_subtractor.apply(frame)

    # Encontra contornos na máscara de movimento
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Define um limiar para considerar uma atividade como anômala
    anomaly_threshold = 500

    for contour in contours:
        # Ignora pequenos contornos que podem ser ruído
        if cv2.contourArea(contour) < anomaly_threshold:
            continue

        # Se encontrar um contorno grande, considera como atividade anômala
        return "anomalous"

    # Se não encontrar contornos grandes, considera como atividade normal
    return "normal"

def main():
    image_folder = 'media'  # Caminho para a pasta de imagens
    video_path = 'media/video.mp4'  # Caminho para o vídeo

    known_face_encodings, known_face_names = load_images_from_folder(image_folder)  # Carrega imagens e codificações

    video_capture = cv2.VideoCapture(video_path)  # Inicia captura de vídeo do arquivo

    activities_summary = []
    emotions_summary = []
    total_frames = 0
    anomalies_count = 0

    # Inicializa o subtrator de fundo
    background_subtractor = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                name = known_face_names[best_match_index]
            else:
                name = "Unknown"

            # Do something with the name (e.g., draw it on the frame)

        # Other processing...

    video_capture.release()
    cv2.destroyAllWindows()

    # Geração de resumo
    generate_summary(total_frames, anomalies_count, activities_summary, emotions_summary)

def generate_summary(total_frames, anomalies_count, activities_summary, emotions_summary):
    with open('summary_report.txt', 'w') as report:
        report.write(f"Total de frames analisados: {total_frames}\n")
        report.write(f"Número de anomalias detectadas: {anomalies_count}\n")
        report.write("Resumo das atividades:\n")
        for activity in activities_summary:
            report.write(f"{activity}\n")
        report.write("Resumo das emoções:\n")
        for emotion in emotions_summary:
            report.write(f"{emotion}\n")

if __name__ == "__main__":
    main()