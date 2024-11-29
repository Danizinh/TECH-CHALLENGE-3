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
        # Analisa as emoções no frame usando DeepFace.
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        # Verifica se o resultado é um dicionário com emoções.
        if isinstance(result, dict) and 'emotion' in result:
            return result['emotion']  # Retorna as emoções encontradas.
        
        # Verifica se o resultado é uma lista com emoções.
        elif isinstance(result, list) and len(result) > 0 and 'emotion' in result[0]:
            return result[0]['emotion']  # Retorna as emoções encontradas.
        
        # Se nada for encontrado, retorna um dicionário vazio.
        else:
            return {}
    except Exception as e:
        # Mostra um erro no terminal se algo der errado.
        print(f"Erro ao analisar emoções: {e}")
        return {}  # Retorna vazio em caso de erro.

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

    # Verifica cada contorno detectado.
    for contour in contours:
        # Ignora pequenos contornos que podem ser ruído
        if cv2.contourArea(contour) >= anomaly_threshold:
            return "anomalous"

    # Se nenhum movimento significativo for encontrado, retorna "normal".
    return "normal"


def generate_summary(total_frames, anomalies_count, activities_summary, emotions_summary):
    try:
        # Mostra mensagem inicial de que o relatório está sendo gerado.
        print("Gerando relatório geral...")

        # Cria ou sobrescreve o arquivo 'summary_report.txt'.
        with open('summary_report.txt', 'w', encoding='utf-8') as report:
            # Escreve o cabeçalho do relatório.
            report.write("=== Relatório Geral ===\n\n")
            report.write(f"Total de frames analisados: {total_frames}\n")
            report.write(f"Número de anomalias detectadas: {anomalies_count}\n\n")

            # Adiciona o resumo das atividades no relatório.
            report.write("Resumo das Atividades:\n")
            for activity in set(activities_summary):  # Para cada tipo de atividade encontrado.
                count = activities_summary.count(activity)  # Conta quantas vezes essa atividade apareceu.
                report.write(f"- {activity}: {count} ocorrências\n")
            
            # Adiciona o resumo das emoções no relatório.
            report.write("\nResumo das Emoções:\n")
            emotion_totals = {}  # Cria um dicionário para contar as emoções.
            for emotions in emotions_summary:  # Passa por todas as emoções detectadas.
                for emotion, value in emotions.items():  # Para cada emoção encontrada.
                    emotion_totals[emotion] = emotion_totals.get(emotion, 0) + 1  # Soma as ocorrências.
            for emotion, count in emotion_totals.items():  # Escreve as emoções e suas contagens no relatório.
                report.write(f"- {emotion}: {count} ocorrências\n")

        # Confirma que o relatório foi gerado com sucesso.
        print("Relatório gerado com sucesso: 'summary_report.txt'")
    except Exception as e:
        # Exibe um erro caso algo dê errado ao gerar o relatório.
        print(f"Erro ao gerar o relatório: {e}")



def main():
    video_path = 'media/video.mp4'  # Caminho para o vídeo

    # Verifica se o arquivo de vídeo existe.
    if not os.path.exists(video_path):
        print(f"Erro: O arquivo de vídeo '{video_path}' não foi encontrado.")
        return

    # Abre o arquivo de vídeo.
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Erro: Não foi possível abrir o arquivo de vídeo '{video_path}'.")
        return

    max_frames = 10  # Número máximo de frames a serem processados.
    activities_summary = []  # Lista para armazenar atividades detectadas.
    emotions_summary = []  # Lista para armazenar emoções detectadas.
    total_frames = 0  # Contador de frames processados.
    anomalies_count = 0  # Contador de anomalias detectadas.

    # Inicializa o subtrator de fundo
    background_subtractor = cv2.createBackgroundSubtractorMOG2()

    print("Iniciando processamento do vídeo...")

    # Loop para processar frames do vídeo.
    while total_frames < max_frames:
        ret, frame = video_capture.read()  # Lê o próximo frame.
        if not ret:  # Para o loop se não houver mais frames.
            print("Fim do vídeo ou erro ao carregar o frame.")
            break

        total_frames += 1  # Atualiza o contador de frames.
        print(f"Processando frame {total_frames}...")

        # Detecta atividade no frame.
        activity = detect_activities(frame, background_subtractor)
        activities_summary.append(activity)  # Adiciona a atividade à lista.
        if activity == "anomalous":  # Conta anomalias.
            anomalies_count += 1

        # Analisa emoções no frame.
        emotions = analyze_emotions(frame)
        if emotions:  # Se detectar emoções, adiciona à lista.
            emotions_summary.append(emotions)
        else:  # Caso contrário, adiciona como indefinido.
            emotions_summary.append({"Indefinido": 100})

    # Libera o vídeo e fecha as janelas.
    video_capture.release()
    cv2.destroyAllWindows()

    print("Processamento concluído.")
    print(f"Frames processados: {total_frames}")
    print(f"Atividades coletadas: {len(activities_summary)}")
    print(f"Emoções coletadas: {len(emotions_summary)}")

    # Gera o relatório com os dados processados.
    generate_summary(total_frames, anomalies_count, activities_summary, emotions_summary)

# Executa o programa principal.
if __name__ == "__main__":
    main()

