# Video Recognition

## Descrição

Este projeto realiza reconhecimento facial, análise de expressões emocionais e detecção de atividades em um vídeo. Ao final, gera um resumo das principais atividades e emoções detectadas.

## Funcionalidades

- **Reconhecimento Facial**: Identificação de faces conhecidas no vídeo.
- **Análise de Emoções**: Detecção de emoções como felicidade, tristeza, surpresa, etc.
- **Detecção de Atividades**: Identificação de atividades como "levantar os braços" e outras.
- **Relatório Automatizado**: Geração de um arquivo de resumo das emoções e atividades detectadas.

## Requisitos

- Python 3.x
- Bibliotecas: `opencv-python`, `numpy`, `face_recognition`, `deepface`, `mediapipe`

## Instalação

1. Clone o repositório:

   ```bash
   git clone https://github.com/Danizinh/TECH-CHALLENGE-4.git
   cd TECH-CHALLENGE-4
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Execução

1. Coloque as imagens de referência na pasta `media`.
2. Coloque o vídeo a ser analisado na pasta `media` com o nome `video.mp4`.
3. Execute o script:
   ```bash
   python video_recognition.py
   ```

## Relatório

O resumo das atividades e emoções detectadas será gerado no arquivo `summary_report.txt`.

## Observações

- O script detecta atividades anômalas com base em uma lógica simples que pode ser ajustada conforme necessário.
