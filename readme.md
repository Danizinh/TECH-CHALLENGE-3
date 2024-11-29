# Video Recognition

## Descrição

Este projeto utiliza reconhecimento facial, análise de emoções e detecção de atividades para processar um vídeo e gerar um relatório detalhado. O sistema é capaz de identificar faces, reconhecer emoções e detectar atividades normais ou anômalas no vídeo. Ao final, é gerado um relatório consolidado das atividades e emoções detectadas.


## Principais Funcionalidades

- **Reconhecimento Facial**: Detecção e marcação de faces nos frames do vídeo.
- **Análise de Emoções**: Identificação de emoções (como alegria, tristeza, raiva, etc.) usando a biblioteca `DeepFace`.
- **Detecção de Atividades**: Identificação de atividades normais e anômalas com base no movimento captado nos frames.
- **Relatório Automatizado**: Geração de um arquivo de resumo das emoções e atividades detectadas.

## Requisitos

- Python 3.x
- Bibliotecas: `opencv-python`, `numpy`, `face_recognition`, `fer`

## Instalação

1. Clone o repositório:

   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd <NOME_DO_REPOSITORIO>
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



