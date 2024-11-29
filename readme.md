# Video Recognition

## Descrição

Este projeto realiza reconhecimento facial, análise de expressões emocionais e detecção de atividades em um vídeo. Ao final, gera um resumo das principais atividades e emoções detectadas.

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



