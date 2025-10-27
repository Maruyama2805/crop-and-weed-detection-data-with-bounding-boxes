# 🌿 Segmentação de Plantas Daninhas com DeepLabV3

Este projeto é um pipeline completo de Deep Learning para **detectar e segmentar culturas (crops) e ervas daninhas (weeds)** usando o modelo DeepLabV3 em PyTorch.

O desafio central foi **converter anotações de Bounding Box (YOLO)** em **Máscaras de Segmentação**, preparando os dados para um modelo de segmentação semântica.

---

## 🎯 Resultado Final

O modelo aprende a classificar cada pixel da imagem em 3 categorias: **Fundo**, **Cultura** ou **Erva Daninha**.

| Imagem Original | Máscara Real | Máscara Prevista pelo Modelo |
| :---: | :---: | :---: |
| _ ![agri_0_975](https://github.com/user-attachments/assets/ef8f8037-ce2b-4cf2-a692-f456daaacab1)
 _ | _ <img width="512" height="512" alt="agri_0_975" src="https://github.com/user-attachments/assets/efba61be-a190-4169-8f9c-7290b01bfd7c" />
 _ | _ <img width="279" height="279" alt="image" src="https://github.com/user-attachments/assets/18f2df89-de94-4499-9a41-9f7727602839" />
_ |

---

## 🚀 O Pipeline (O que foi feito)

O *notebook* (`SegmentacaoDeepLabV3.ipynb`) executa o seguinte fluxo de trabalho:

1.  **📥 Download dos Dados:**
    * Baixa o dataset do Kaggle (que usa anotações YOLO).

2.  **🔳 ➔ 🎨 Conversão (BBox para Máscara):**
    * Lê os arquivos `.txt` (YOLO) e "desenha" máscaras `.png` preenchidas, convertendo retângulos em segmentação pixel-a-pixel.

3.  **🗂️ Organização dos Dados:**
    * Divide os pares de Imagem/Máscara em pastas `train/` e `val/`.

4.  **✨ Data Augmentation:**
    * Usa `albumentations` para aplicar transformações (Flips, Rotação, etc.) de forma idêntica nas imagens e máscaras.

5.  **🧠 Carregamento do Modelo (Transfer Learning):**
    * Carrega o `deeplabv3_resnet50` pré-treinado no COCO.
    * Substitui a camada final (de 21 classes) por uma nova camada para as nossas **3 classes**.

6.  **🏋️ Treinamento:**
    * Treina o modelo usando `CrossEntropyLoss` e salva o melhor checkpoint (`best_deeplab_model.pth`).

---

## 🛠️ Tecnologias Usadas

* **PyTorch** (para o modelo e treinamento)
* **Albumentations** (para Data Augmentation)
* **OpenCV** & **Pillow** (para manipulação de imagens)
* **Scikit-learn** (para a divisão treino/validação)
* **Kaggle API** (para download dos dados)

---

## ⚡ Como Executar

Este projeto é feito para o **Google Colab**.

1.  **Dependências (`requirements.txt`):**
    * Este repositório inclui um arquivo `requirements.txt` que lista todas as bibliotecas Python (PyTorch, Albumentations, OpenCV, etc.) que este projeto utiliza.
    * Se você decidir rodar o projeto localmente (fora do Colab), você pode criar um ambiente virtual (`venv`) e instalar tudo facilmente com um único comando:
    pip install -r requirements.txt

2.  **Credenciais do Kaggle (`kaggle.json`):**
    * Para que o download do dataset funcione no Google Colab, você **precisa** fazer o upload do seu arquivo `kaggle.json`.
    * **LOCALIZAÇÃO CRÍTICA:** É essencial que você solte este arquivo no diretório raiz do Colab, **fora de qualquer outra pasta**. O script procura o arquivo no caminho exato `/content/kaggle.json`. Se ele for colocado dentro de outra pasta, o download falhará.

3.  **Ambiente:** Certifique-se de estar usando um ambiente com **GPU** (`Ambiente de execução > Alterar tipo`).

4.  **Número de Épocas (Epochs):**
    * O *notebook* está configurado com `NUM_EPOCHS = 2`.
    * Isso é intencional para permitir um **teste rápido** do pipeline, já que o `deeplabv3_resnet50` é um modelo grande e cada época de treino demora.
    * Para resultados reais e um modelo bem treinado, **aumente este valor** (ex: 10, 20 ou mais), como sugerido no comentário do próprio código.

5.  **Executar:** Rode as células do *notebook* em ordem.
