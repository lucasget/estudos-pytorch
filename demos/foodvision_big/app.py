### 1. Importar e configurar os nomes das classes ###
import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Configurar os nomes das classes
with open("class_names.txt", "r") as f:
    class_names = [food_name.strip() for food_name in f.readlines()]

### 2. Preparação do modelo e da transformação ###
# Criar o modelo e as transformações
effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes = 101)

# Salvar os pesos salvos
effnetb2.load_state_dict(
    torch.load(f = "09_pretrained_effnetb2_feature_extractor_food101_20_percent.pth",
    map_location = torch.device("cpu") ) # carregar para a CPU
)

### 3. Função de previsão ###
def predict(img) -> Tuple[Dict, float]:
    # Iniciar um cronometro
    start_time = timer()

    # Transformar a imagem de entrada para usar com EffNetB2
    img = effnetb2_transforms(img).unsqueeze(0) # unsqueeze = adicionar a dimensão de lote no indice 0

    # Colocar o modelo em modo eval(), fazer a previsão
    effnetb2.eval()
    with torch.inference_mode():
        # Passar a imagem transformada pelo modelo e transformar os logits de previsão em probabilidades
        pred_probs = torch.softmax(effnetb2(img), dim = 1)

    # Criar um dicionario de rotulo de previsão e probabilidade de previsão
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    # Calcular o tempo de previsão
    end_time = timer()
    pred_time = round(end_time - start_time, 4 )
    # Retornar o dicionario de previsão e o tempo de previsão
    return pred_labels_and_probs, pred_time

### 4. App Gradio ###
# Create title, description and article strings
title = "FoodVision BIG 🍔"
description = "An EfficientNetB2 feature extractor computer vision model to classify images of 101 classes of food from Food101 dataset ."
article = "Created at [09. PyTorch Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/#11-turning-our-foodvision-big-model-into-a-deployable-app"

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=gr.Image(type="pil"), # what are the inputs?
                    outputs=[gr.Label(num_top_classes=5, label="Predictions"), # what are the outputs?
                             gr.Number(label="Prediction time (s)")], # our fn has two outputs, therefore we have two outputs
                    # Create examples list from "examples/" directory
                    examples=example_list, 
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
demo.launch()
