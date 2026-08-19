### 1. Importaçao e configuraçao dos nomes das classes ###
import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Configurar os nomes das classes
class_names = ['pizza', 'steak', 'sushi']

### 2. Preparaçao do modelo e da transformaçao ###
effnetb2, effnetb2_transforms = create_effnetb2_model(
    num_classes = 3)

# Carregar os pesos salvos
effnetb2.load_state_dict(
    torch.load(f = '09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth',
    map_location('cpu')) # carregar o modelo na CPU
)

### 3. Funçao de previsao ###

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
# Criar titulo, descrição e artigo
title = "FoodVision Mini 🍕🥩🍣"
description = "Um modelo de visão computacional: EfficientNetB2 feature extractor para classificar imagens como pizza, bife ou sushi"
article = "Criado em [09. Deploy de modelos em PyTorch](https://www.learnpytorch.io/09_pytorch_model_deployment/#74-building-a-gradio-interface)."

# Criar uma lista de exemplos
example_list = [['examples/' + example] for example in os.listdir('examples')]

# Criar uma demosntração Gradio
demo = gr.Interface(fn = predict, # mapeia entradas para saidas
                    inputs = gr.Image(type = "pil"),
                    outputs = [gr.Label(num_top_classes = 3, label = "previsões"),
                               gr.Number(label = "Tempo de previsão (s)")],
                               examples = example_list,
                               title = title,
                               description = description,
                               article = article)  

# Lançar o demo
demo.launch(debug = False, # imprimir erros localmente? 
            share = True) # Gerar um URL compartilhável?
