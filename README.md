# Classificador de Doenças Pulmonares

Este bot no Telegram utiliza um modelo de classificação de doenças pulmonares para analisar imagens de pulmão e fornecer diagnósticos. Ele é capaz de classificar quatro tipos de doenças e identificar imagens normais.

## Instruções de Uso

1. **Classificar uma Imagem:**
   - Envie uma imagem ao bot usando o comando `/classificar`.
   - O bot processará a imagem e fornecerá uma classificação.


2. **Menu de Ajuda:**
   - Para obter informações sobre como usar o bot, digite `/help`.


3. **Retornar ao Menu Principal:**
   - A qualquer momento, você pode retornar ao menu principal usando o comando `/menu`.



## Observações
- Certifique-se de enviar imagens relevantes para obter uma classificação precisa.
- Em caso de problemas ou dúvidas, digite ou clique em `/help` para obter assistência.

## Importações que irão utilizar
import os
import telebot
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

##Criação
Criado por João Pedro Araujo Queiroz Barbosa





