import os
import telebot
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Configuração do bot
bot = telebot.TeleBot("6920707306:AAHeu3kGIqsuav5_GPUlv8_ZYIBMzE4fiBM")

# Configuração do modelo de classificação
class classificador(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(64, 64, (3, 3))
        self.activation = nn.ReLU()
        self.bnorm = nn.BatchNorm2d(num_features=64)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=14 * 14 * 64, out_features=256)
        self.linear2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 5)

    def forward(self, X):
        X = self.pool(self.bnorm(self.activation(self.conv1(X))))
        X = self.pool(self.bnorm(self.activation(self.conv2(X))))
        X = self.flatten(X)

        # Camadas densas
        X = self.activation(self.linear1(X))
        X = self.activation(self.linear2(X))

        # Saída
        X = self.output(X)

        return X

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classificadorLoaded = classificador()
state_dict = torch.load('app/checkpoint.pth', map_location=device)
classificadorLoaded.load_state_dict(state_dict)
classificadorLoaded.to(device)
classificadorLoaded.eval()

# Função para transformar a imagem
def transform_imagem(imagem_teste):
    imagem = Image.open(imagem_teste)
    imagem = imagem.resize((64, 64))
    imagem = imagem.convert('RGB')
    imagem = np.array(imagem.getdata()).reshape(*imagem.size, -1)
    imagem = imagem / 255
    imagem = imagem.transpose(2, 0, 1)
    imagem = torch.tensor(imagem, dtype=torch.float).view(-1, *imagem.shape)
    return imagem

# Função para classificar a imagem
def classificar_imagem(file):
    listaDoencas = ['Atelectasis', 'Effusion', 'Infiltration', 'Nodule', 'Normal']
    imagem = transform_imagem(file)
    classificadorLoaded.eval()
    imagem_teste = imagem.to(device)
    output = classificadorLoaded.forward(imagem_teste)
    output = F.softmax(output, dim=1)
    top_p, top_class = output.topk(k=1, dim=1)
    output = output.detach().numpy()
    index = np.argmax(output)
    return listaDoencas[index]

# Comando para lidar com as imagens enviadas pelos usuários
@bot.message_handler(content_types=['photo'])
def handle_image(message):
    try:
        # Obtém o ID do chat
        chat_id = message.chat.id

        # Obtém o arquivo da imagem
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        # Salva a imagem localmente
        image_path = "temp_image.jpg"
        with open(image_path, 'wb') as new_file:
            new_file.write(downloaded_file)

        # Classifica a imagem
        resultado = classificar_imagem(image_path)

        # Envia a resposta de volta ao usuário
        bot.reply_to(message, f"A imagem foi classificada como: {resultado}\n\n\nPara classificar outra imagem digite ou clique em /classificar \n\nPara retornar ao Menu digite ou clique em /menu.")

        # Remove a imagem temporária
        os.remove(image_path)

    except Exception as e:
        print(e)
        bot.reply_to(message, "Ocorreu um erro ao processar a imagem.")

@bot.message_handler(commands=["classificar"])
def conversao(mensagem):
      print(mensagem)
      bot.send_message(mensagem.chat.id,"você escolheu classificar, coloque sua imagem para fazer a classificação")

@bot.message_handler(commands=["menu"])
def responder(mensagem):
    texto = "Olá! \n\nAqui sou um classificador de doenças pulmonares. Para ter um diagnostico de uma imagem pulmonar digite ou clique em /classificar  \n\n Caso precise de ajuda digite ou clique em /help"
    bot.reply_to(mensagem,texto)

@bot.message_handler(commands=["help"])
def responder(mensagem):
    texto = "Você solicitou por ajuda. O classificador aceita imagens para fazer a classificação. Ele classifica quatro tipos de doença e faz a classificação de imagens normais (imagens sem presença de doença). Para classificar uma imagem digite ou clique em /classificar "
    bot.reply_to(mensagem,texto)


def verificar(mensagem):
        return True

@bot.message_handler(func=verificar)
def responder(mensagem):
    texto = "Olá! \n\nAqui sou um classificador de doenças pulmonares. Para ter um diagnostico de uma imagem pulmonar digite ou clique em /classificar \n\n Caso precise de ajuda digite ou clique em /help"
    bot.reply_to(mensagem,texto)

# Inicia o bot
bot.polling(none_stop=True)
