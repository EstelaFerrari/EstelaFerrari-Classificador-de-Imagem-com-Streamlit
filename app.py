import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Carregar o modelo treinado
model = load_model('modelo_cachorro_gato.h5')

# Função para pré-processar a imagem
def preprocess_image(img):
    img = img.resize((128, 128))  # Redimensiona para o tamanho que o modelo espera
    img = np.array(img)  # Converte a imagem para um array numpy
    img = img / 255.0  # Normaliza a imagem
    img = np.expand_dims(img, axis=0)  # Adiciona a dimensão do batch
    return img

# Interface do Streamlit
st.title("Classificador de Imagens de Gato e Cachorro")

# Carregar a imagem
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Exibir a imagem carregada
    img = Image.open(uploaded_file)
    st.image(img, caption='Imagem carregada', use_column_width=True)

    # Pré-processar a imagem
    img_array = preprocess_image(img)

    # Fazer a predição
    prediction = model.predict(img_array)

    # Exibir o resultado
    if prediction >= 0.5:
        st.write("A imagem é de um Cachorro!")
    else:
        st.write("A imagem é de um Gato!")
