from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

# Ruta relativa al archivo del modelo
model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model.keras')

# Cargar el modelo
model = load_model(model_path)

# Define las dimensiones de entrada del modelo
IMG_SIZE = (224, 224)  # Ajusta según tu modelo

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']

    try:
        # Abrir la imagen
        image = Image.open(file)
        # Preprocesar la imagen
        image = image.resize(IMG_SIZE)
        image = img_to_array(image) / 255.0  # Normalizar entre 0 y 1
        image = np.expand_dims(image, axis=0)  # Añadir dimensión batch

        # Hacer predicción
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction, axis=1)[0]  # Clase con mayor probabilidad
        confidence = float(np.max(prediction))  # Confianza de la predicción

        result = {'predicted_class': int(predicted_class), 'confidence': confidence}
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)