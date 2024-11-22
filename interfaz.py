import gradio as gr
import requests

# URL de la API Flask
API_URL = "http://127.0.0.1:5000/predict"

def predict_with_interface(image):
    # Guardar la imagen en memoria para enviarla a la API
    with open("temp_image.jpg", "wb") as temp_file:
        image.save(temp_file)

    # Enviar la imagen a la API
    with open("temp_image.jpg", "rb") as img_file:
        files = {"file": img_file}
        response = requests.post(API_URL, files=files)

    # Procesar la respuesta de la API
    if response.status_code == 200:
        data = response.json()
        predicted_class = data.get("predicted_class", "Unknown")  # Nombre del ave predicha
        confidence = data.get("confidence", 0.0)  # Confianza en la predicción

        # Convertir la confianza en un porcentaje
        confidence_percentage = confidence * 100  # Multiplicamos por 100 para convertirlo en porcentaje
        return predicted_class, f"{confidence_percentage:.2f}%"  # Devuelve el nombre del ave y el porcentaje de confianza
    else:
        return "Error", "No se pudo realizar la predicción"

# Crear la interfaz
interface = gr.Interface(
    fn=predict_with_interface,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Textbox(label="Nombre del Ave"), gr.Textbox(label="Porcentaje de Confianza")],
    title="Predictor de Imágenes de Aves del Tolima",
    description="Carga una imagen de un ave y predice su especie con el modelo."
)

if __name__ == '__main__':
    interface.launch()
