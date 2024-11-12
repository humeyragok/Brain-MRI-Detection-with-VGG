from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Modeli yükle
model = tf.keras.models.load_model('model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    if file:
        # Resmi yükle ve işleme



        img = Image.open(file)
        img = img.resize((224, 224))  # Modelinizin gerektirdiği boyut
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Batch boyutunu ekle

        # Tahmin yap
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction[0])
        class_label = 'No Tumor' if class_idx == 0 else 'Tumor'
        confidence = prediction[0][class_idx]

        return render_template('index.html', prediction=class_label, confidence=confidence)
    

if __name__ == '__main__':
    app.run(debug=True)