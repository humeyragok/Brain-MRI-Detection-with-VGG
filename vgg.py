import os
import cv2
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import vgg16
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import tensorflow as tf

# Veri seti dizinleri

yes_dir = 'C:/Users/hmyrg/OneDrive/Brain MRI/yes'
no_dir = 'C:/Users/hmyrg/OneDrive/Brain MRI/no'


# Resimlerin yüklenmesi
X = []
y = []

def load_images_from_directory(directory, label):
    for img_name in tqdm(os.listdir(directory)):
        img_path = os.path.join(directory, img_name)
        if not os.path.isfile(img_path):
            print(f"Dosya mevcut değil: {img_path}")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"Resim yüklenemedi: {img_path}")
            continue
        img = cv2.resize(img, (224, 224))
        X.append(img)
        y.append(label)

# Resimleri yükle

load_images_from_directory(no_dir, 'No Tumor') #0
load_images_from_directory(yes_dir, 'Tumor') #1


X = np.array(X)
y = np.array(y)

# Veri kontrolü
if len(X) == 0 or len(y) == 0:
    raise ValueError("Veri kümesi boş. Resimlerin doğru yüklendiğinden emin olun.")

# Verilerin eğitim ve test setlerine ayrılması
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print("Eğitim seti şekli: ", X_train.shape)
print("Test seti şekli: ", X_test.shape)

# Etiketleri dönüştürme
le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)  # Fit transform sadece eğitim verisi için kullanılır
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

# VGG16 modelini yükleme
img_rows, img_cols = 224, 224
vgg = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))

# Transfer öğrenme
for layer in vgg.layers:
    layer.trainable = False

# Üst katmanları ekleme
def lw(bottom_model, num_classes):
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(512, activation='relu')(top_model)
    top_model = Dense(num_classes, activation='softmax')(top_model)
    return top_model

FC_Head = lw(vgg, 2)
model = Model(inputs=vgg.input, outputs=FC_Head)

# Modeli derleme
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
history = model.fit(X_train, y_train, epochs=6, validation_data=(X_test, y_test), verbose=1)

# Eğitim ve doğrulama doğruluğu çizimi
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Eğitim doğruluğu')
plt.plot(epochs, val_acc, 'b', label='Doğrulama doğruluğu')
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.legend(loc=0)
plt.show()

# Karışıklık matrisi çizimi
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
class_names = ['Tumor', 'No Tumor']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Tahmin Edilen Etiketler')
plt.ylabel('Gerçek Etiketler')
plt.title('Karışıklık Matrisi')
plt.show() 


model.save('model.h5')