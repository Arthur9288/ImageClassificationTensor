import os
import tensorflow as tf
from tensorflow.keras import layers, models

# Configuração para evitar consumo total da memória da GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 1. Carregamento dos dados
data_dir = 'data'
img_height = 224
img_width = 224
batch_size = 32

raw_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='binary'
)

# Divisão dos dados (70% treino, 20% validação, 10% teste)
dataset_size = len(raw_dataset)
train_size = int(dataset_size * 0.7)
val_size = int(dataset_size * 0.2)
test_size = dataset_size - train_size - val_size

train_dataset = raw_dataset.take(train_size)
remaining = raw_dataset.skip(train_size)
val_dataset = remaining.take(val_size)
test_dataset = remaining.skip(val_size)

# 2. Pré-processamento
def preprocess(image, label):
    # Normalização de pixel para o MobileNetV2: escala [-1, 1]
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

train_dataset = train_dataset.map(preprocess).prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.map(preprocess).prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.map(preprocess).prefetch(buffer_size=tf.data.AUTOTUNE)

# 3. Definição do modelo com Data Augmentation e Transfer Learning
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# Base do MobileNetV2 pré-treinada
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Congelar pesos da base

# Arquitetura final
inputs = layers.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)

# 4. Compilação e Treinamento
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

print("Iniciando o treinamento...")
epochs = 10
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=val_dataset
)

# Avaliação no conjunto de teste
print("Avaliando no conjunto de teste...")
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# 5. Salvar o modelo final
# Formato nativo .keras (recomendado pelo Keras 3.x)
model_name = 'modelo_amazonia.keras'
model.save(model_name)
print(f"Modelo salvo com sucesso como {model_name}")

# Salvar também no formato legado .h5 para compatibilidade
model.save('modelo_amazonia.h5')
print("Backup salvo em modelo_amazonia.h5")
