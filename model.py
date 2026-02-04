# model.py — код модели для распознавания котов и собак

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Загружаем предобученную модель MobileNetV2 без верхнего слоя
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # замораживаем веса

# Добавляем свои слои для классификации двух классов (кот/собака)
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

# Создаём модель
model = Model(inputs=base_model.input, outputs=output)

# Компилируем модель
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Выводим архитектуру модели
model.summary()
