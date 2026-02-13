import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense

model = Sequential([
    Input(shape=(10,)),
    Dense(1)
])
print("Model başarılı bir şekilde çalıştı!")