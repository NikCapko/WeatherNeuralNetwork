import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils, plot_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

INPUT = 8
OUTPUT = 82
EPOCHS = 200
SEED = 45
BATCH_SIZE = 32

def my_model():
    model = Sequential()
    model.add(Dense(INPUT, input_dim=INPUT, kernel_initializer='normal', activation='relu'))
    
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(OUTPUT, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model

def normalize_feature(data, d_min, d_max):
    f_min = -1.0
    f_max = 1.0
    factor = (f_max - f_min) / (d_max - d_min)
    normalized = f_min + (data - d_min) * factor
    return normalized

dataframe = pd.read_csv('Moscow_test.csv', header=0)
dataset = dataframe.values
X = dataset[:, 1:9].astype(float)
Y = dataset[:, 9]

X[:, 0] = normalize_feature(X[:, 0], 0, 12)
X[:, 1] = normalize_feature(X[:, 1], 0, 24)
X[:, 2] = normalize_feature(X[:, 2], -50, 50)
X[:, 3] = normalize_feature(X[:, 3], 700, 800)
X[:, 4] = normalize_feature(X[:, 4], 0, 100)
X[:, 5] = normalize_feature(X[:, 5], 0, 34)
X[:, 6] = normalize_feature(X[:, 6], 0, 50)
X[:, 7] = normalize_feature(X[:, 7], 0, 100)

y = np_utils.to_categorical(Y, OUTPUT)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.1, random_state=SEED)

model = my_model()

results = model.fit(X_train, Y_train, batch_size=BATCH_SIZE,
                    epochs=EPOCHS, validation_split=0.2, verbose=2)

print("\n Train-Accuracy: %.2f \n" % (np.mean(results.history["acc"])*100))
print("\n Train-Accuracy (val): %.2f \n" % (np.mean(results.history["val_acc"])*100))
print("\n Train-Loss: %.2f \n" % (np.mean(results.history["loss"])*100))
print("\n Train-Loss (val): %.2f \n" % (np.mean(results.history["val_loss"])*100))

scores = model.evaluate(X_test, Y_test)
print("\n Test-Loss: %.2f \n" % (scores[0] * 100))
print("\n Test-Accuracy: %.2f \n" % (scores[1] * 100))

# Генерируем описание модели в формате json
model_json = model.to_json()
# Записываем модель в файл
json_file = open("my_model.json", "w")
json_file.write(model_json)
json_file.close()

# Сохраняем веса
model.save_weights('my_model_weights.h5')

