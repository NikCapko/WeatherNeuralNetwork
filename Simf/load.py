
import pandas as pd
from keras.models import model_from_json
from keras.utils import np_utils
from convert_ww import convert

def normalize_feature(data, d_min, d_max):
    f_min = -1.0
    f_max = 1.0
    factor = (f_max - f_min) / (d_max - d_min)
    normalized = f_min + (data - d_min) * factor
    return normalized

dataframe = pd.read_csv('control.csv', header=6)
dataset = dataframe.values
X = dataset[:, 1:9].astype(float)
Y = dataset[:, 9]
#Y = np_utils.to_categorical(Y, 67)

X[:, 0] = normalize_feature(X[:, 0], 0, 12)
X[:, 1] = normalize_feature(X[:, 1], 0, 24)
X[:, 2] = normalize_feature(X[:, 2], -50, 50)
X[:, 3] = normalize_feature(X[:, 3], 700, 800)
X[:, 4] = normalize_feature(X[:, 4], 0, 100)
X[:, 5] = normalize_feature(X[:, 5], 0, 34)
X[:, 6] = normalize_feature(X[:, 6], 0, 50)
X[:, 7] = normalize_feature(X[:, 7], 0, 100)

# Загружаем данные об архитектуре сети из файла json
json_file = open("my_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
# Создаем модель на основе загруженных данных
loaded_model = model_from_json(loaded_model_json)
# Загружаем веса в модель
loaded_model.load_weights("my_model_weights.h5")

# Компилируем модель
loaded_model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
# Проверяем модель на тестовых данных
scores = loaded_model.predict(X) 

out_list = list()

for l in scores:
    s = list()
    for i in range(len(l)):
        if (round(l[i] * 100, 2) >= 3):
            s.append({convert(i): round(l[i] * 100, 2)})
    out_list.append(s)

for i in range(len(out_list)):
    print('', i, "Состояние погоды: ", convert(Y[i]))
    print('', i, "Предсказание: ")
    for i in out_list[i]:
        print("    ", i)
    print()
print()

