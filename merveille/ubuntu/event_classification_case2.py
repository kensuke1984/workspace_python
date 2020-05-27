from util import event_folders
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from matplotlib import pyplot as plt

root = '/home/kensuke/workspace/ml/event_classification/case2/syn'
events = event_folders(root)


def down_sampl(data):
    return data[::20]


def normalize(data):
    return data / np.max(data)


def create_encoder(files):
    event_ids = list(set(file.name.split('.')[1] for file in files))
    list.sort(event_ids)
    encoder = LabelEncoder()
    encoder.fit(event_ids)
    return encoder


def read_dat(filenames, preprocess=lambda d: d):
    from readsac import read_sac
    return np.array([preprocess(read_sac(file)) for file in filenames])


def get_labels(files):
    files = [str(file) for file in files]
    return label_encoder.transform([id for file in files for id in label_encoder.classes_ if id in file])


allfiles = list(file for event in events for file in event.glob('*Ts'))

sacfiles = np.random.choice(allfiles, 3000, replace=False)
sacfiles = allfiles

print('Number of files:', len(sacfiles))
label_encoder = create_encoder(sacfiles)
num_class = len(label_encoder.classes_)
print('Number of events:', num_class)

preprocess = lambda d: normalize(down_sampl(d))

x = read_dat(sacfiles, preprocess)
y = get_labels(sacfiles)
num_one_data = len(x[0])
print('Number of datapoints in one file:', num_one_data)

train_x, test_x, train_y, test_y = train_test_split(x, y)
print('Number of train files:', len(train_x))
print('Number of test files:', len(test_x))

model = keras.Sequential([
    keras.layers.Dense(200, activation='relu', input_shape=(num_one_data,)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(num_class, activation='softmax')
])
model.summary()
epochs = 50
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
hist = model.fit(train_x, train_y, epochs=epochs, validation_data=(test_x, test_y))

plt.plot(range(1, epochs + 1), hist.history['accuracy'],label='train')
plt.plot(range(1, epochs + 1), hist.history['val_accuracy'],label='test')
plt.show()