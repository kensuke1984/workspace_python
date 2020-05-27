from util import event_folders
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.utils import to_categorical


def get_files(event: Path, cond=None):
    if cond is None:
        return [file for file in event.glob('*')]
    else:
        return [file for file in event.glob('*') if cond(file)]
    pass


def down_sampl(data):
    return data[::20]


def read_dat(filenames, preprocess=lambda d: d):
    from readsac import read_sac
    return np.array([preprocess(read_sac(file)) for file in filenames])


def get_labels(files):
    files = [str(file) for file in files]
    return id_label.transform([id for file in files for id in id_label.classes_ if id in file])


transv = lambda name: name.suffix.endswith('.Ts')
rad = lambda name: name.suffix.endswith('.Rs')
vert = lambda name: name.suffix.endswith('.Zs')

preprocess = lambda d: down_sampl(d / np.max(d))
root = Path('/home/kensuke/workspace/ml/event_classification/case1')
events = event_folders(root)
ids = [event.name for event in events]
id_label = LabelEncoder()
id_label.fit(ids)

id_num = len(events)

all_files = [file for event in events for file in get_files(event, transv)]
print('Number of traces:', len(all_files))
labels = get_labels(all_files)
data = read_dat(all_files, preprocess)
print('Number of datapoints in each trace:', len(data[0]))
print('Number of labels:', id_num)
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, train_size=0.8)

print('Number of train data:', len(train_data))
print('Number of test data:', len(test_data))
# labels = to_categorical(labels, 5)
print('Shape of train_data:', train_data.shape)
print('Shape of train_labels:', train_labels.shape)
categ = to_categorical(labels, id_num)
print('Shape of categorical_labels:', categ.shape)
print(labels.shape)
# exit()
model = keras.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=(len(data[0]),)),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(5, activation='softmax')
])
model.summary()
print(labels.shape)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              # loss='categorical_crossentropy',
              metrics=['accuracy'])
# hist = model.fit(data, categ, epochs=5, verbose=1)
hist = model.fit(data, labels, epochs=50, verbose=1, validation_data=(test_data, test_labels))
test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
print('Test accuracy:', test_acc)

predictions = model.predict(test_data)
print(hist.history.keys())
accuracy = hist.history['accuracy']

plt.plot(range(1, len(accuracy) + 1), hist.history['accuracy'], label='training')
plt.plot(range(1, len(accuracy) + 1), hist.history['val_accuracy'], label='test')
plt.show()

# print(model.predict([data[0]]))

# data = read_sac(list(p.glob('*'))[0])
#
# plt.plot(data)
# plt.show()
