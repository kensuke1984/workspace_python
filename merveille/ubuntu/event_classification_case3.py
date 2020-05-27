from pathlib import Path
from matplotlib import pyplot as plt
from readsac import normalize
from ubuntu.western_pacific_data import prem_root
from util import event_folders, SACFileName
import numpy as np
from sklearn.preprocessing import LabelEncoder
from readsac import read_sac
from sklearn.model_selection import train_test_split
from tslearn.clustering import KShape


def create_encoder(sacfiles):
    encoder = LabelEncoder()
    ids = list(set(sacfile.event_id for sacfile in sacfiles))
    encoder.fit(ids)
    return encoder


def to_timeseries(sacfiles, preprocess=lambda d: d):
    return np.array([preprocess(read_sac(sacfile)) for sacfile in sacfiles])


def get_labels(sacfiles):
    return label_encoder.transform([id for file in sacfiles for id in label_encoder.classes_ if id in str(file)])


events = np.array(list(event_folders(prem_root)))
np.random.shuffle(events)
events = events[:2]

all_files = np.array([SACFileName(file) for event in events for file in event.glob('*Ts')])
used_files = all_files

print('Number of files:', len(used_files))
label_encoder = create_encoder(used_files)
print('Labels:', *label_encoder.classes_)
num_class = len(label_encoder.classes_)
print('Number of events:', num_class)
preprocess = lambda d: (normalize(d)[::20])[900:1250]

x = to_timeseries(used_files, preprocess)
y = get_labels(used_files)
num_one_data = len(x[0])
print('Number of datapoints in one file:', num_one_data)

x_train, x_test, y_train, y_test = train_test_split(x, y)
print('Number of train files:', len(x_train))
print('Number of test files:', len(x_test))
seed = 0
kshape = KShape(n_clusters=num_class, verbose=True, random_state=seed)

for i, j in zip(x_train, y_train):
    plt.subplot(3, 1, 1 + j)
    plt.plot(i)

plt.show()

# exit()
y_pred = kshape.fit_predict(x_train)

plt.figure()
for yi in range(num_class):
    plt.subplot(num_class, 1, 1 + yi)
    for xx in x_train[y_pred == yi]:
        plt.plot(xx.ravel(), 'k-')
    plt.plot(kshape.cluster_centers_[yi].ravel(), 'r-')
    plt.xlim(0, len(x[0]))
    plt.ylim(-4, 4)
    plt.title('cluster %d' % (yi + 1))

plt.tight_layout()
plt.show()
print(y_train)
print(y_pred)

# print(x[0])
# plt.plot(x[0])
# plt.show()
