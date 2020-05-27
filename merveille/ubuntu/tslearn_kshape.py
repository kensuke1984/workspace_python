from tslearn.utils import to_time_series_dataset
from tslearn.clustering import KShape
import numpy as np
from matplotlib import pyplot as plt
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

seed = 0
np.random.seed(seed)
x_train, y_train, x_test, y_test = CachedDatasets().load_dataset('Trace')

x_train = x_train[y_train < 4]
x_train = x_train[:50]
np.random.shuffle(x_train)
print(x_train.shape, y_train.shape)

# for i in x_train:
#     plt.plot(i)
# plt.show()
# exit()


x_train = TimeSeriesScalerMeanVariance().fit_transform(x_train)
sz = x_train.shape[1]

ks = KShape(n_clusters=3, verbose=True, random_state=seed)
y_pred = ks.fit_predict(x_train)

plt.figure()
for yi in range(3):
    plt.subplot(3, 1, 1 + yi)
    for xx in x_train[y_pred == yi]:
        plt.plot(xx.ravel(), 'k-')
    plt.plot(ks.cluster_centers_[yi].ravel(), 'r-')
    plt.xlim(0, sz)
    plt.ylim(-4, 4)
    plt.title('cluster %d' % (yi + 1))

plt.tight_layout()
plt.show()
