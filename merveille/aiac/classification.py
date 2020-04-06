from sklearn import datasets, svm, metrics
import numpy as np
import matplotlib.pyplot as plt

digits = datasets.load_digits()

print(dir(digits))
print(digits.data)
print(digits.data.shape)
print(digits.target.shape)
img = np.reshape(digits.data[0], (8, 8))
plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
plt.axis('off')

images_and_labels = list(zip(digits.images, digits.target))

for index, (image, label) in enumerate(images_and_labels[:10]):
    plt.subplot(2, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)
num = len(digits.data)
training_num = int(num * 2 / 3)
print('training num=' + str(training_num))
print('training num type' + str(type(training_num)))

train_data = digits.data[:training_num]
train_target = digits.target[:training_num]
test_data = digits.data[training_num:]
test_target = digits.target[training_num:]

classifier = svm.SVC(gamma=0.001)

classifier.fit(train_data, train_target)

predicted = classifier.predict(test_data)

images_and_predictions = list(zip(digits.images[training_num:], predicted))

for index, (image, prediction) in enumerate(images_and_predictions[:10]):
    plt.subplot(2, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)
plt.show()

print(metrics.accuracy_score(test_target, predicted))

# plt.show()
