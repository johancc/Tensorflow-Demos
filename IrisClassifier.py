from __future__ import absolute_import, division, print_function

import os 
import matplotlib.pyplot as plt 
import tensorflow as tf 

tf.enable_eager_execution()

# getting the training data

train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"
train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

print("Local copy of the dataset file: {}".format(train_dataset_fp))

column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
feature_names = column_names[:-1]
label_name = column_names[-1]

class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
batch_size = 32

train_dataset = tf.contrib.data.make_csv_dataset(
                train_dataset_fp,
                batch_size,
                column_names = column_names,
                label_name = label_name,
                num_epochs = 1)

features, labels = next(iter(train_dataset))

plt.scatter(features['petal_length'],
            features['sepal_length'])

plt.xlabel("Petal length")
plt.ylabel("Sepal length")
plt.show()
