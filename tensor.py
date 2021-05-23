import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import pickle
import minvc as vc
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
#                 'Acceleration', 'Model Year', 'Origin']

# raw_dataset = pd.read_csv("measurement1test.p", names=column_names,
#                           na_values='?', comment='\t',
#                           sep=' ', skipinitialspace=True)



filename = "measurement1_train.p"
with open(filename, 'rb') as f:
    measurements = pickle.load(f)
print(type(measurements))



beacons: vc.Beacon = pd.read_csv("beacons_proj.csv", header=0, index_col="name")
columns = beacons.index.values.tolist()
columns.append("x")
columns.append("y")
n_columns = len(columns)
n_output=2
print(columns)
nan_values=0

df:pd.DataFrame = None

data_list=[]

for meas in measurements:
    # print(meas)
    row = np.full(n_columns-n_output, nan_values).tolist()
    # print(row)
    for name, rssi in zip(meas.get_beacon_names(), meas.get_beacon_rssis()):

        index = int(name.replace("b", ""))
        row[index] = ((-rssi)-0)/(100-0)

    loc = meas.get_real_location()
    row.append(loc[0])
    row.append(loc[1])
    # print(row)
    # print(n_columns, len(row))
    data_list.append(row)

df = pd.DataFrame(data_list, columns=columns)
# print("df: {}".format(df))
    # temp_list = meas.get_beacon_rssis().tolist()
    # temp_list.append(row)

# train_labels = train_features.pop('MPG')
# test_labels = test_features.pop('MPG')
train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)

print(len(train_dataset))
print(len(test_dataset))

# sns.pairplot(train_dataset[columns], diag_kind='kde')

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_x_labels = train_features.pop('x')
train_y_labels = train_features.pop('y')

test_x_labels = test_features.pop('x')
test_y_labels = test_features.pop('y')

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(n_columns-n_output,)),
    tf.keras.layers.Dense(50),
    tf.keras.layers.Dense(8),
    tf.keras.layers.Dense(4),
    tf.keras.layers.Dense(2),
    tf.keras.layers.Dense(n_output)])

model.summary()

model.compile(
    # optimizer=tf.keras.optimizers.SGD(
    # learning_rate=0.01, momentum=0.01, nesterov=False, name='SGD'),
    optimizer=tf.optimizers.Adam(learning_rate=0.0001),
    # metrics=[ tf.keras.metrics.Accuracy()],
    loss='mean_squared_error')

earlystop_callback = tf.keras.callbacks.EarlyStopping(
     monitor='val_loss', min_delta=0.001, patience=3, mode='min', restore_best_weights=True)

print(type(train_features))
history = model.fit(
    train_features, [train_x_labels, train_y_labels], workers=8, use_multiprocessing=True,
    epochs=100,
    # suppress logging
    verbose=1,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2
    ,callbacks=[earlystop_callback]
)

# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch
# hist.tail()

loss = model.evaluate(test_features, [test_x_labels,test_y_labels], verbose=1)
print("Untrained model, accuracy: {:5.2f}%".format(100 * loss))

model.save('saved_model/my_model.h5')

model_new = tf.keras.models.load_model('saved_model/my_model.h5')

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 40])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)
  plt.show()


# plot_loss(history)
