#importing necessary library
import os

import numpy as np
import tensorflow as tf
import keras
from keras import layers
import pandas as pd
from keras.src.metrics.accuracy_metrics import accuracy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#initializaion of encoder to map text => work numbers
encoder = LabelEncoder()

#loading the dataset
dataset = pd.read_csv('diabetes_prediction_dataset.csv')
#printing the sample of data set
print('--data sample--')
print(dataset.head().to_string())

#prinitng the share of dataset
print('--data shape--')
print(dataset.shape)

#printing the columns in dataset
print('--data columns--')
print(dataset.columns)

#counting the unique vales per column
print('--data unique count--')
print(dataset.nunique(axis=0))

columns = dataset.columns
for column in columns:
    print('Unique in column ', column)
    print(dataset[column].unique())

#get the information of data set
print(dataset.info)

#get the data description
print(dataset.describe().to_string())

#mapping the text data to numbers
dataset['smoking_history_num'] = encoder.fit_transform(dataset['smoking_history'])

print("-----encoded----")
print(dataset['smoking_history_num'].unique())

#remove the column with no use currently
feature_set = dataset.drop(['diabetes', 'smoking_history', 'gender'], axis=1)

#print the sample of feature set
print('--data Features--')
print(feature_set.head().to_string())
print(feature_set.shape)

#assing the label set
label_set = dataset['diabetes']
print('--data Features--')
print(label_set.head().to_string())
print(label_set.shape)

#spliting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(feature_set, label_set,
                                                    test_size=0.4, random_state=42,
                                                    shuffle=False)
x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test,
                                                              test_size=0.8,
                                                              shuffle=True)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_validation.shape)
print(y_validation.shape)


# defining the deep model
Diabetes_model = keras.Sequential([
    keras.Input(shape=[7]),
    layers.Dense(units=32, activation='relu'),
    layers.Dense(units=64, activation='relu'),
    layers.Dense(units=32, activation='relu'),
    layers.Dense(units=1, activation='sigmoid')
])

Diabetes_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss= keras.losses.binary_crossentropy,
    metrics=['accuracy']
)

Diabetes_model.summary()
Diabetes_model.fit(x=x_train,
                y=y_train,
                batch_size=32,
                epochs=10,
                validation_data=(x_validation,
                                 y_validation))

eval_loss, eval_accuracy = Diabetes_model.evaluate(x_test, y_test, verbose=1)

evaluation = pd.DataFrame(
    {'accuracy': [eval_accuracy,],
     'Loss': [eval_loss, ]
     }
)

evaluation = evaluation.to_csv('Evaluation Metrix.csv', index=False)
print(evaluation)

save_dir = 'C:/Users/GB-YL-18/Desktop/Machine Learning/Diabetes Prediction/Model'
os.makedirs(save_dir, exist_ok=True)

# Save the model
Diabetes_model.save(os.path.join(save_dir, 'Diabetes_prediction.keras'))
Diabetes_model.save('C:/Users/GB-YL-18/Desktop/Machine Learning/Diabetes Prediction/Model/Diabetes_prediction.keras')