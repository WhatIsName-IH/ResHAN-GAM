import contextlib
import os
import random
import time
import warnings
from tensorflow.keras.layers import Input, Reshape
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Flatten, Conv1D, LeakyReLU, BatchNormalization, Dropout, Activation, GlobalAveragePooling1D, \
    GlobalMaxPooling1D
from keras.regularizers import l2
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input

class GAM_Attention(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_chammels, rate=4):
        super(GAM_Attention, self).__init__()
        self.channel_attention = tf.keras.Sequential([
            tf.keras.layers.Dense(in_channels // rate, activation=None),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.Dense(in_channels)
        ])
        self.spatial_attention = tf.keras.Sequential([
            tf.keras.layers.Dense(in_channels // rate, activation=None),
            tf.keras.layers.LeakyReLU(alpha=0.01),
            tf.keras.layers.Dense(out_chammels, activation='linear')
        ])
    def call(self, x):
        x_att = self.channel_attention(x)
        x = x * x_att
        x_spatial_att = self.spatial_attention(x)
        out = x * x_spatial_att

        return out
class ResidualBlock1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, activation='linear'):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = Conv1D(filters, kernel_size, strides=strides, padding='same', activation=activation,kernel_regularizer=l2(0.01),
                kernel_initializer=tf.keras.initializers.HeNormal())
        self.conv2 = Conv1D(filters, kernel_size, strides=1, padding='same', activation=None, kernel_regularizer=l2(0.01),
                kernel_initializer=tf.keras.initializers.HeNormal())
        self.batch_norm = BatchNormalization()
        self.identity_mapping = Conv1D(filters, kernel_size=1, strides=strides, padding='valid', activation=None, kernel_regularizer=l2(0.01),
                kernel_initializer=tf.keras.initializers.HeNormal())

    def call(self, x):
        identity = x
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.batch_norm(x)
        identity = self.identity_mapping(identity)
        x += identity
        x = tf.nn.relu(x)
        return x


def create_HAN_model(gru_size, class_num, sentence_num, sentence_length):
    
    dr = 0.3
    inputs = tf.keras.Input(shape=(sentence_length, sentence_num, 1))
    
    x = tf.reshape(inputs, [-1, sentence_length, 1])
    
    word_gru = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(gru_size, return_sequences=True))
    
    word_attention_layer = GAM_Attention(gru_size * 2, gru_size * 2)
    
    for _ in range(3):  
        x1 = ResidualBlock1D(gru_size * 2, 3)(x)
    word_encoded = word_gru(x1)
    word_encoded = Dropout(dr)(word_encoded)
    word_attention = word_attention_layer(word_encoded)
    word_attention_sum = tf.reduce_sum(word_attention, axis=1)
    
    word_attention_reshape = tf.reshape(word_attention_sum, [-1, sentence_num, word_encoded.shape[-1]])
    

    sentence_gru = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(gru_size, return_sequences=True))

    sentence_attention_layer = GAM_Attention(gru_size * 2, gru_size * 2)
    sentence_encoded = sentence_gru(word_attention_reshape)
    sentence_attention = sentence_attention_layer(sentence_encoded)
    sentence_attention = Flatten()(sentence_attention)
    
    sentence_attention = LeakyReLU(alpha=0.01)(sentence_attention)
    sentence_attention = Dropout(dr)(sentence_attention)
    output = tf.keras.layers.Dense(class_num, activation=None, kernel_regularizer=l2(0.01))(sentence_attention)
    
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    file_name = r"data/LUCAS_ok_fractional_diff.csv"
    print("读取数据完成！")
    df = pd.read_csv(file_name)
    data_name = file_name.split("/")[-1][:-4]
    df = df.sample(frac=1).reset_index(drop=True)
    y = df["OC"]  # 标签
    X = df.drop('OC', axis=1)  # 特征
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    sentence_length = 20
    X_train = X_train.values.reshape((13325, sentence_length, -1, 1))
    X_test = X_test.values.reshape((5711, sentence_length, -1, 1))
    sentence_num = 100
    gru_size = 50
    class_num = 1
    model = create_HAN_model(gru_size, class_num, sentence_num, sentence_length)
    input_shape = (None, X_train.shape[1], X_train.shape[2], 1)
    epochs = 5000
    early_stopping = EarlyStopping(monitor='val_loss', patience=200, restore_best_weights=True)
    optimizer = tf.optimizers.Adam(lr=0.001, epsilon=1e-7, amsgrad=False)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(delta=1.0), metrics=['mse', 'mae'])
    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=1, batch_size=1024, validation_data=(X_test, y_test), verbose=1,
                        callbacks=[early_stopping])
    end_time = time.time()
    elapsed_time_seconds = end_time - start_time
    elapsed_time_minutes = elapsed_time_seconds / 60
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    r2_score_train = r2_score(y_train, y_pred_train)
    r2_score_test = r2_score(y_test, y_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    rpd_test = np.std(y_test) / rmse_test
    rpd_train = np.std(y_train) / rmse_train
    df_train = pd.DataFrame(
        {'Actual_Train': np.array(y_train).flatten(), 'Predicted_Train': np.array(y_pred_train).flatten()})
    df_test = pd.DataFrame({'Actual_Test': np.array(y_test).flatten(), 'Predicted_Test': np.array(y_pred).flatten()})

    evaluation_metrics = {
        'Metric': ['Train R2', 'Test R2', 'Train RMSE', 'Test RMSE', 'RPD Train', 'RPD Test'],
        'Value': [r2_score_train, r2_score_test, rmse_train, rmse_test, rpd_train, rpd_test]
    }
    df_metrics = pd.DataFrame(evaluation_metrics)
    print("************" + data_name + "结果展示" + "************")
    print('Train R2: %.3f' % r2_score_train)
    print('Test R2: %.3f' % r2_score_test)
    print('Train RMSE: %.3f' % rmse_train)
    print('Test RMSE: %.3f' % rmse_test)
    print('RPD Train: %.3f' % rpd_train)
    print('RPD Test: %.3f' % rpd_test)
    print("\n")
