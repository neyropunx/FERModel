from scipy import spatial
import cv2
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, BatchNormalization


class FERModel(tf.keras.Model):
    '''
        Для загрузки весов модели необходимо в аргументе path_model_weights метода __init__ передать путь до весов
        Фаил весов должен быть в формате .h5
        path_to_weights = '/path/to/weights.h5'

        class модель:
            model = FER_model(path_model_weights='/path/to/class_weights.h5', type_model='class')
            выходом будет вектор с вероятностями 9 классов (эмоций)
            output_shape=(9)
            get_emotion() получает на вход изображение и выводит номер эмоции
            и название эмоции с самой высокой вероятностью
            emotion_number, emotion_name = FER_model.get_emotion(img)
            
        va model:
            model = FER_model(path_model_weights='/path/to/va_weights.h5', type_model='va')
            выходом будет вектор с координатами в системе valence-arousal
            output_shape=(2) ([-1..1],[-1..1])
            get_emotion() получает на вход изображение и выводит координаты
            и название эмоции исходя из предсказзаных координат
            prediction, emotion_name = FER_model.get_emotion(img)
    '''

    def __init__(self, path_model_weights='weights/class_weights.h5', type_model='class'):
        super(FERModel, self).__init__()
        self.type_model = type_model
        self.emotions_coord = {'happy': [.9, .2],
                               'surprise': [.4, .9],
                               'neutral': [0, 0],
                               'anger': [-.4, .8],
                               'contempt': [-.5, .25],
                               'disgust': [-.7, .5],
                               'fear': [-.1, .8],
                               'sad': [-.8, -0.4],
                               'uncertain': [0, -.3]}
        self.emotions_dict = {'anger': 0,
                              'contempt': 1,
                              'disgust': 2,
                              'fear': 3,
                              'happy': 4,
                              'neutral': 5,
                              'sad': 6,
                              'surprise': 7,
                              'uncertain': 8}
        self.model = self._build_model()
        self.model.load_weights(path_model_weights)

    def _build_model(self):
        # Входящий слой
        base_model_input = tf.keras.layers.Input(shape=(48, 48, 1))
        # Первый сверточный слой
        base_model = Conv2D(64, (5, 5), input_shape=(48, 48, 1), activation='relu',
                            padding='same', kernel_initializer='zeros')(base_model_input)
        base_model = Conv2D(64, (5, 5), activation='relu',
                            padding='same', kernel_initializer='zeros')(base_model)
        base_model = BatchNormalization()(base_model)
        base_model = MaxPooling2D(pool_size=(2, 2))(base_model)
        base_model = Dropout(0.5)(base_model) # Необходим для предотвращения прееобучения модели

        # Второй сверточный слой
        base_model = Conv2D(128, (5, 5), activation='relu',
                            padding='same', kernel_initializer='zeros')(base_model)
        base_model = Conv2D(128, (5, 5), activation='relu',
                            padding='same', kernel_initializer='zeros')(base_model)
        base_model = BatchNormalization()(base_model)
        base_model = MaxPooling2D(pool_size=(2, 2))(base_model)
        base_model = Dropout(0.5)(base_model)

        # Третий сверточный слой
        base_model = Conv2D(256, (3, 3), activation='relu',
                            padding='same', kernel_initializer='zeros')(base_model)
        base_model = Conv2D(256, (3, 3), activation='relu',
                            padding='same', kernel_initializer='zeros')(base_model)
        base_model = BatchNormalization()(base_model)
        base_model = MaxPooling2D(pool_size=(2, 2))(base_model)
        base_model = Dropout(0.5)(base_model)
        base_model_output = Flatten()(base_model) # Векторизация полученной карты признаков

        # Выбор вывода модели
        if self.type_model == 'class':
            model = Dense(128, kernel_initializer='zeros')(base_model_output)
            model = BatchNormalization()(model)
            model = Activation('relu')(model)
            model = Dropout(0.2)(model)
            model = Dense(9, kernel_initializer='zeros')(model)
            model_output = Activation('softmax')(model)
        else:
            model = Dense(128)(base_model_output)
            model = BatchNormalization()(model)
            model = Activation('relu')(model)
            model = Dropout(0.2)(model)
            model_output = Dense(2)(model)
            # KDTree для нахождения ближайшей эмоции по координатам
            self.emotions_KDTree = spatial.KDTree(
                list(self.emotions_coord.values()))

        # Создание модели
        model = tf.keras.Model(inputs=base_model_input,
                               outputs=model_output, name="FERModel")
        model.summary()

        return model

    def call(self, inputs):
        return self.model(inputs)

    def get_emotion(self, img):
        # Предобработка изображения
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Получение изображения в оттенках серого
        img = img / 255. # Нормализация размера пикселя [0..1]  
        img = cv2.resize(img, (48, 48)) # Изменение размера изображения
        img = np.expand_dims(img, 0) # Добавления размера батча

        # Выполнение предсказания по изображению и получение номера (координат) и названия эмоции
        if self.type_model == 'class':
            emotions_dict = dict((v, k) for k, v in zip(
                self.emotions_dict.keys(), self.emotions_dict.values()))
            emotion_number = np.argmax(self.model.predict(img, verbose=0))
            emotion_name = str(emotions_dict[emotion_number])
            return emotion_number, emotion_name
        else:
            prediction = self.model.predict(img, verbose=0)
            emotion_number = int(self.emotions_KDTree.query(prediction)[1])
            emotion_name = list(self.emotions_coord.keys())[emotion_number]
            return prediction, emotion_name