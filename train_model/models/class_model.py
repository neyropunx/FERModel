import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, BatchNormalization

class FER_class_model(tf.keras.Model):
    '''
        Для загрузки предобученных весов в метод __init__ 
        необходимо передать аргумент с путем до файла с весами в формате .h5
        path_to_weights = '/path/to/weights.h5'

        Входящее изображение должно быть размера 48х48 
        в оттенках серого и нормализованно [0..1]
        input_shape=(48,48,1)

        Выходящий тензор будет иметь вероятности для 9 классов
        output_shape=(9)
    ''' 
    def __init__(self, path_model_weights=None):
        super(FER_class_model, self).__init__()
        self.path_model_weights = path_model_weights
        self.model = self._build_model()
        if path_model_weights is not None:
            self.model.load_weights(path_model_weights)

        emotion_dict = {'anger': 0,
                        'contempt': 1,
                        'disgust': 2,
                        'fear': 3,
                        'happy': 4,
                        'neutral': 5,
                        'sad': 6,
                        'surprise': 7,
                        'uncertain': 8}
        self.emotion_dict = dict((v,k) for k, v in zip(emotion_dict.keys(), emotion_dict.values()))
        
    def _build_model(self):
        #import Tensorflow framework and required Layers
        model_input = tf.keras.layers.Input(shape=(48,48,1))
        #1st convolution layer
        model = Conv2D(64, (5, 5), input_shape=(48,48,1), activation='relu', padding='same')(model_input)
        model = Conv2D(64, (5, 5), activation='relu', padding='same')(model)
        model = BatchNormalization()(model)
        model = MaxPooling2D(pool_size=(2, 2))(model)
        model = Dropout(0.5)(model)

        #2nd convolution layer
        model = Conv2D(128, (5, 5),activation='relu',padding='same')(model)
        model = Conv2D(128, (5, 5),activation='relu',padding='same')(model)
        model = BatchNormalization()(model)
        model = MaxPooling2D(pool_size=(2, 2))(model)
        model = Dropout(0.5)(model)

        #3rd convolution layer
        model = Conv2D(256, (3, 3),activation='relu',padding='same')(model)
        model = Conv2D(256, (3, 3),activation='relu',padding='same')(model)
        model = BatchNormalization()(model)
        model = MaxPooling2D(pool_size=(2, 2))(model)
        model = Dropout(0.5)(model)
        model = Flatten()(model)
        
        model = Dense(128)(model)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = Dropout(0.2)(model)
        model = Dense(9)(model)
        model_output = Activation('softmax')(model)

        model = tf.keras.Model(inputs=model_input, outputs=model_output, name="FER_model")

        model.summary()

        return model

    def call(self, inputs):
        return self.model(inputs)

    def get_weights(self):
        print(self.model.get_weights())
    
    def get_emotion(self, img):
        prediction = np.argmax(self.model.predict(img[None, ...], verbose=0))
        emotion_name = str(self.emotion_dict[prediction])
        return emotion_name, prediction