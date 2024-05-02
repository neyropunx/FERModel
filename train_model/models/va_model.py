from scipy import spatial

class FER_va_model(FER_class_model):
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
    def __init__(self, path_class_model_weights=None, path_va_model_weights=None):
        super(FER_class_model, self).__init__()
        self.class_model = self._build_model()

        if path_class_model_weights is not None:
            self.class_model.load_weights(path_class_model_weights)
        self.va_model = self._build_va_model()

        if path_va_model_weights is not None:
            self.va_model.load_weights(path_va_model_weights)

        self.emotions_coord = {'happy': [.9, .2],
                          'surprise': [.4, .9],
                          'neutral': [0, 0],
                          'anger': [-.4, .8],
                          'contempt': [-.5, .25],
                          'disgust': [-.7, .5],
                          'fear': [-.1, .8],
                          'sad': [-.8, -0.4],
                          'uncertain': [0, -.3]}
        self.emotions = spatial.KDTree(list(emotions_coord.values()))

    def _build_model(self):
        va_model = tf.keras.Model(self.class_model.input, self.class_model.layers[16].output)
        model = Dense(128)(va_model.output)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = Dropout(0.2)(model)
        model_output = Dense(2)(model)

        va_model = tf.keras.Model(va_model.input, model_output)

        va_model.summary()

        return va_model

    def call(self, inputs):
        return self.va_model(inputs)

    def get_weights(self):
        print(self.va_model.get_weights())
    
    def get_emotion(self, img):
        prediction = self.va_model.predict(img[None, ...], verbose=0)
        emotion_number = int(self.emotions.query(prediction)[1])
        emotion_name = list(self.emotions_coord.keys())[emotion_number]
        return emotion_name, prediction