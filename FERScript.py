import cv2
import mediapipe as mp
from FERModel import FERModel
from pathlib import Path
import time

camera = cv2.VideoCapture(0,cv2.CAP_DSHOW) #объект захвата с камеры
fps = 30 # Максимальное количество кадров в секунду
camera.set(cv2.CAP_PROP_FPS, fps)

# Цикл с выбором вывода модели
while True:
    model_type = input('Enter output type (class or va): ')

    # Цикл с проверкой правильности ввода
    if model_type == 'class' or 'va':
        break
    else:
        print ('TypeModelError')
        continue

# Создание детектора лица из фреймворка MediaPipe
faceDetector = mp.solutions.face_detection

# Путь до предобученных весов
weight_path = Path(f'weights\{model_type}_weights.h5')
# Создание модели с предобученными весами
model = FERModel(path_model_weights=weight_path, type_model=model_type)

if __name__ == '__main__':
    # Менеджер контекста для детектора лиц и его конфигурация
    with faceDetector.FaceDetection(
        model_selection=0, min_detection_confidence=0.7) as face_detection:
        while (camera.isOpened()):
            start = time.time() # Время старта цикла для вычисления обработки кадров в секунду
            frameFlag, frame = camera.read()
            # Поиск лица на кадре
            faces = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Если лицо не обнаружено основной цикл начинается заново
            if not faces.detections:
                frame = cv2.putText(frame, 'Face don`t recognition', (5, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1)
                cv2.imshow('camera', frame)
                if cv2.waitKey(33) == ord('q'):
                    break
                continue
            try:
                # Цикл обработки кадра и предсказания эмоции
                for id, detection in enumerate(faces.detections):
                    # Координаты и размеры бокса с лицом на кадре
                    bbox = detection.location_data.relative_bounding_box
                    mul_h = frame.shape[0]
                    mul_w = frame.shape[1]
                    x = int(abs(bbox.xmin) * mul_w)
                    y = int(abs(bbox.ymin) * mul_h)
                    w = int(abs(bbox.width) * mul_w)
                    h = int(abs(bbox.height) * mul_h)
                    # Обрезка лица из кадра
                    frame_pred = frame[y:y+h, x:x+w, :]
                    # Обозначение оптимального расстояния до вебкамеры (размера бокса с лицом)
                    if h > 180 and h < 230:
                        color_bbox = (0, 255, 0)
                    else:
                        color_bbox = (0, 0, 255)
                    # Получение номера эмоции или координат и названия эмоции
                    emotion_number, emotion_name = model.get_emotion(frame_pred)
                    # Добавление прямоугольника вокруг лица на кадре
                    frame = cv2.rectangle(frame, (x, y), (x+w, y+h), color_bbox, 2)
                    # Добавление текста с предсказанной эмоцией моделью
                    frame = cv2.putText(frame, f'{emotion_number}: {emotion_name}', (x, y-15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                totalTime = time.time() - start
                # Вычисление количества обработанных кадров в секунду
                fps = 1 / totalTime 
                frame = cv2.putText(
                    frame, f'FPS: {int(fps)}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.imshow('camera', frame) # Вывод полученного предобработанного кадра

                # Выход из основного цикла при нажатии 'q'
                if cv2.waitKey(33) == ord('q'):
                    break
            except:
                camera.release() # Закрытие объекта захвата кадров с вебкамеры
                break

cv2.destroyAllWindows() # Закрытие окна с выводом

