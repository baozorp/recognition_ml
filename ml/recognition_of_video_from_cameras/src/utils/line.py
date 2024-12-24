import datetime
import ntpath
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from tqdm import tqdm

sys.path.append("ml/recognition_of_video_from_cameras/src")

from utils import multy_insert, execute_sql
from config import SQLConfig

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])


class LineHandler:
    def __init__(self,
                 source_weights_path: str,
                 source_video_path: str,
                 target_video_path: str = None,
                 confidence_threshold: float = 0.3,
                 iou_threshold: float = 0.7,
                 lines: dict = None,
                 show_video: bool = True,
                 save_db: bool = True,
                 is_streaming: bool = True,
                 procces_video_id: int = None,
                 session_id: str = None,
                 ) -> None:
        self.model = YOLO(source_weights_path)
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.show_video = show_video
        self.save_db = save_db
        self.is_streaming = is_streaming
        self.session_id = session_id
        self.frame = None  # Переменная для тарнспортивроки фреймов в stream
        self.crossed_count_in = 0  # Счетчик in
        self.crossed_count_out = 0  # Счетчик out
        self.data_trace = []  # Данные о трассировках
        self.data_lines = []  # Данные о линиях
        self.line_counter = []  # Переменная для хранения створ
        self.line_annotator = []  # Переменная для объектов отображения створ
        self.trigger = sv.ByteTrack()  # Объект-триггер пересечения детектированных объектьов со створами
        self.trace_annotator = sv.TraceAnnotator(  # Объект для отображения трассировок
            color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2
        )
        self.lines = lines
        self.lines_info = []

        for line in lines:
            # Инициализируем створы
            point_start = sv.Point(line['points'][0][0], line['points'][0][1])
            point_end = sv.Point(line['points'][1][0], line['points'][1][1])
            self.line_counter.append(
                [sv.LineZone(start=point_start, end=point_end, triggering_anchors=(sv.Position.CENTER,)),
                 # Объект LineZone
                 [0, 1, 2, 3, 5, 7],  # Детектируемые классы
                 sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5),  # Аннотатор линии
                 sv.ByteTrack(),  # Трекер линии
                 line['name']])  # Наименование линии

        # Создания объекта, отображающего боксы
        self.box_annotator = sv.BoxAnnotator(color=COLORS)

        # Создания объекта, отображающего ярлыки на объекты
        self.label_annotator = sv.LabelAnnotator(
            color=COLORS, text_color=sv.Color.BLACK
        )
        # Объект, содержащий информацию о видео
        self.video_info = sv.VideoInfo.from_video_path(self.source_video_path)

        # определение id обработки видео
        self.procces_video_id = procces_video_id

    def send_frame(self, frame):
        '''
        Передает файлы в self.frame для дальнейшей передачи в Response
        '''
        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            self.frame = jpeg.tobytes()

    def get_frame(self):
        '''
        Генератор поочередного отображения фреймов в stream
        '''
        while True:
            if self.frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + self.frame + b'\r\n')
            time.sleep(0.01)

    def stop_stream(self):
        '''
        Остановка потока stream и обработки видео
        '''
        self.is_streaming = False  # Остановка потока

    def get_procces_video_id(self):
        return self.procces_video_id

    def process_video(self):
        '''
        Функция обработки видео с учетом:
        self.show_video         -> наблюдать процесс обработки видео / не наблюдать
        self.target_video_path  -> сохранять конечное обработанное видео / не сохранять
        self.is_streaming       -> опция остановки процесса видео
        '''

        # Инициализация генератора фреймов
        generator = sv.get_video_frames_generator(self.source_video_path)
        frames_batch = []
        try:
            # Запись о старте
            execute_sql(f"insert into {SQLConfig.SQL_SCHEME}.{SQLConfig.TASKS} values"
                        f"('{self.session_id}', {self.procces_video_id}, '{ntpath.basename(self.source_video_path)}',"
                        f" 0, {self.video_info.total_frames}, 'start', "
                        f"'{datetime.datetime.now()}', '{datetime.datetime.now()}', '{datetime.date.today()}')",
                        params={},
                        is_select=False,
                        message='Write status')

            if self.target_video_path:
                with sv.VideoSink(self.target_video_path, self.video_info) as sink:
                    for i, frame in enumerate(tqdm(generator, total=self.video_info.total_frames), start=1):
                        frames_batch.append((i, frame))  # Сохраняем номер фрейма и сам фрейм
                        # Если достигнут размер пакета, обрабатываем его
                        if len(frames_batch) == 10:
                            frame_numbers = [item[0] for item in frames_batch]  # Извлекаем номера фреймов
                            annotated_frames = self.process_batch([item[1] for item in frames_batch], frame_numbers)
                            for annotated_frame in annotated_frames:
                                if self.show_video:
                                    self.send_frame(annotated_frame)
                                sink.write_frame(annotated_frame)
                            # Очищаем пакет для следующей обработки
                            frames_batch = []
                        if self.is_streaming:
                            continue
                        else:
                            break

                # Обрабатываем оставшиеся кадры, если есть
                if frames_batch:
                    frame_numbers = [item[0] for item in frames_batch]  # Извлекаем номера фреймов
                    annotated_frames = self.process_batch([item[1] for item in frames_batch], frame_numbers)
                    for annotated_frame in annotated_frames:
                        if self.show_video:
                            self.send_frame(annotated_frame)
                        sink.write_frame(annotated_frame)
                    frame_numbers = self.video_info.total_frames
                    execute_sql(f"update {SQLConfig.SQL_SCHEME}.{SQLConfig.TASKS} set status = 'done', "
                                f"current_frame = {frame_numbers}, end_process_dttm = '{datetime.datetime.now()}'"
                                f"where task_id = {self.procces_video_id}",
                                params={},
                                is_select=False,
                                message='Write status')

            else:
                for i, frame in enumerate(tqdm(generator, total=self.video_info.total_frames), start=1):
                    frames_batch.append((i, frame))  # Сохраняем номер фрейма и сам фрейм
                    # Если достигнут размер пакета, обрабатываем его
                    if len(frames_batch) == 10:
                        frame_numbers = [item[0] for item in frames_batch]  # Извлекаем номера фреймов
                        annotated_frames = self.process_batch([item[1] for item in frames_batch], frame_numbers)
                        for annotated_frame in annotated_frames:
                            if self.show_video:
                                self.send_frame(annotated_frame)
                        # Очищаем пакет для следующей обработки
                        frames_batch = []
                    if self.is_streaming:
                        continue
                    else:
                        break

                if frames_batch:
                    frame_numbers = [item[0] for item in frames_batch]  # Извлекаем номера фреймов
                    annotated_frames = self.process_batch([item[1] for item in frames_batch], frame_numbers)
                    for annotated_frame in annotated_frames:
                        if self.show_video:
                            self.send_frame(annotated_frame)
                    frame_numbers = self.video_info.total_frames
                    execute_sql(f"update {SQLConfig.SQL_SCHEME}.{SQLConfig.TASKS} set status = 'done', "
                                f"current_frame = {frame_numbers}, end_process_dttm = '{datetime.datetime.now()}'"
                                f"where task_id = {self.procces_video_id}",
                                params={},
                                is_select=False,
                                message='Write status')

        except Exception as e:
            print('LineHandler.process_video ->', e)

        finally:
            if self.save_db is True:
                # задание линий
                for line in self.lines:
                    self.lines_info.append((
                        int(self.procces_video_id),
                        str(line['name']),
                        str(line['points']),
                        ntpath.basename(self.source_video_path),  # file_name
                        datetime.datetime.now(),  # process_dttm
                        datetime.date.today()  # day
                    ))

                # определение id записи об объектах
                for i, trace in enumerate(self.data_trace):
                    self.data_trace[i] = (self.procces_video_id,) + trace

                # определение procces_video_id в записи о линиях
                for i, line in enumerate(self.data_lines):
                    self.data_lines[i] = (self.procces_video_id,) + line

                # Запись объектов детекции
                multy_insert(self.data_trace,
                             ["task_id", "x_min", "y_min", "x_max", "y_max", "x_center", "y_center", "class_id",
                              "confidence",
                              "tracker_id", "class_name", "current_frame", "total_frames", "is_polygon", "file_name",
                              "process_dttm", "day"],
                             SQLConfig.SQL_SCHEME,
                             SQLConfig.TRAFFIC,
                             message='Connection completed -> TRAFFIC')

                # Запись линий
                multy_insert(self.data_lines,
                             ["task_id", "name", "vector_start", "vector_end", "count_in", "count_out",
                              "file_name", "process_dttm", "day"],
                             SQLConfig.SQL_SCHEME,
                             SQLConfig.TRAFFIC_COUNT_LINE,
                             message='Connection completed -> TRAFFIC_COUNT_LINE')

                # записи информации о самих линиях
                multy_insert(self.lines_info,
                             ["task_id", "name", "line", "file_name", "process_dttm", "day"],
                             SQLConfig.SQL_SCHEME,
                             SQLConfig.TRAFFIC_LINES,
                             message='Connection completed -> TRAFFIC_LINES')

                execute_sql(f"update {SQLConfig.SQL_SCHEME}.{SQLConfig.TASKS} set status = 'done', "
                            f"end_process_dttm = '{datetime.datetime.now()}'"
                            f"where task_id = {self.procces_video_id}",
                            params={},
                            is_select=False,
                            message='Write status')

    def process_batch(self, frames: list, frame_numbers: list) -> list:
        results = self.model(frames, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold)
        annotated_frames = []
        for frame, result, frame_number in zip(frames, results, frame_numbers):
            triggering_anchors = [sv.Position.CENTER]
            detections = sv.Detections.from_ultralytics(result)

            # Определение уникальных классов для створы
            class_id = []
            for line in self.line_counter:
                class_id += line[1]
            class_id = np.unique(class_id)

            # Отображение трассировки
            detections = detections[(np.isin(detections.class_id, class_id))]
            detections_tracker = self.trigger.update_with_detections(detections)
            self.trace_annotator.annotate(frame, detections_tracker)

            for i in range(len(detections_tracker.xyxy)):
                # Массив данных для выгрузки в базу
                self.data_trace.append((float(detections_tracker.xyxy[i][0]),  # x_min
                                        float(detections_tracker.xyxy[i][1]),  # y_min
                                        float(detections_tracker.xyxy[i][2]),  # x_max
                                        float(detections_tracker.xyxy[i][3]),  # y_max
                                        (float(detections_tracker.xyxy[i][0]) + float(  # x_center
                                            detections_tracker.xyxy[i][2])) / 2,
                                        (float(detections_tracker.xyxy[i][1]) + float(  # y_center
                                            detections_tracker.xyxy[i][3])) / 2,
                                        int(detections_tracker.class_id[i]),  # class_id
                                        float(detections_tracker.confidence[i]),  # confidence
                                        int(detections_tracker.tracker_id[i]),  # tracker_id
                                        str(detections_tracker.data['class_name'][i]),  # class_name
                                        int(frame_number),  # frame_number
                                        int(self.video_info.total_frames),  # total frames
                                        False,  # is_polygon
                                        ntpath.basename(self.source_video_path),  # file_name
                                        datetime.datetime.now(),  # process_dttm
                                        datetime.date.today()  # day
                                        ))
            for line in self.line_counter:
                detection = detections[(np.isin(detections.class_id, np.unique(line[1])))]

                # Обновление триггера на детекциях и обновление счетчиков вхождения
                detection = line[3].update_with_detections(detection)
                line[0].triggering_anchors = triggering_anchors
                crossed_in, crossed_out = line[0].trigger(detection)

                if np.any(crossed_in):
                    self.crossed_count_in += np.sum(crossed_in)

                if np.any(crossed_out):
                    self.crossed_count_out += np.sum(crossed_out)

                if (np.any(crossed_in) or np.any(crossed_out)) and self.save_db is True:
                    # Массив данных для выгрузки в базу
                    self.data_lines.append((
                        str(line[4]),  # Наименование створы
                        str(line[0].vector.start),  # стартовая точка
                        str(line[0].vector.end),  # конечная точка
                        int(line[0].in_count),  # сколько вошло
                        int(line[0].out_count),  # сколько вышло
                        ntpath.basename(self.source_video_path),  # наименование видео
                        datetime.datetime.now(),  # время timestamp
                        datetime.date.today())  # время date
                    )

                # Отображения трассировки
                frame = self.trace_annotator.annotate(frame, detection)

                # Отображение ярлыков
                label = [f"{tracker_id} {dtype['class_name']} {confidence:0.2f}"
                         for xyxy, mask, confidence, class_id, tracker_id, dtype
                         in detection]
                frame = self.label_annotator.annotate(
                    frame, detection, label
                )

                # Отображение боксов
                frame = self.box_annotator.annotate(scene=frame,
                                                    detections=detection)
                frame = line[2].annotate(frame, line[0])

                # Отображения подсчета на створах
            if frame_number % 100 == 0 or frame_number == self.video_info.total_frames:
                execute_sql(f"update {SQLConfig.SQL_SCHEME}.{SQLConfig.TASKS} set status = 'run', "
                            f"current_frame = {frame_number}, end_process_dttm = '{datetime.datetime.now()}'"
                            f"where task_id = {self.procces_video_id}",
                            params={},
                            is_select=False)
            cv2.putText(frame, f"Crossed In: {self.crossed_count_in}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            cv2.putText(frame, f"Crossed Out: {self.crossed_count_out}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            annotated_frames.append(frame)
        return annotated_frames
