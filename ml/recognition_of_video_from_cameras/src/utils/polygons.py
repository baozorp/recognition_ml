import datetime
import json
import ntpath
import os
import sys
import time
import traceback
from typing import Dict, Iterable, List, Set, Any
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv

sys.path.append("ml/recognition_of_video_from_cameras/src")
from utils import DetectionsManager
from utils import multy_insert, execute_sql
from config import SQLConfig

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])


class PolygonHandler:
    """
    Инициализирует PolygonHandler объект

    Параметры:
        source_weights_path     (str)   -> путь до весов модели
        source_video_path       (str)   -> путь до видео-источника
        target_video_path       (str)   -> путь сохранения файла
        confidence_threshold    (float) -> степень уверенности модели при детекции
        iou_threshold           (float) -> значение метрики Intersection over Union
        is_streaming            (bool)  -> опция остановки процесса видео
        save_db                 (bool)  -> сохранять ли данные в бд
        show_video              (bool)  -> показывать ли обработку видео в реальном времени
        zones_in_polygons       (list)  -> список in-полигонов
        zones_out_polygons      (list)  -> список out-полигонов
        class_id                (list)  -> список классов для задания разным трекерам (временно не используется)
    """

    def __init__(
            self,
            source_weights_path: str,
            source_video_path: str,
            target_video_path: str = None,
            confidence_threshold: float = 0.3,
            iou_threshold: float = 0.7,
            is_streaming: bool = True,
            save_db: bool = True,
            show_video: bool = True,
            zones_in_polygons: list = None,
            zones_out_polygons: list = None,
            procces_video_id: int = None,
            session_id: str = None,
            class_id: list = [0, 1, 2, 3, 5, 7]
    ) -> None:
        self.model = YOLO(source_weights_path)
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.is_streaming = is_streaming
        self.save_db = save_db
        self.session_id = session_id
        self.frame = None
        self.show_video = show_video
        self.class_id = class_id

        self.data_trace = []  # Данные о трассировках
        self.data_polygons = []  # Данные о инофрмации на полигонах
        self.polygons = []  # Данные о самих полигонах

        self.video_info = sv.VideoInfo.from_video_path(source_video_path)

        self.tracker = sv.ByteTrack()

        self.zones_in = self.initiate_polygon_zones(zones_in_polygons, [sv.Position.CENTER])
        self.zones_out = self.initiate_polygon_zones(zones_out_polygons, [sv.Position.CENTER])

        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.label_annotator = sv.LabelAnnotator(
            color=COLORS, text_color=sv.Color.BLACK
        )
        self.trace_annotator = sv.TraceAnnotator(
            color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2
        )
        self.detections_manager = DetectionsManager()

        # определение id обработки видео
        self.procces_video_id = procces_video_id

    @staticmethod
    def initiate_polygon_zones(
            polygons: Dict,
            triggering_anchors: Iterable[sv.Position] = [sv.Position.CENTER],
    ) -> List[Dict]:
        """
        Инициализация полигонов
        Args:
            polygons -> словарь {'name': str, 'points': List[List]}
            triggering_anchors -> Iterable[sv.Position]
        Return: List[Dict['polygon': sv.PolygonZone, 'name': str]]
        """
        return [{
            'polygon': sv.PolygonZone(
                polygon=np.array(polygon['points']),
                triggering_anchors=triggering_anchors),
            'name': polygon['name'],
            'permanent_counting': 0,
            'permanent_counting_dict': {}
        } for polygon in polygons]

    def send_frame(self, frame):
        """
        Передает файлы в self.frame для дальнейшей передачи в Response
        """
        _, jpeg = cv2.imencode('.jpg', frame)
        self.frame = jpeg.tobytes()

    def get_frame(self):
        """
        Генератор поочередного отображения фреймов в stream
        """
        while True:
            if self.frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + self.frame + b'\r\n')
            time.sleep(0.01)

    def stop_stream(self):
        """
        Остановка потока stream и обработки видео
        """
        self.is_streaming = False  # Остановка потока

    def get_procces_video_id(self):
        return self.procces_video_id

    def process_video(self):
        """
        Функция обработки видео с учетом:
        self.show_video         -> наблюдать процесс обработки видео / не наблюдать
        self.target_video_path  -> сохранять конечное обработанное видео / не сохранять
        self.is_streaming       -> опция остановки процесса видео
        """

        # Инициализация генератора фреймов
        generator = sv.get_video_frames_generator(
            source_path=self.source_video_path
        )
        frames_batch = []
        try:
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
                        if self.is_streaming is True:
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
                    if self.is_streaming is True:
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
            print('PolygonHandler.process_video ->', e)
            traceback.print_exc()

        finally:
            if self.save_db:

                # задание in-полигонов
                for in_polygon in self.zones_in:
                    self.polygons.append((
                        int(self.procces_video_id),
                        str(in_polygon['name']),
                        str(in_polygon['polygon'].polygon.tolist()),
                        True,
                        ntpath.basename(self.source_video_path),
                        datetime.datetime.now(),
                        datetime.date.today()
                    ))

                # задание out-полигонов
                for out_polygon in self.zones_out:
                    self.polygons.append((
                        int(self.procces_video_id),
                        str(out_polygon['name']),
                        str(out_polygon['polygon'].polygon.tolist()),
                        False,
                        ntpath.basename(self.source_video_path),
                        datetime.datetime.now(),
                        datetime.date.today()
                    ))

                # определение id записи об объектах
                for i, trace in enumerate(self.data_trace):
                    self.data_trace[i] = (self.procces_video_id,) + trace

                # определение procces_video_id в записи о полигонах
                for i, polygon in enumerate(self.data_polygons):
                    self.data_polygons[i] = (self.procces_video_id,) + polygon

                # Запись объектов детекции
                multy_insert(self.data_trace,
                             ["task_id", "x_min", "y_min", "x_max", "y_max", "x_center", "y_center", "class_id",
                              "confidence",
                              "tracker_id", "class_name", "current_frame", "total_frames", "is_polygon", "file_name",
                              "process_dttm", "day"],
                             SQLConfig.SQL_SCHEME,
                             SQLConfig.TRAFFIC,
                             message='Connection completed -> TRAFFIC')

                # Запись информации на полигонах
                multy_insert(self.data_polygons,
                             ["task_id", "name", "is_in", "coordinates", "current_count", "permanent_counting",
                              "permanent_counting_dict", "file_name", "process_dttm", "day"],
                             SQLConfig.SQL_SCHEME,
                             SQLConfig.TRAFFIC_COUNT_POLYGON,
                             message='Connection completed -> TRAFFIC_COUNT_POLYGON')

                # Записи самих полигонов
                multy_insert(self.polygons,
                             ["task_id", "name", "polygon", "is_in", "file_name", "process_dttm", "day"],
                             SQLConfig.SQL_SCHEME,
                             SQLConfig.TRAFFIC_POLYGONS,
                             message='Connection completed -> TRAFFIC_POLYGONS')

                execute_sql(f"update {SQLConfig.SQL_SCHEME}.{SQLConfig.TASKS} set status = 'done', "
                            f"end_process_dttm = '{datetime.datetime.now()}'"
                            f"where task_id = {self.procces_video_id}",
                            params={},
                            is_select=False,
                            message='Write status')

    def process_batch(self, frames: list, frame_numbers: list) -> list:
        # Прохождение одного фрейма через модель и вычленение детекций
        results = self.model(frames, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold)
        annotated_frames = []
        for frame, result, frame_number in zip(frames, results, frame_numbers):
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[(np.isin(detections.class_id, self.class_id))]
            detections = self.tracker.update_with_detections(detections)
            for i in range(len(detections.xyxy)):
                # Массив данных для выгрузки в базу
                self.data_trace.append((float(detections.xyxy[i][0]),  # x_min
                                        float(detections.xyxy[i][1]),  # y_min
                                        float(detections.xyxy[i][2]),  # x_max
                                        float(detections.xyxy[i][3]),  # y_max
                                        (float(detections.xyxy[i][0]) + float(  # x_center
                                            detections.xyxy[i][2])) / 2,
                                        (float(detections.xyxy[i][1]) + float(  # y_center
                                            detections.xyxy[i][3])) / 2,
                                        int(detections.class_id[i]),  # class_id
                                        float(detections.confidence[i]),  # confidence
                                        int(detections.tracker_id[i]),  # tracker_id
                                        str(detections.data['class_name'][i]),  # class_name
                                        int(frame_number),  # frame_number
                                        int(self.video_info.total_frames),  # total frames
                                        True,  # is_polygon
                                        ntpath.basename(self.source_video_path),  # file_name
                                        datetime.datetime.now(),  # process_dttm
                                        datetime.date.today()  # date
                                        ))

            detections_in_zones = []
            detections_out_zones = []

            for zone_in, zone_out in zip(self.zones_in, self.zones_out):
                detections_in_zone = detections[zone_in['polygon'].trigger(detections=detections)]
                detections_in_zones.append(detections_in_zone)

                detections_out_zone = detections[zone_out['polygon'].trigger(detections=detections)]
                detections_out_zones.append(detections_out_zone)

            detections = self.detections_manager.update(
                detections, detections_in_zones, detections_out_zones
            )
            if frame_number % 100 == 0 or frame_number == self.video_info.total_frames:
                execute_sql(f"update {SQLConfig.SQL_SCHEME}.{SQLConfig.TASKS} set status = 'run', "
                            f"current_frame = {frame_number}, end_process_dttm = '{datetime.datetime.now()}'"
                            f"where task_id = {self.procces_video_id}",
                            params={},
                            is_select=False)
            frame = self.annotate_frame(frame, detections)
            annotated_frames.append(frame)
        return annotated_frames

    def annotate_frame(
            self,
            frame: np.ndarray,  # фрейм видео
            detections: sv.Detections  # детекции из DetectionsManager()
    ) -> np.ndarray:
        annotated_frame = frame.copy()

        # отрисовка полигонов
        for i, (zone_in, zone_out) in enumerate(zip(self.zones_in, self.zones_out)):
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_in['polygon'].polygon, COLORS.colors[i]
            )
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_out['polygon'].polygon, COLORS.colors[i]
            )

        # задание надписи для каждого объекта
        labels = [f"{tracker_id} {dtype['class_name']} {confidence:0.2f}"
                  for xyxy, mask, confidence, class_id, tracker_id, dtype
                  in detections]

        # отрисовка трассировок, боксов и надписей соответственно
        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.label_annotator.annotate(annotated_frame, detections, labels)

        # отрисовка подсчета стриггеренных уникальных объектов на out-полигонов
        for zone_out_id, zone_out in enumerate(self.zones_out):
            zone_center = sv.get_polygon_center(polygon=zone_out['polygon'].polygon)
            if zone_out_id in self.detections_manager.counts:
                counts = self.detections_manager.counts[zone_out_id]
                for i, zone_in_id in enumerate(counts):
                    count = len(self.detections_manager.counts[zone_out_id][zone_in_id])
                    text_anchor = sv.Point(x=zone_center.x, y=zone_center.y + 40 * i)
                    annotated_frame = sv.draw_text(
                        scene=annotated_frame,
                        text=str(count),
                        text_anchor=text_anchor,
                        background_color=COLORS.colors[zone_in_id],
                    )

        # подсчет на момент данного фрейма стриггеренных объектов на out-полигонах
        for zone_out_id, zone_out in enumerate(self.zones_out):
            if zone_out_id in self.detections_manager.counts:
                # множество детекций по zone_in_id
                counts = self.detections_manager.counts[zone_out_id]  # список Dict[zone_in_id: int, Set[int]]
                for zone_in_id in counts:

                    # текущий in-полигон
                    zone_in = self.zones_in[zone_in_id]
                    # подсчет объектов
                    count = len(self.detections_manager.counts[zone_out_id][zone_in_id])

                    # добавление zone_in в zone_out['permanent_counting_dict'] при отсутствии zone_in_id
                    if zone_in['name'] not in zone_out['permanent_counting_dict'].keys():
                        zone_out['permanent_counting_dict'][zone_in['name']] = 0
                    # добавление zone_out в zone_in['permanent_counting_dict'] при отсутствии zone_out_id
                    if zone_out['name'] not in zone_in['permanent_counting_dict'].keys():
                        zone_in['permanent_counting_dict'][zone_out['name']] = 0

                    # сравнение объектов на in-полигоне, которые попали на определенные out-полигоны
                    if zone_in['permanent_counting_dict'][zone_out['name']] != count:
                        # обновляем каунт по текущему zone_out_id
                        zone_in['permanent_counting_dict'][zone_out['name']] = count
                        # обновляем общий подсчет на zone_in
                        zone_in['permanent_counting'] = sum(zone_in['permanent_counting_dict'].values())

                        if self.save_db:
                            self.data_polygons.append((
                                zone_in['name'],  # наименование полигона             -> str
                                True,  # типа полигона                     -> bool
                                zone_in['polygon'].polygon.tolist(),  # список координатов полигона       -> (N, 2)
                                int(zone_in['polygon'].current_count),  # кол-во объектов в полигоне на фрейме  -> int
                                int(zone_in['permanent_counting']),  # счет объектов за все время        -> int
                                json.loads(json.dumps(zone_in['permanent_counting_dict'])),
                                # словарь объектов по полигонам     -> dict
                                ntpath.basename(self.source_video_path),  # путь до видео
                                datetime.datetime.now(),  # дата со временем
                                datetime.date.today()  # дата
                            ))

                    # сравнение объектов на out-полигоне с определенных in-полигонов
                    if zone_out['permanent_counting_dict'][zone_in['name']] != count:
                        # обновляем каунт по текущему zone_in_id
                        zone_out['permanent_counting_dict'][zone_in['name']] = count
                        # обновляем общий подсчет на zone_out
                        zone_out['permanent_counting'] = sum(zone_out['permanent_counting_dict'].values())

                        if self.save_db:
                            self.data_polygons.append((
                                zone_out['name'],  # наименование полигона             -> str
                                False,  # типа полигона                     -> bool
                                zone_out['polygon'].polygon.tolist(),  # список координатов полигона       -> (N, 2)
                                int(zone_out['polygon'].current_count),  # кол-во объектов в полигоне на фрейме  -> int
                                int(zone_out['permanent_counting']),  # счет объектов за все время        -> int
                                json.loads(json.dumps(zone_out['permanent_counting_dict'])),
                                # словарь объектов по полигонам     -> dict
                                ntpath.basename(self.source_video_path),  # путь до видео
                                datetime.datetime.now(),  # дата со временем
                                datetime.date.today()  # дата
                            ))

        return annotated_frame
