import ntpath
import os
import sys
import re
import cv2

sys.path.append("ml/recognition_of_video_from_cameras/src")
from utils import execute_sql
from config import SQLConfig

current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))


def painter(path: str, traces: dict, thickness=1, alpha=0.5):
    image = cv2.imread(path)
    # Создаём маску для полупрозрачных линий
    overlay = image.copy()
    color = (255, 255, 255)
    color_person = (255, 255, 0)

    for trace in traces:
        for (x1, y1, start_class), (x2, y2, end_class) in zip(traces[trace], traces[trace][1:]):
            if start_class == 'person':
                overlay = cv2.line(overlay, (x1, y1), (x2, y2), color_person, thickness)
            else:
                overlay = cv2.line(overlay, (x1, y1), (x2, y2), color, thickness)

            # Наложение маски с полупрозрачностью на исходное изображение
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    save_path = PROJECT_ROOT + '/recognition_of_video_from_cameras/data/features/images/' + \
                re.sub(r'_task_id_\d+', '', ntpath.basename(path).split('.')[0].split('.')[0]) + '_trace.jpg'
    cv2.imwrite(save_path, image)


def trace_create(path: str, task_id: int):
    # Отрисовка трасс объектов на скриншоте
    query_id = (f"select tracker_id, x_center, y_center, class_name from {SQLConfig.SQL_SCHEME}.{SQLConfig.TRAFFIC} "
                f"where task_id = {task_id} order by process_dttm")
    result = execute_sql(query_id, params={})
    traces = {}
    for i in result:
        if i['tracker_id'] not in traces:
            traces[i['tracker_id']] = [(int(i['x_center']), int(i['y_center']), str(i['class_name']))]
        else:
            traces[i['tracker_id']].append((int(i['x_center']), int(i['y_center']), str(i['class_name'])))
    painter(path, traces)
