import asyncio
import base64
import datetime
import json
import ntpath
import os
import re
import sys
import traceback
import uuid
from fastapi import FastAPI, Request, Form, UploadFile, File, BackgroundTasks, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pathlib import Path
from urllib.parse import quote

sys.path.append("ml/recognition_of_video_from_cameras/src")
from utils import LineHandler, PolygonHandler
from utils import save_frame
from utils import execute_sql, multy_insert
from utils import trace_create
from config import SQLConfig

app = FastAPI()

# Проставление путей
current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = current_dir.parent.parent.parent
UPLOAD_FOLDER = PROJECT_ROOT / 'recognition_of_video_from_cameras/data/raw'

# Путь к папке со статикой
static_dir = Path("ml/recognition_of_video_from_cameras/src/visualization/static")

# Путь к шаблонам
template_dir = Path("ml/recognition_of_video_from_cameras/src/visualization/templates")

# Подключение статических файлов
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=template_dir)
# Добавление middleware для работы с сессиями
app.add_middleware(
    SessionMiddleware,
    secret_key=SQLConfig.SECRET_KEY,
    max_age=86400,  # Время жизни куки в секундах
    session_cookie="session_id",  # Имя куки
    same_site="lax"  # Политика SameSite для куки (None, Lax, Strict)
)

ACTIVE_TASKS = {}


# Модель для хранения настроек
class Model:
    def __init__(self):
        self.model_path = PROJECT_ROOT / 'recognition_of_video_from_cameras/src/models/yolov10l.pt'
        self.target_video = False
        self.confidence = 0.2
        self.iou = 0.2
        self.directory = PROJECT_ROOT / 'recognition_of_video_from_cameras/data/raw/'
        self.files = os.listdir(self.directory)
        self.processor = None
        self.save_db = True
        self.show_video = False
        self.is_polygons = False
        self.save_path_line = ''
        self.id = None
        self.image_path = 'recognition_of_video_from_cameras/data/stage/image/'


def save_session_data_to_db(session_id, user_id, session_data):
    query = f"""
        INSERT INTO {SQLConfig.SQL_SCHEME}.{SQLConfig.SESSIONS} (session_id, task_id, session_data) 
        VALUES (%s, %s, %s)
        ON CONFLICT (session_id) DO UPDATE 
        SET task_id = EXCLUDED.task_id, session_data = EXCLUDED.session_data, updated_at = NOW();
    """
    execute_sql(query,
                (session_id, user_id, json.dumps(session_data)),
                is_select=False)


def restore_session(request):
    """
    Восстановление сессии при разрыве соединения
    """
    session_id = request.session.get("session_id")

    if session_id:
        session_data = execute_sql(f"SELECT session_data FROM {SQLConfig.SQL_SCHEME}.{SQLConfig.SESSIONS} "
                                   f"WHERE session_id = '{session_id}'",
                                   params={},
                                   is_select=True)
        return session_data[0]['session_data']


def create_model(request, session):
    request.session["session_data"] = session
    ACTIVE_TASKS[session["task_id"]] = Model()
    ACTIVE_TASKS[session["task_id"]].id = session["task_id"]


def get_task_id(request):
    """
    Получение номера задачи
    """
    session_id = request.session.get("session_id")
    session_data = request.session.get("session_data", {})
    if "task_id" not in session_data:
        # Если task_id нет значит берем задачу
        # определение id обработки видео
        id_counter = execute_sql(f'SELECT MAX(task_id) FROM {SQLConfig.SQL_SCHEME}.{SQLConfig.SESSIONS}',
                                 params={},
                                 is_select=True)
        session_data["task_id"] = 0 if id_counter[0]['max'] is None else id_counter[0]['max'] + 1
        create_model(request, session_data)
    else:
        # Если task_id есть значит пробуем восстановить сессию
        if session_data["task_id"] not in ACTIVE_TASKS:
            # Если такой задачи нет значит в списке классов значит есть что восстановить
            data = restore_session(request)
            request.session["session_data"] = data
            ACTIVE_TASKS[session_data["task_id"]] = Model()
            ACTIVE_TASKS[session_data["task_id"]].id = session_data["task_id"]
            ACTIVE_TASKS[session_data["task_id"]].iou = data.get('iou', ACTIVE_TASKS[session_data["task_id"]].iou)
            ACTIVE_TASKS[session_data["task_id"]].confidence = data.get('confidence',
                                                                        ACTIVE_TASKS[
                                                                            session_data["task_id"]].confidence)
            ACTIVE_TASKS[session_data["task_id"]].model_path = data.get('model_path',
                                                                        ACTIVE_TASKS[
                                                                            session_data["task_id"]].model_path)
            ACTIVE_TASKS[session_data["task_id"]].show_video = data.get('show_video',
                                                                        ACTIVE_TASKS[
                                                                            session_data["task_id"]].show_video)
            ACTIVE_TASKS[session_data["task_id"]].is_polygons = data.get('is_polygons',
                                                                         ACTIVE_TASKS[
                                                                             session_data["task_id"]].is_polygons)
            ACTIVE_TASKS[session_data["task_id"]].target_video = data.get('target_video',
                                                                          ACTIVE_TASKS[
                                                                              session_data["task_id"]].target_video)
            ACTIVE_TASKS[session_data["task_id"]].save_path_line = data.get('save_path_line',
                                                                            ACTIVE_TASKS[
                                                                                session_data["task_id"]].save_path_line)

        task_id = ACTIVE_TASKS[session_data["task_id"]].id
        id_task = execute_sql(f"SELECT task_id FROM {SQLConfig.SQL_SCHEME}.{SQLConfig.TASKS} "
                              f"where task_id = '{task_id}'",
                              params={},
                              is_select=True)

        if len(id_task) != 0:
            # определение id обработки видео
            id_counter = execute_sql(f'SELECT MAX(task_id) FROM {SQLConfig.SQL_SCHEME}.{SQLConfig.SESSIONS}',
                                     params={},
                                     is_select=True)

            session_data = {"preferences": {}, "task_id": id_counter[0]['max'] + 1}
            create_model(request, session_data)
    # Сохраняем изменения в сессии
    save_session_data_to_db(session_id, ACTIVE_TASKS[session_data["task_id"]].id, session_data)


@app.get("/")
async def show_main_page(request: Request):
    """
    Отображение главной страницы
    """
    session = request.session.get("session_data")

    if not session:
        # Создаем новую сессию
        session_id = str(uuid.uuid4())
        request.session["session_id"] = session_id
        session = {"preferences": {}}
        # Сохраняем изменения в сессии
        request.session["session_data"] = session
    get_task_id(request)

    return templates.TemplateResponse("index.html", {"request": request,
                                                     "files": ACTIVE_TASKS[session["task_id"]].files})


@app.post("/", response_class=HTMLResponse)
async def select_video(request: Request, comp_select: str = Form(...)):
    """
    Выбор видео из списка ранее загруженных
    """
    session_id = request.session.get("session_id")
    session_data = request.session.get("session_data", {})

    session_data["file_path"] = str(UPLOAD_FOLDER / comp_select)

    session_data["path_image"] = str(PROJECT_ROOT / f"recognition_of_video_from_cameras/data/features" \
                                                    f"/images/{ntpath.basename(session_data['file_path']).split('.')[0]}_trace.jpg")

    if not os.path.exists(session_data["path_image"]):
        save_frame(session_data["file_path"], 1, session_data["path_image"])
        session_data["path_image"] = str(PROJECT_ROOT / f"{ACTIVE_TASKS[session_data['task_id']].image_path}"
                                                        f"{ntpath.basename(session_data['file_path']).split('.')[0]}"
                                                        f"_task_id_{ACTIVE_TASKS[session_data['task_id']].id}.jpg")

    # Обновляем сессию в базе данных
    get_task_id(request)
    # Сохраняем изменения в сессии
    request.session["session_data"] = session_data
    save_session_data_to_db(session_id, ACTIVE_TASKS[session_data["task_id"]].id, session_data)

    return templates.TemplateResponse("draw.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def handle_file_upload(request: Request, files: list[UploadFile] = File(...)):
    """
    Загрузка видео на сервер
    """
    session_id = request.session.get("session_id")
    session_data = request.session.get("session_data", {})
    for uploaded_file in files:
        filename = uploaded_file.filename
        file_path = UPLOAD_FOLDER / filename
        session_data["file_path"] = str(file_path)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.file.read())

        path_image = PROJECT_ROOT / f"{ACTIVE_TASKS[session_data['task_id']].image_path}" \
                                    f"{ntpath.basename(session_data['file_path']).split('.')[0]}" \
                                    f"_task_id_{ACTIVE_TASKS[session_data['task_id']].id}.jpg"
        session_data["path_image"] = str(path_image)
        session_data["lines"] = []
        session_data["polygons"] = []
        session_data["is_polygons"] = True
        # Сохраняем изменения в сессии
        request.session["session_data"] = session_data
        save_frame(file_path, 1, path_image)
    # Обновляем сессию в базе данных
    get_task_id(request)
    # Сохраняем изменения в сессии
    request.session["session_data"] = session_data
    save_session_data_to_db(session_id, ACTIVE_TASKS[session_data["task_id"]].id, session_data)
    ACTIVE_TASKS[session_data["task_id"]].files = os.listdir(ACTIVE_TASKS[session_data["task_id"]].directory)
    # Запуск обработки видео сразу после загрузки
    await process_polygons_video_async(request, None)

    return templates.TemplateResponse("index.html",
                                      {"request": request, "status": "Файл успешно загружен и обработка запущена!",
                                       "files": ACTIVE_TASKS[session_data["task_id"]].files})


@app.get("/draw", response_class=HTMLResponse)
async def draw_page(request: Request):
    """
    Отображение страницы canvas, для размещения створ и полигонов
    """
    return templates.TemplateResponse("draw.html", {"request": request})


@app.get("/getfile", response_class=JSONResponse)
def return_files(request: Request):
    """
    Передача изображения на страницу draw
    """
    session_data = request.session.get("session_data", {})
    image = {'name': os.path.basename(session_data["path_image"])}
    with open(session_data["path_image"], 'rb') as img:
        image['file'] = base64.b64encode(img.read()).decode()

    return JSONResponse([image])  # Возвращаем JSON-ответ


@app.post('/saveCoords', response_class=JSONResponse)
async def save_coords(request: Request):
    """
    Сохранение координат в переменную объект model и бд
    """
    # Выгрузка координатов фигур
    session_id = request.session.get("session_id")
    session_data = request.session.get("session_data", {})
    body = await request.body()  # Получаем тело запроса
    coordinates = body.decode('utf-8')
    coordinates = re.sub(r'\"|\'|b|\s|" "', '', coordinates)
    coordinates = re.sub(r'\\', '"', coordinates)

    coordinates = json.loads(coordinates)  # dict

    polygons = []
    lines = []
    is_polygons = None
    # Условие для определения какая фигура к какому классу относится
    for figure in coordinates:
        if len(figure['points']) > 2:
            polygons.append(figure)
            is_polygons = True
        else:
            lines.append(figure)
            is_polygons = False

    session_data["polygons"] = polygons
    session_data["lines"] = lines
    session_data["is_polygons"] = is_polygons
    # Сохраняем изменения в сессии
    request.session["session_data"] = session_data
    save_session_data_to_db(session_id, ACTIVE_TASKS[session_data["task_id"]].id, session_data)
    ACTIVE_TASKS[session_data["task_id"]].files = os.listdir(ACTIVE_TASKS[session_data["task_id"]].directory)

    return templates.TemplateResponse('draw.html',
                                      {"request": request, "status": 'Координаты сохранены',
                                       "files": ACTIVE_TASKS[session_data["task_id"]].files})


@app.post("/save_settings", response_class=JSONResponse)
async def save_settings(request: Request, data: dict):
    """
    Производим сохранение настроек
    """
    session_id = request.session.get("session_id")
    session_data = request.session.get("session_data", {})
    session_data["model_path"] = str(data.get('model_path', ACTIVE_TASKS[session_data["task_id"]].model_path))
    session_data["target_video"] = data.get('target_video', ACTIVE_TASKS[session_data["task_id"]].target_video)
    session_data["show_video"] = data.get('show_video', ACTIVE_TASKS[session_data["task_id"]].show_video)
    session_data["confidence"] = float(data.get('confidence', ACTIVE_TASKS[session_data["task_id"]].confidence))
    session_data["iou"] = float(data.get('iou', ACTIVE_TASKS[session_data["task_id"]].iou))

    # Сохраняем изменения в сессии
    request.session["session_data"] = session_data
    ACTIVE_TASKS[session_data["task_id"]].model_path = session_data["model_path"]
    ACTIVE_TASKS[session_data["task_id"]].target_video = session_data["target_video"]
    ACTIVE_TASKS[session_data["task_id"]].show_video = session_data["show_video"]
    ACTIVE_TASKS[session_data["task_id"]].confidence = session_data["confidence"]
    ACTIVE_TASKS[session_data["task_id"]].iou = session_data["iou"]
    save_session_data_to_db(session_id, ACTIVE_TASKS[session_data["task_id"]].id, session_data)

    return JSONResponse(content={}, status_code=201)


@app.post('/select_trace', response_class=JSONResponse)
def select_trace(request: Request):
    """
    Список доступных изображений для отрисовки полигонов и створ
    """
    session_data = request.session.get("session_data", {})
    return templates.TemplateResponse('select_trace.html', {"request": request})


@app.post("/trace", response_class=HTMLResponse)
async def trace_image(request: Request, task_id: int = Form(...)):
    """
    Отрисовка трассировки на первом кадре видео. Получение имени
    изображения и пути для сохранения обработанного изображения.
    """
    session_data = request.session.get("session_data", {})
    directory = PROJECT_ROOT / f"{ACTIVE_TASKS[session_data['task_id']].image_path}"
    pattern = f'task_id_{task_id}.'
    image_name = [f for f in os.listdir(directory) if pattern in f]
    if len(image_name) > 0:
        image_name = image_name[0]
    else:
        return JSONResponse(content={"status": "Задача не найдена."}, status_code=404)
    file_path = PROJECT_ROOT / f"{ACTIVE_TASKS[session_data['task_id']].image_path}{image_name}"
    trace_create(file_path, task_id)
    return JSONResponse(content={"status": "Изображение обработано."}, status_code=202)


async def process_lines_video_async(request, target_video_path):
    """
    Вызов класса для работы с линиями
    """
    session_id = request.session.get("session_id")
    session_data = request.session.get("session_data", {})
    ACTIVE_TASKS[session_data["task_id"]].processor = LineHandler(
        source_weights_path=ACTIVE_TASKS[session_data["task_id"]].model_path,
        source_video_path=session_data["file_path"],
        target_video_path=target_video_path,
        confidence_threshold=ACTIVE_TASKS[session_data["task_id"]].confidence,
        iou_threshold=ACTIVE_TASKS[session_data["task_id"]].iou,
        lines=session_data["lines"],
        show_video=ACTIVE_TASKS[session_data["task_id"]].show_video,
        save_db=ACTIVE_TASKS[session_data["task_id"]].save_db,
        procces_video_id=session_data["task_id"],
        session_id=session_id
    )
    await asyncio.to_thread(ACTIVE_TASKS[session_data["task_id"]].processor.process_video)


async def process_polygons_video_async(request, target_video_path):
    """
    Вызов класса для работы с полигонами
    """
    session_id = request.session.get("session_id")
    session_data = request.session.get("session_data", {})
    ACTIVE_TASKS[session_data["task_id"]].processor = PolygonHandler(
        source_weights_path=ACTIVE_TASKS[session_data["task_id"]].model_path,
        source_video_path=session_data["file_path"],
        target_video_path=target_video_path,
        confidence_threshold=ACTIVE_TASKS[session_data["task_id"]].confidence,
        iou_threshold=ACTIVE_TASKS[session_data["task_id"]].iou,
        zones_in_polygons=[poly for poly in session_data["polygons"][::2]],
        zones_out_polygons=[poly for poly in session_data["polygons"][1::2]],
        show_video=ACTIVE_TASKS[session_data["task_id"]].show_video,
        save_db=ACTIVE_TASKS[session_data["task_id"]].save_db,
        procces_video_id=session_data["task_id"],
        session_id=session_id
    )
    await asyncio.to_thread(ACTIVE_TASKS[session_data["task_id"]].processor.process_video)


@app.post("/run", response_class=HTMLResponse)
async def run_video_processing(request: Request, background_tasks: BackgroundTasks):
    """
    Главная функция обработки видео и отображения его обработки в реальном времени
    """

    session_data = request.session.get("session_data", {})
    try:
        if session_data["target_video"]:
            target_video_path = PROJECT_ROOT / f"recognition_of_video_from_cameras/data/features/" \
                                               f"{ntpath.basename(session_data['file_path']).split('.')[0]}.mp4"
        else:
            target_video_path = None

        if session_data["is_polygons"]:
            background_tasks.add_task(process_polygons_video_async, request, target_video_path)
        else:
            background_tasks.add_task(process_lines_video_async, request, target_video_path)

        if session_data["show_video"]:
            return templates.TemplateResponse("video_feed.html", {"request": request})
        else:
            return JSONResponse(content={"status": "Обработка происходит в фоновом режиме."}, status_code=202)
    except Exception as exp:
        traceback.print_exc()  # Показывает traceback в консоли
        return JSONResponse(content={"error": str(exp)}, status_code=500)


@app.post("/save_image", response_class=JSONResponse)
async def save_image(request: Request, data: dict):
    """
    Сохранение изображения с холста на сервере
    """
    session_id = request.session.get("session_id")
    session_data = request.session.get("session_data", {})

    image_data = data.get('image')
    if image_data.startswith('data:image/png;base64,'):
        image_data = image_data.replace('data:image/png;base64,', '')
    image = base64.b64decode(image_data)
    filename = f"{ntpath.basename(session_data['file_path']).split('.')[0]}_task_id_{ACTIVE_TASKS[session_data['task_id']].id}.png"
    session_data["save_path_line"] = str(PROJECT_ROOT / f"{ACTIVE_TASKS[session_data['task_id']].image_path}"
                                                        f"{ntpath.basename(session_data['file_path']).split('.')[0]}"
                                                        f"_task_id_{ACTIVE_TASKS[session_data['task_id']].id}.png")
    # Сохраняем изменения в сессии
    request.session["session_data"] = session_data
    save_session_data_to_db(session_id, ACTIVE_TASKS[session_data["task_id"]].id, session_data)

    with open(session_data["save_path_line"], 'wb') as f:
        f.write(image)

    # Возвращаем изображение для автоматической загрузки
    return StreamingResponse(open(session_data["save_path_line"], 'rb'), media_type="image/png", headers={
        "Content-Disposition": f'attachment; filename=task_id_{ACTIVE_TASKS[session_data["task_id"]].id}'  # Используем имя файла
    })


def get_tasks_data(request):
    """
    Получение списка задач
    """
    session_id = request.session.get("session_id")
    tasks_from_db = execute_sql(
        f"SELECT task_id, file_name, current_frame, total_frames, status, start_process_dttm, end_process_dttm, day "
        f"FROM {SQLConfig.SQL_SCHEME}.{SQLConfig.TASKS} ORDER BY task_id DESC",
        params={},
        is_select=True,
        message='Get Tasks'
    )
    return tasks_from_db


@app.post("/tasks", response_class=HTMLResponse)
async def tasks_page(request: Request):
    """
    Отображение страницы со списком задач
    """
    return templates.TemplateResponse("tasks.html", {"request": request, "tasks": get_tasks_data(request)})


@app.get("/video_feed", response_class=HTMLResponse)
async def video_feed(request: Request):
    """
    Страница отображения потока видео
    """

    session_data = request.session.get("session_data", {})
    return templates.TemplateResponse("video_feed.html",
                                      {"request": request,
                                       "process_video_id": ACTIVE_TASKS[
                                           session_data["task_id"]].processor.get_procces_video_id()})


@app.get("/stream")
async def stream_video(request: Request):
    """
     Передача фреймов для показа трансляции обработки видео
     """
    session_data = request.session.get("session_data", {})
    return StreamingResponse(ACTIVE_TASKS[session_data["task_id"]].processor.get_frame(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/stop_stream", response_class=JSONResponse)
async def stop_stream(task_id: str = Query(...)):
    """
    Остановка процесса обработки видео и трансляции
    """
    # Используем task_id из параметров запроса
    if int(task_id) in ACTIVE_TASKS:
        ACTIVE_TASKS[int(task_id)].processor.stop_stream()
        return JSONResponse(content={"status": "Stream stopped."}, status_code=200)
    else:
        return JSONResponse(content={"status": "Task not found."}, status_code=404)


if __name__ == "__main__":
    import uvicorn

    # f_port = sys.argv[1]
    f_port = 8000
    uvicorn.run(app, host="0.0.0.0", port=int(f_port))
