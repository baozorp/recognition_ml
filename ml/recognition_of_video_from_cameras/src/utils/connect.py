import json

import pandas as pd
import psycopg2 as db
import psycopg2.extras
import io
import sys


sys.path.append("ml/recognition_of_video_from_cameras/src")

from config import SQLConfig


def get_connection(message):
    '''
    Создание подключения
    '''
    conn_opts = {
        'host': SQLConfig.SQL_HOSTNAME,
        'port': SQLConfig.SQL_PORT,
        'user': SQLConfig.SQL_USERNAME,
        'password': SQLConfig.SQL_PASSWORD,
        'dbname': SQLConfig.SQL_MAIN_DATABASE
    }
    conn = db.connect(**conn_opts)
    if message is not None:
        print(message)

    return conn


def multy_insert(data, columns, schema, table, message='Connection completed'):
    '''
    Вставка в базу сразу всего list
    '''
    conn = get_connection(message)
    df = pd.DataFrame(data, columns=columns)
    # Почему такой инсерт
    # https://stackoverflow.com/questions/47429651/how-i-can-insert-data-from-dataframein-python-to-greenplum-table
    csv_io = io.StringIO()
    df.to_csv(csv_io, sep='\t', header=False, index=False)
    csv_io.seek(0)
    gp_cursor = conn.cursor()
    gp_cursor.execute(f"SET search_path = {schema}")
    gp_cursor.copy_from(csv_io, table)

    conn.commit()
    conn.close()


def execute_sql(sql, params, is_select=True, message=None):
    """
    Выполняем запрос с параметрами
    """
    data = []
    conn = get_connection(message)
    with conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            cursor.execute(sql, params)
            if is_select is True:
                data = cursor.fetchall()
            conn.commit()

    return data

