CREATE SCHEMA IF NOT EXISTS traffic_data_dev;

CREATE table if NOT EXISTS traffic_data_dev.sessions (
  session_id text NOT NULL,
  task_id int4 NULL,
  session_data jsonb NULL,
  created_at timestamptz DEFAULT now() NULL,
  updated_at timestamptz DEFAULT now() NULL,
  CONSTRAINT sessions_pkey PRIMARY KEY (session_id)
);

CREATE table if NOT EXISTS traffic_data_dev.tasks (
  session_id text NULL,
  task_id int4 NULL,
  task_type text NULL,
  file_name text NULL,
  current_frame int4 NULL,
  total_frames int4 NULL,
  status text NULL,
  start_process_dttm timestamp NULL,
  end_process_dttm timestamp NULL,
  "day" date NULL
);

CREATE TABLE IF NOT EXISTS traffic_data_dev.traffic (
  task_id int4 NULL,
  x_min float8 NULL,
  y_min float8 NULL,
  x_max float8 NULL,
  y_max float8 NULL,
  x_center float8 NULL,
  y_center float8 NULL,
  class_id int4 NULL,
  confidence float8 NULL,
  tracker_id int4 NULL,
  class_name text NULL,
  current_frame int4 NULL,
  total_frames int4 NULL,
  is_polygon bool NULL,
  file_name text NULL,
  process_dttm timestamp NULL,
  "day" date NULL
);

CREATE TABLE IF NOT EXISTS traffic_data_dev.traffic_count_line (
  task_id int4 NULL,
  "name" text NULL,
  vector_start text NULL,
  vector_end text NULL,
  count_in int4 NULL,
  count_out int4 NULL,
  file_name text NULL,
  process_dttm timestamp NULL,
  "day" date NULL
);

CREATE TABLE IF NOT EXISTS traffic_data_dev.traffic_count_polygon (
  task_id int4 NULL,
  "name" text NULL,
  is_in bool NULL,
  coordinates text NULL,
  current_count int4 NULL,
  permanent_counting int4 NULL,
  permanent_counting_dict text NULL,
  file_name text NULL,
  process_dttm timestamp NULL,
  "day" date NULL
);

CREATE TABLE IF NOT EXISTS traffic_data_dev.traffic_lines (
  task_id int4 NULL,
  "name" text NULL,
  line text NULL,
  file_name text NULL,
  process_dttm timestamp NULL,
  "day" date NULL
);

CREATE TABLE IF NOT EXISTS traffic_data_dev.traffic_polygons (
  task_id int4 NULL,
  "name" text NULL,
  polygon text NULL,
  is_in bool NULL,
  file_name text NULL,
  process_dttm timestamp NULL,
  "day" date NULL
);