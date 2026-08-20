-- Employee attendance system schema.
--
--   mysql -u root -p < schema.sql
--
-- The database name must match ATTENDANCE_DB_NAME (default: attend).

CREATE DATABASE IF NOT EXISTS attend
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE attend;

CREATE TABLE IF NOT EXISTS employees (
    employee_id   VARCHAR(64)  NOT NULL,
    employee_name VARCHAR(255) NOT NULL,
    image1        LONGBLOB,
    image2        LONGBLOB,
    image3        LONGBLOB,
    is_active     BOOLEAN      NOT NULL DEFAULT TRUE,
    PRIMARY KEY (employee_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS attendance (
    employee_id     VARCHAR(64)  NOT NULL,
    employee_name   VARCHAR(255) NOT NULL,
    attendance_date DATE         NOT NULL,
    arrival_time    TIME         NOT NULL,
    status          VARCHAR(32)  NOT NULL DEFAULT 'present',
    -- One row per employee per day. The application relies on this
    -- constraint for its ON DUPLICATE KEY UPDATE on repeat check-ins.
    UNIQUE KEY uniq_employee_day (employee_id, attendance_date),
    CONSTRAINT fk_attendance_employee
        FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
        ON DELETE CASCADE
) ENGINE=InnoDB;
