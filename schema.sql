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
    departure_time  TIME         NULL,
    working_duration_seconds INT UNSIGNED NULL,
    status          VARCHAR(32)  NOT NULL DEFAULT 'present',
    -- One row per employee per day; check-in is never overwritten.
    UNIQUE KEY uniq_employee_day (employee_id, attendance_date),
    CONSTRAINT fk_attendance_employee
        FOREIGN KEY (employee_id) REFERENCES employees (employee_id)
        ON DELETE CASCADE
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS attendance_settings (
    settings_id TINYINT UNSIGNED NOT NULL PRIMARY KEY,
    checkin_time TIME NOT NULL DEFAULT '09:00:00',
    checkin_cutoff TIME NOT NULL DEFAULT '09:30:00',
    checkout_start TIME NOT NULL DEFAULT '18:00:00',
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;

INSERT INTO attendance_settings
    (settings_id, checkin_time, checkin_cutoff, checkout_start)
VALUES
    (1, '09:00:00', '09:30:00', '18:00:00')
ON DUPLICATE KEY UPDATE settings_id = VALUES(settings_id);
