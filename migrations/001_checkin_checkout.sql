-- Safe, additive migration for check-in/check-out.
-- BACK UP the attend database before running this file.
-- Existing attendance rows and arrival_time values are preserved.

USE attend;

ALTER TABLE attendance
    ADD COLUMN departure_time TIME NULL AFTER arrival_time,
    ADD COLUMN working_duration_seconds INT UNSIGNED NULL AFTER departure_time;

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
