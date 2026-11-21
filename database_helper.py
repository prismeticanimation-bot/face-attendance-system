import os

import mysql.connector
from mysql.connector import Error
import cv2
import numpy as np
from datetime import datetime, date


class DatabaseHelper:


    def __init__(self, host=None, user=None, password=None, database=None):
        # Credentials come from the environment so they are not baked into
        # the source; the previous hard-coded values remain the defaults.
        host = host if host is not None else os.environ.get('ATTENDANCE_DB_HOST', 'localhost')
        user = user if user is not None else os.environ.get('ATTENDANCE_DB_USER', 'root')
        password = password if password is not None else os.environ.get('ATTENDANCE_DB_PASSWORD', '1234')
        database = database if database is not None else os.environ.get('ATTENDANCE_DB_NAME', 'attend')

        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None

    def connect(self):

        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            if self.connection.is_connected():
                return True
        except Error as e:
            print(f"Database connection error: {e}")
            return False

    def disconnect(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()

    def add_employee(self, employee_id, employee_name, image1_path, image2_path, image3_path):

        try:
            cursor = self.connection.cursor()

            # Read image files as binary data
            with open(image1_path, 'rb') as f:
                image1_data = f.read()
            with open(image2_path, 'rb') as f:
                image2_data = f.read()
            with open(image3_path, 'rb') as f:
                image3_data = f.read()

            query = """INSERT INTO employees 
                      (employee_id, employee_name, image1, image2, image3) 
                      VALUES (%s, %s, %s, %s, %s)"""
            cursor.execute(query, (employee_id, employee_name, image1_data, image2_data, image3_data))
            self.connection.commit()
            cursor.close()
            return True
        except Error as e:
            print(f"Error adding employee: {e}")
            return False

    def get_all_employees(self):

        try:
            cursor = self.connection.cursor(dictionary=True)
            query = "SELECT employee_id, employee_name FROM employees WHERE is_active = TRUE"
            cursor.execute(query)
            employees = cursor.fetchall()
            cursor.close()
            return employees
        except Error as e:
            print(f"Error fetching employees: {e}")
            return []

    def get_employee_images(self, employee_id):

        try:
            cursor = self.connection.cursor()
            query = """SELECT image1, image2, image3 FROM employees 
                      WHERE employee_id = %s AND is_active = TRUE"""
            cursor.execute(query, (employee_id,))
            result = cursor.fetchone()
            cursor.close()

            if result:
                images = []
                for img_data in result:
                    # Decode binary image data
                    nparr = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    images.append(img)
                return images
            return None
        except Error as e:
            print(f"Error fetching employee images: {e}")
            return None

    def get_attendance_schedule(self):
        """Read the persistent office schedule; migration seeds the standard times."""
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(
                "SELECT checkin_time, checkin_cutoff, checkout_start "
                "FROM attendance_settings WHERE settings_id = 1"
            )
            row = cursor.fetchone()
            cursor.close()
            if not row:
                return {'checkin_time': '09:00:00', 'checkin_cutoff': '09:30:00',
                        'checkout_start': '18:00:00'}
            return {key: str(value) for key, value in row.items()}
        except Error as e:
            print(f"Error loading attendance schedule: {e}")
            return {'checkin_time': '09:00:00', 'checkin_cutoff': '09:30:00',
                    'checkout_start': '18:00:00'}

    def record_check_in(self, employee_id, checkin_at=None):
        """Insert exactly one check-in per employee/day; never overwrite an existing one."""
        cursor = None
        try:
            checkin_at = checkin_at or datetime.now()
            today = checkin_at.date()
            checkin_time = checkin_at.time().replace(microsecond=0)
            schedule = self.get_attendance_schedule()
            cutoff = datetime.strptime(schedule['checkin_cutoff'], '%H:%M:%S').time()
            on_time = checkin_time <= cutoff
            status = 'present' if checkin_time <= datetime.strptime(
                schedule['checkin_time'], '%H:%M:%S').time() else 'late'
            if checkin_time > cutoff:
                return {'ok': False, 'reason': 'checkin_closed'}

            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT employee_name FROM employees "
                "WHERE employee_id = %s AND is_active = TRUE", (employee_id,)
            )
            employee = cursor.fetchone()
            if not employee:
                return {'ok': False, 'reason': 'unknown_employee'}

            cursor.execute(
                "SELECT arrival_time FROM attendance "
                "WHERE employee_id = %s AND attendance_date = %s",
                (employee_id, today)
            )
            if cursor.fetchone():
                return {'ok': False, 'reason': 'already_checked_in'}

            cursor.execute(
                "INSERT INTO attendance "
                "(employee_id, employee_name, attendance_date, arrival_time, status) "
                "VALUES (%s, %s, %s, %s, %s)",
                (employee_id, employee[0], today, checkin_time, status)
            )
            self.connection.commit()
            return {'ok': True, 'status': status, 'time': str(checkin_time)}
        except Error as e:
            if self.connection:
                self.connection.rollback()
            print(f"Error recording check-in: {e}")
            return {'ok': False, 'reason': 'database_error'}
        finally:
            if cursor:
                cursor.close()

    def record_check_out(self, employee_id, checkout_at=None):
        """Save one checkout and elapsed work duration for a previously checked-in employee."""
        cursor = None
        try:
            checkout_at = checkout_at or datetime.now()
            today = checkout_at.date()
            checkout_time = checkout_at.time().replace(microsecond=0)
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT arrival_time, departure_time FROM attendance "
                "WHERE employee_id = %s AND attendance_date = %s",
                (employee_id, today)
            )
            row = cursor.fetchone()
            if not row:
                return {'ok': False, 'reason': 'no_checkin'}
            arrival_time, departure_time = row
            if departure_time is not None:
                return {'ok': False, 'reason': 'already_checked_out'}

            arrival_dt = datetime.combine(today, arrival_time)
            checkout_dt = datetime.combine(today, checkout_time)
            if checkout_dt < arrival_dt:
                return {'ok': False, 'reason': 'checkout_before_checkin'}
            duration_seconds = int((checkout_dt - arrival_dt).total_seconds())
            cursor.execute(
                "UPDATE attendance SET departure_time = %s, "
                "working_duration_seconds = %s "
                "WHERE employee_id = %s AND attendance_date = %s "
                "AND departure_time IS NULL",
                (checkout_time, duration_seconds, employee_id, today)
            )
            if cursor.rowcount != 1:
                self.connection.rollback()
                return {'ok': False, 'reason': 'already_checked_out'}
            self.connection.commit()
            hours, remainder = divmod(duration_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return {'ok': True, 'time': str(checkout_time),
                    'duration': f"{hours:02d}:{minutes:02d}:{seconds:02d}",
                    'duration_seconds': duration_seconds}
        except Error as e:
            if self.connection:
                self.connection.rollback()
            print(f"Error recording check-out: {e}")
            return {'ok': False, 'reason': 'database_error'}
        finally:
            if cursor:
                cursor.close()

    def get_pending_checkouts(self, attendance_date=None):
        attendance_date = attendance_date or date.today()
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT employee_id FROM attendance "
                "WHERE attendance_date = %s AND departure_time IS NULL",
                (attendance_date,)
            )
            rows = cursor.fetchall()
            cursor.close()
            return {row[0] for row in rows}
        except Error as e:
            print(f"Error fetching pending check-outs: {e}")
            return set()

    def get_daily_attendance(self, attendance_date=None):
        attendance_date = attendance_date or date.today()
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(
                "SELECT e.employee_id, e.employee_name, a.arrival_time, "
                "a.departure_time, a.working_duration_seconds, a.status "
                "FROM employees e LEFT JOIN attendance a "
                "ON a.employee_id = e.employee_id AND a.attendance_date = %s "
                "WHERE e.is_active = TRUE ORDER BY e.employee_name",
                (attendance_date,)
            )
            records = cursor.fetchall()
            cursor.close()
            schedule = self.get_attendance_schedule()
            cutoff = datetime.strptime(schedule['checkin_cutoff'], '%H:%M:%S').time()
            absent_is_final = datetime.now().time() > cutoff
            result = []
            for row in records:
                status = row['status'].upper() if row['arrival_time'] else (
                    'ABSENT' if absent_is_final else 'PENDING'
                )
                seconds = row['working_duration_seconds']
                duration = None
                if seconds is not None:
                    hours, remainder = divmod(int(seconds), 3600)
                    minutes, secs = divmod(remainder, 60)
                    duration = f"{hours:02d}:{minutes:02d}:{secs:02d}"
                result.append({
                    'employee_id': row['employee_id'],
                    'employee_name': row['employee_name'],
                    'arrival_time': str(row['arrival_time']) if row['arrival_time'] else '-',
                    'departure_time': str(row['departure_time']) if row['departure_time'] else '-',
                    'working_duration': duration or '-',
                    'status': status
                })
            return result
        except Error as e:
            print(f"Error getting daily attendance: {e}")
            return []
