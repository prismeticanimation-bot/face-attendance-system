import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import sys
import csv
import tempfile
from datetime import date, datetime, time, timedelta
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QLineEdit, QDialog, QFileDialog, QMessageBox, 
                             QTableWidget, QTableWidgetItem, QGroupBox, QFormLayout, QTimeEdit,
                             QCheckBox, QListWidget, QListWidgetItem, QScrollArea, QHeaderView)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QTime, QSize, QRect
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QIcon, QPainter, QBrush
import cv2
import numpy as np
from database_helper import DatabaseHelper
from insightface_embeddings import InsightFaceEmbeddingExtractor

DARK_BG = "#18191A"
DARK_SECONDARY = "#242526"
DARK_TERTIARY = "#3A3B3C"
ACCENT_BLUE = "#0A66C2"
ACCENT_BLUE_HOVER = "#0952A4"
TEXT_PRIMARY = "#E4E6EB"
TEXT_SECONDARY = "#B0B3B9"
SUCCESS_GREEN = "#31A24C"
ERROR_RED = "#E4163A"
WARNING_ORANGE = "#F57C00"


def create_camera_icon(size=400):
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(DARK_SECONDARY))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    camera_width = size * 0.6
    camera_height = size * 0.5
    x = (size - camera_width) / 2
    y = (size - camera_height) / 2 + size * 0.05

    painter.setBrush(QBrush(QColor(DARK_TERTIARY)))
    painter.setPen(Qt.NoPen)
    painter.drawRoundedRect(int(x), int(y), int(camera_width), int(camera_height), 20, 20)

    lens_size = size * 0.25
    lens_x = (size - lens_size) / 2
    lens_y = y + (camera_height - lens_size) / 2
    painter.setBrush(QBrush(QColor(ACCENT_BLUE)))
    painter.drawEllipse(int(lens_x), int(lens_y), int(lens_size), int(lens_size))

    inner_lens = lens_size * 0.6
    inner_x = lens_x + (lens_size - inner_lens) / 2
    inner_y = lens_y + (lens_size - inner_lens) / 2
    painter.setBrush(QBrush(QColor(DARK_BG)))
    painter.drawEllipse(int(inner_x), int(inner_y), int(inner_lens), int(inner_lens))

    painter.end()
    return pixmap


class ImageViewerDialog(QDialog):
    
    def __init__(self, parent=None, image_path=None, image_data=None):
        super().__init__(parent)
        self.setWindowTitle("View Photo")
        self.setFixedSize(500, 500)
        self.setStyleSheet(f"""
            QDialog {{ background-color: {DARK_BG}; }}
        """)
        
        layout = QVBoxLayout()
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(f"border: 2px solid {ACCENT_BLUE}; border-radius: 5px;")
        
        if image_path and os.path.exists(image_path):
            pixmap = QPixmap(image_path)
        elif image_data is not None:
            # Convert numpy array to QPixmap
            rgb_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width
            q_img = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
        else:
            pixmap = QPixmap()
            
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(450, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
        else:
            self.image_label.setText("No Image Available")
            self.image_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 14px;")
        
        layout.addWidget(self.image_label)
        
        close_btn = QPushButton("Close")
        close_btn.setFont(QFont("Segoe UI", 10))
        close_btn.setMinimumHeight(35)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {DARK_TERTIARY};
                color: white;
                border: none;
                border-radius: 5px;
            }}
            QPushButton:hover {{ background-color: #4A4B4C; }}
        """)
        close_btn.clicked.connect(self.close)
        
        layout.addWidget(close_btn)
        self.setLayout(layout)


class InsightFaceCameraThread(QThread):
    frame_ready = pyqtSignal(object)
    face_recognized = pyqtSignal(str, str, float, dict)
    unknown_face = pyqtSignal()

    def __init__(self, extractor, embeddings_data, mode="checkin", eligible_ids=None):
        super().__init__()
        self.extractor = extractor
        self.embeddings_data = embeddings_data
        self.mode = mode
        self.eligible_ids = set(eligible_ids) if eligible_ids is not None else None
        self.running = False
        self.camera = None
        self.emitted_ids = set()
        self.unknown_notified = False

    def run(self):
        self.camera = cv2.VideoCapture(0, cv2.CAP_MSMF)
        self.running = True
        if not self.camera.isOpened():
            self.running = False
            self.frame_ready.emit(None)
            return
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                continue
            employee_id, similarity, employee_name, face_info = (
                self.extractor.recognize_face_from_embedding(frame, self.embeddings_data)
            )
            if employee_id and similarity > self.extractor.threshold:
                self.unknown_notified = False
                allowed = self.eligible_ids is None or employee_id in self.eligible_ids
                if allowed and employee_id not in self.emitted_ids:
                    self.emitted_ids.add(employee_id)
                    self.face_recognized.emit(employee_id, employee_name, float(similarity),
                                               face_info if face_info else {})
            elif face_info is not None and not self.unknown_notified:
                self.unknown_notified = True
                self.unknown_face.emit()
            elif face_info is None:
                self.unknown_notified = False
            self.frame_ready.emit(frame)
        if self.camera:
            self.camera.release()
            self.camera = None

    def stop(self):
        self.running = False
        self.wait()


class AdminLoginDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Admin Login")
        self.setFixedSize(400, 220)
        self.setStyleSheet(f"""
            QDialog {{ background-color: {DARK_BG}; }}
            QMessageBox {{ background-color: {DARK_BG}; }}
            QMessageBox QLabel {{ color: {TEXT_PRIMARY}; }}
        """)
        self.authenticated = False
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Admin Login")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {TEXT_PRIMARY};")
        layout.addWidget(title)

        pass_label = QLabel("Password:")
        pass_label.setStyleSheet(f"color: {TEXT_PRIMARY};")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setFont(QFont("Segoe UI", 11))
        self.password_input.setMinimumHeight(40)
        self.password_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {DARK_SECONDARY};
                color: {TEXT_PRIMARY};
                border: 1px solid {DARK_TERTIARY};
                border-radius: 5px;
                padding: 8px;
            }}
            QLineEdit:focus {{ border: 2px solid {ACCENT_BLUE}; }}
        """)
        self.password_input.returnPressed.connect(self.check_password)
        layout.addWidget(pass_label)
        layout.addWidget(self.password_input)

        login_btn = QPushButton("Login")
        login_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        login_btn.setMinimumHeight(40)
        login_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_BLUE};
                color: white;
                border: none;
                border-radius: 5px;
            }}
            QPushButton:hover {{ background-color: {ACCENT_BLUE_HOVER}; }}
        """)
        login_btn.clicked.connect(self.check_password)
        layout.addWidget(login_btn)
        layout.addStretch()
        self.setLayout(layout)

    def check_password(self):
        if self.password_input.text() == "1234":
            self.authenticated = True
            self.accept()
        else:
            msg = QMessageBox(self)
            msg.setWindowTitle("Error")
            msg.setText("Incorrect password")
            msg.setStyleSheet(f"""
                QMessageBox {{ background-color: {DARK_BG}; }}
                QMessageBox QLabel {{ color: {TEXT_PRIMARY}; }}
                QPushButton {{ 
                    color: {TEXT_PRIMARY}; 
                    background-color: {ACCENT_BLUE};
                    border: none;
                    border-radius: 3px;
                    padding: 5px;
                    min-width: 50px;
                }}
            """)
            msg.exec_()
            self.password_input.clear()


class EditEmployeeDialog(QDialog):

    def __init__(self, parent=None, db_helper=None, extractor=None, employee_id=None, employee_name=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Employee")
        self.setMinimumSize(600, 500)
        self.setStyleSheet(f"""
            QDialog {{ background-color: {DARK_BG}; }}
            QMessageBox {{ background-color: {DARK_BG}; }}
            QMessageBox QLabel {{ color: {TEXT_PRIMARY}; }}
        """)
        self.db_helper = db_helper
        self.extractor = extractor
        self.employee_id = employee_id
        self.image_paths = [None, None, None]
        self._setup_ui(employee_name)

    def _setup_ui(self, employee_name):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Edit Employee")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {TEXT_PRIMARY};")
        layout.addWidget(title)

        form_layout = QFormLayout()

        id_label = QLabel("Employee ID:")
        id_label.setStyleSheet(f"color: {TEXT_PRIMARY};")
        self.id_input = QLineEdit()
        self.id_input.setText(self.employee_id)
        self.id_input.setReadOnly(True)
        self.id_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {DARK_TERTIARY};
                color: {TEXT_SECONDARY};
                border: 1px solid {DARK_TERTIARY};
                border-radius: 5px;
                padding: 8px;
            }}
        """)
        form_layout.addRow(id_label, self.id_input)

        name_label = QLabel("Employee Name:")
        name_label.setStyleSheet(f"color: {TEXT_PRIMARY};")
        self.name_input = QLineEdit()
        self.name_input.setText(employee_name)
        self.name_input.setFont(QFont("Segoe UI", 11))
        self.name_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {DARK_SECONDARY};
                color: {TEXT_PRIMARY};
                border: 1px solid {DARK_TERTIARY};
                border-radius: 5px;
                padding: 8px;
            }}
        """)
        form_layout.addRow(name_label, self.name_input)
        layout.addLayout(form_layout)

        photos_label = QLabel("Update Photos (Select 3 or leave empty):")
        photos_label.setStyleSheet(f"color: {TEXT_PRIMARY};")
        layout.addWidget(photos_label)

        self.photos_display = QLabel("No photos selected")
        self.photos_display.setAlignment(Qt.AlignCenter)
        self.photos_display.setStyleSheet(f"color: {TEXT_SECONDARY}; padding: 10px;")
        layout.addWidget(self.photos_display)

        browse_btn = QPushButton("Select 3 Photos")
        browse_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        browse_btn.setMinimumHeight(40)
        browse_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_BLUE};
                color: white;
                border: none;
                border-radius: 5px;
            }}
        """)
        browse_btn.clicked.connect(self.browse_images)
        layout.addWidget(browse_btn)

        layout.addStretch()

        buttons_layout = QHBoxLayout()
        save_btn = QPushButton("Save Changes")
        save_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        save_btn.setMinimumHeight(40)
        save_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {SUCCESS_GREEN};
                color: white;
                border: none;
                border-radius: 5px;
            }}
        """)
        save_btn.clicked.connect(self.save_changes)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        cancel_btn.setMinimumHeight(40)
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {DARK_TERTIARY};
                color: white;
                border: none;
                border-radius: 5px;
            }}
        """)
        cancel_btn.clicked.connect(self.reject)

        buttons_layout.addWidget(save_btn)
        buttons_layout.addWidget(cancel_btn)
        layout.addLayout(buttons_layout)

        self.setLayout(layout)

    def browse_images(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select 3 Employee Photos", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )

        if file_paths and len(file_paths) != 3:
            msg = QMessageBox(self)
            msg.setWindowTitle("Error")
            msg.setText("Please select exactly 3 photos")
            msg.setStyleSheet(f"""
                QMessageBox {{ background-color: {DARK_BG}; }}
                QMessageBox QLabel {{ color: {TEXT_PRIMARY}; }}
                QPushButton {{ 
                    color: {TEXT_PRIMARY}; 
                    background-color: {ACCENT_BLUE};
                    border: none;
                    border-radius: 3px;
                    padding: 5px;
                    min-width: 50px;
                }}
            """)
            msg.exec_()
            return

        if file_paths:
            self.image_paths = file_paths
            names = [p.split("/")[-1] for p in file_paths]
            self.photos_display.setText("\n".join([f"{i+1}. {n}" for i, n in enumerate(names)]))
            self.photos_display.setStyleSheet(f"color: {SUCCESS_GREEN}; padding: 10px;")

    def save_changes(self):
        new_name = self.name_input.text().strip()
        if not new_name:
            msg = QMessageBox(self)
            msg.setWindowTitle("Error")
            msg.setText("Please enter employee name")
            msg.setStyleSheet(f"""
                QMessageBox {{ background-color: {DARK_BG}; }}
                QMessageBox QLabel {{ color: {TEXT_PRIMARY}; }}
                QPushButton {{ 
                    color: {TEXT_PRIMARY}; 
                    background-color: {ACCENT_BLUE};
                    border: none;
                    border-radius: 3px;
                    padding: 5px;
                    min-width: 50px;
                }}
            """)
            msg.exec_()
            return

        try:
            cursor = self.db_helper.connection.cursor()
            cursor.execute("UPDATE employees SET employee_name = %s WHERE employee_id = %s", 
                         (new_name, self.employee_id))
            self.db_helper.connection.commit()
            cursor.close()

            if any(p is not None for p in self.image_paths):
                if self.db_helper.add_employee(self.employee_id, new_name, 
                                              self.image_paths[0], self.image_paths[1], 
                                              self.image_paths[2]):
                    if self.extractor:
                        self.extractor.embeddings_data = self.extractor.extract_embeddings_for_all_employees()
                        self.extractor.save_embeddings(self.extractor.embeddings_data)

            msg = QMessageBox(self)
            msg.setWindowTitle("Success")
            msg.setText("Employee updated successfully!")
            msg.setStyleSheet(f"""
                QMessageBox {{ background-color: {DARK_BG}; }}
                QMessageBox QLabel {{ color: {TEXT_PRIMARY}; }}
                QPushButton {{ 
                    color: {TEXT_PRIMARY}; 
                    background-color: {ACCENT_BLUE};
                    border: none;
                    border-radius: 3px;
                    padding: 5px;
                    min-width: 50px;
                }}
            """)
            msg.exec_()
            self.accept()
        except Exception as e:
            msg = QMessageBox(self)
            msg.setWindowTitle("Error")
            msg.setText(str(e))
            msg.setStyleSheet(f"""
                QMessageBox {{ background-color: {DARK_BG}; }}
                QMessageBox QLabel {{ color: {TEXT_PRIMARY}; }}
                QPushButton {{ 
                    color: {TEXT_PRIMARY}; 
                    background-color: {ACCENT_BLUE};
                    border: none;
                    border-radius: 3px;
                    padding: 5px;
                    min-width: 50px;
                }}
            """)
            msg.exec_()


class ViewAllEmployeesDialog(QDialog):

    def __init__(self, parent=None, db_helper=None, extractor=None, on_update=None):
        super().__init__(parent)
        self.setWindowTitle("Employee Management - Full Screen")
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        width = int(screen_geometry.width() * 0.95)
        height = int(screen_geometry.height() * 0.90)
        self.setGeometry(50, 50, width, height)
        self.setStyleSheet(f"""
            QDialog {{ background-color: {DARK_BG}; }}
            QMessageBox {{ background-color: {DARK_BG}; }}
            QMessageBox QLabel {{ color: {TEXT_PRIMARY}; }}
        """)
        self.db_helper = db_helper
        self.extractor = extractor
        self.on_update = on_update
        self.count_label = None
        self.employees_table = None
        self.selected_employee = None
        self.search_input = None
        self.last_selected_row = -1
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        title = QLabel("Employee Management System")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {TEXT_PRIMARY};")
        layout.addWidget(title)

        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        search_label.setStyleSheet(f"color: {TEXT_PRIMARY};")
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search by name or ID...")
        self.search_input.setFont(QFont("Segoe UI", 11))
        self.search_input.setMinimumHeight(40)
        self.search_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {DARK_SECONDARY};
                color: {TEXT_PRIMARY};
                border: 2px solid {DARK_TERTIARY};
                border-radius: 8px;
                padding: 10px;
                font-size: 11px;
            }}
            QLineEdit:focus {{ border: 2px solid {ACCENT_BLUE}; }}
        """)
        self.search_input.textChanged.connect(self.search_employees)

        clear_btn = QPushButton("✕ Clear")
        clear_btn.setFont(QFont("Segoe UI", 10, QFont.Bold))
        clear_btn.setMaximumWidth(110)
        clear_btn.setMinimumHeight(40)
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ERROR_RED};
                color: white;
                border: none;
                border-radius: 5px;
            }}
            QPushButton:hover {{ background-color: #D41830; }}
        """)
        clear_btn.clicked.connect(self.clear_search)

        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_input, 1)
        search_layout.addWidget(clear_btn)
        search_layout.addStretch()
        layout.addLayout(search_layout)

        table = QTableWidget()
        table.setColumnCount(4) 
        table.setHorizontalHeaderLabels(["Employee ID", "Employee Name", "Photos", "Select"])
        table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {DARK_SECONDARY};
                color: {TEXT_PRIMARY};
                gridline-color: {DARK_TERTIARY};
            }}
            QHeaderView::section {{
                background-color: {DARK_TERTIARY};
                color: {TEXT_PRIMARY};
                padding: 10px;
                font-weight: bold;
                border: none;
            }}
            QTableWidget::item {{
                padding: 10px;
            }}
            QTableWidget::item:selected {{
                background-color: {ACCENT_BLUE};
            }}
        """)
        table.horizontalHeader().setStretchLastSection(False)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QTableWidget.SingleSelection)
        table.itemSelectionChanged.connect(self.on_employee_selected)

        self.employees_table = table
        self.count_label = QLabel("0")
        self.count_label.setStyleSheet(f"color: {SUCCESS_GREEN}; font-weight: bold; font-size: 12px;")
        self.load_employees()
        layout.addWidget(table, 1)

        status_layout = QHBoxLayout()
        status_label = QLabel("Total Employees:")
        status_label.setStyleSheet(f"color: {TEXT_PRIMARY}; font-weight: bold;")
        status_layout.addWidget(status_label)
        status_layout.addWidget(self.count_label)
        status_layout.addStretch()
        layout.addLayout(status_layout)

        action_layout = QHBoxLayout()
        action_layout.setSpacing(10)

        self.edit_btn = QPushButton("✎ Edit Selected")
        self.edit_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.edit_btn.setMinimumHeight(50)
        self.edit_btn.setMinimumWidth(180)
        self.edit_btn.setEnabled(False)
        self.edit_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_BLUE};
                color: white;
                border: none;
                border-radius: 8px;
            }}
            QPushButton:hover:enabled {{ background-color: {ACCENT_BLUE_HOVER}; }}
            QPushButton:disabled {{
                background-color: {DARK_TERTIARY};
            }}
        """)
        self.edit_btn.clicked.connect(self.edit_selected_employee)

        self.delete_btn = QPushButton("🗑 Delete Selected")
        self.delete_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.delete_btn.setMinimumHeight(50)
        self.delete_btn.setMinimumWidth(180)
        self.delete_btn.setEnabled(False)
        self.delete_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ERROR_RED};
                color: white;
                border: none;
                border-radius: 8px;
            }}
            QPushButton:hover:enabled {{ background-color: #D41830; }}
            QPushButton:disabled {{
                background-color: {DARK_TERTIARY};
            }}
        """)
        self.delete_btn.clicked.connect(self.delete_employee)

        action_layout.addWidget(self.edit_btn)
        action_layout.addWidget(self.delete_btn)
        action_layout.addStretch()
        layout.addLayout(action_layout)

        close_btn = QPushButton("✕ Close")
        close_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        close_btn.setMinimumHeight(50)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {DARK_TERTIARY};
                color: white;
                border: none;
                border-radius: 8px;
            }}
            QPushButton:hover {{ background-color: {DARK_TERTIARY}; }}
        """)
        close_btn.clicked.connect(self.reject)
        layout.addWidget(close_btn)

        self.setLayout(layout)

    def load_employees(self):
        self.employees_table.setRowCount(0)
        employees = self.db_helper.get_all_employees()

        for row, emp in enumerate(employees):
            self.employees_table.insertRow(row)

            id_item = QTableWidgetItem(emp['employee_id'])
            id_item.setFont(QFont("Segoe UI", 10))
            id_item.setForeground(QColor(ACCENT_BLUE))
            self.employees_table.setItem(row, 0, id_item)

            name_item = QTableWidgetItem(emp['employee_name'])
            name_item.setFont(QFont("Segoe UI", 10))
            self.employees_table.setItem(row, 1, name_item)

            photos_widget = QWidget()
            photos_layout = QHBoxLayout(photos_widget)
            photos_layout.setSpacing(5)
            photos_layout.setContentsMargins(5, 5, 5, 5)
            
            images = self.db_helper.get_employee_images(emp['employee_id'])
            if images:
                for i, img in enumerate(images[:3]):  
                    if img is not None:
                        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb_image.shape
                        bytes_per_line = ch * w
                        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(qt_image)
                        
                   
                        scaled_pixmap = pixmap.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        
                        photo_label = QLabel()
                        photo_label.setPixmap(scaled_pixmap)
                        photo_label.setStyleSheet(f"border: 1px solid {ACCENT_BLUE}; border-radius: 3px;")
                        photo_label.setCursor(Qt.PointingHandCursor)
                        photo_label.mousePressEvent = lambda event, img_data=img, emp_name=emp['employee_name']: self.view_photo(img_data, emp_name)
                        photos_layout.addWidget(photo_label)
            
            photos_layout.addStretch()
            self.employees_table.setCellWidget(row, 2, photos_widget)

            checkbox = QCheckBox()
            checkbox.setProperty("employee_id", emp['employee_id'])
            self.employees_table.setCellWidget(row, 3, checkbox)

        if self.count_label:
            self.count_label.setText(str(len(employees)))

        if self.last_selected_row >= 0 and self.last_selected_row < self.employees_table.rowCount():
            self.employees_table.selectRow(self.last_selected_row)

    def view_photo(self, image_data, employee_name):
        dialog = ImageViewerDialog(self, image_data=image_data)
        dialog.setWindowTitle(f"Photo - {employee_name}")
        dialog.exec_()

    def on_employee_selected(self):
        if self.employees_table.currentRow() >= 0:
            self.last_selected_row = self.employees_table.currentRow()
            self.selected_employee = self.last_selected_row
            self.edit_btn.setEnabled(True)
            self.delete_btn.setEnabled(True)
        else:
            self.selected_employee = None
            self.edit_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)

    def search_employees(self):
        search_text = self.search_input.text().lower().strip()
        if not search_text:
            self.load_employees()
            return

        self.employees_table.setRowCount(0)
        employees = self.db_helper.get_all_employees()
        count = 0

        for emp in employees:
            if search_text not in emp['employee_name'].lower() and search_text not in emp['employee_id'].lower():
                continue

            row = self.employees_table.rowCount()
            self.employees_table.insertRow(row)

            id_item = QTableWidgetItem(emp['employee_id'])
            id_item.setForeground(QColor(ACCENT_BLUE))
            self.employees_table.setItem(row, 0, id_item)

            name_item = QTableWidgetItem(emp['employee_name'])
            self.employees_table.setItem(row, 1, name_item)

            # Photos
            photos_widget = QWidget()
            photos_layout = QHBoxLayout(photos_widget)
            photos_layout.setSpacing(5)
            photos_layout.setContentsMargins(5, 5, 5, 5)
            
            images = self.db_helper.get_employee_images(emp['employee_id'])
            if images:
                for i, img in enumerate(images[:3]):
                    if img is not None:
                        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb_image.shape
                        bytes_per_line = ch * w
                        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(qt_image)
                        scaled_pixmap = pixmap.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        
                        photo_label = QLabel()
                        photo_label.setPixmap(scaled_pixmap)
                        photo_label.setStyleSheet(f"border: 1px solid {ACCENT_BLUE}; border-radius: 3px;")
                        photo_label.setCursor(Qt.PointingHandCursor)
                        photo_label.mousePressEvent = lambda event, img_data=img, emp_name=emp['employee_name']: self.view_photo(img_data, emp_name)
                        photos_layout.addWidget(photo_label)
            
            photos_layout.addStretch()
            self.employees_table.setCellWidget(row, 2, photos_widget)

            checkbox = QCheckBox()
            checkbox.setProperty("employee_id", emp['employee_id'])
            self.employees_table.setCellWidget(row, 3, checkbox)
            count += 1

        if self.count_label:
            self.count_label.setText(str(count))

    def clear_search(self):
        self.search_input.clear()
        self.load_employees()

    def edit_selected_employee(self):
        selected_count = 0
        selected_employee = None
        
        for row in range(self.employees_table.rowCount()):
            checkbox = self.employees_table.cellWidget(row, 3)
            if checkbox and checkbox.isChecked():
                selected_count += 1
                if selected_count == 1:
                    selected_employee = row
        
        if selected_count == 0:
            msg = QMessageBox(self)
            msg.setWindowTitle("Warning")
            msg.setText("Please select an employee to edit")
            msg.setStyleSheet(f"""
                QMessageBox {{ background-color: {DARK_BG}; }}
                QMessageBox QLabel {{ color: {TEXT_PRIMARY}; }}
                QPushButton {{ 
                    color: {TEXT_PRIMARY}; 
                    background-color: {ACCENT_BLUE};
                    border: none;
                    border-radius: 3px;
                    padding: 5px;
                    min-width: 50px;
                }}
            """)
            msg.exec_()
            return
            
        if selected_count > 1:
            msg = QMessageBox(self)
            msg.setWindowTitle("Warning")
            msg.setText("Please select only one employee to edit")
            msg.setStyleSheet(f"""
                QMessageBox {{ background-color: {DARK_BG}; }}
                QMessageBox QLabel {{ color: {TEXT_PRIMARY}; }}
                QPushButton {{ 
                    color: {TEXT_PRIMARY}; 
                    background-color: {ACCENT_BLUE};
                    border: none;
                    border-radius: 3px;
                    padding: 5px;
                    min-width: 50px;
                }}
            """)
            msg.exec_()
            return

        if selected_employee is not None:
            emp_id = self.employees_table.item(selected_employee, 0).text()
            emp_name = self.employees_table.item(selected_employee, 1).text()

            dialog = EditEmployeeDialog(self, self.db_helper, self.extractor, emp_id, emp_name)
            if dialog.exec_() == QDialog.Accepted:
                self.load_employees()
                if self.on_update:
                    self.on_update()

    def delete_employee(self):
        if self.employees_table.currentRow() < 0:
            return

        row = self.employees_table.currentRow()
        emp_id = self.employees_table.item(row, 0).text()
        emp_name = self.employees_table.item(row, 1).text()

        msg = QMessageBox(self)
        msg.setWindowTitle("Confirm Delete")
        msg.setText(f"Delete {emp_name}?\nThis will also delete all attendance records.")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setStyleSheet(f"""
            QMessageBox {{ background-color: {DARK_BG}; }}
            QMessageBox QLabel {{ color: {TEXT_PRIMARY}; }}
            QPushButton {{ 
                color: {TEXT_PRIMARY}; 
                background-color: {ACCENT_BLUE};
                border: none;
                border-radius: 3px;
                padding: 5px;
                min-width: 50px;
            }}
        """)

        if msg.exec_() != QMessageBox.Yes:
            return

        try:
            cursor = self.db_helper.connection.cursor()
            cursor.execute("DELETE FROM attendance WHERE employee_id = %s", (emp_id,))
            cursor.execute("DELETE FROM employees WHERE employee_id = %s", (emp_id,))
            self.db_helper.connection.commit()
            cursor.close()

            if self.extractor:
                self.extractor.embeddings_data = self.extractor.extract_embeddings_for_all_employees()
                self.extractor.save_embeddings(self.extractor.embeddings_data)

            msg = QMessageBox(self)
            msg.setWindowTitle("Success")
            msg.setText(f"{emp_name} deleted successfully!")
            msg.setStyleSheet(f"""
                QMessageBox {{ background-color: {DARK_BG}; }}
                QMessageBox QLabel {{ color: {TEXT_PRIMARY}; }}
                QPushButton {{ 
                    color: {TEXT_PRIMARY}; 
                    background-color: {ACCENT_BLUE};
                    border: none;
                    border-radius: 3px;
                    padding: 5px;
                    min-width: 50px;
                }}
            """)
            msg.exec_()
            self.load_employees()
            if self.on_update:
                self.on_update()
        except Exception as e:
            msg = QMessageBox(self)
            msg.setWindowTitle("Error")
            msg.setText(str(e))
            msg.setStyleSheet(f"""
                QMessageBox {{ background-color: {DARK_BG}; }}
                QMessageBox QLabel {{ color: {TEXT_PRIMARY}; }}
                QPushButton {{ 
                    color: {TEXT_PRIMARY}; 
                    background-color: {ACCENT_BLUE};
                    border: none;
                    border-radius: 3px;
                    padding: 5px;
                    min-width: 50px;
                }}
            """)
            msg.exec_()


class DeadlineSettingsDialog(QDialog):

    def __init__(self, parent=None, deadline_time=None):
        super().__init__(parent)
        self.setWindowTitle("Set Deadline")
        self.setMinimumSize(450, 300)
        self.setStyleSheet(f"QDialog {{ background-color: {DARK_BG}; }}")
        self.deadline_time = deadline_time or time(12, 0)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Set Attendance Deadline")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {TEXT_PRIMARY};")
        layout.addWidget(title)

        time_label = QLabel("Set Time:")
        time_label.setStyleSheet(f"color: {TEXT_PRIMARY};")
        self.time_edit = QTimeEdit()
        self.time_edit.setTime(QTime(self.deadline_time.hour, self.deadline_time.minute))
        self.time_edit.setFont(QFont("Segoe UI", 11))
        self.time_edit.setMinimumHeight(35)
        layout.addWidget(time_label)
        layout.addWidget(self.time_edit)

        buttons_layout = QHBoxLayout()
        save_btn = QPushButton("✓ Save")
        save_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        save_btn.setMinimumHeight(40)
        save_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {SUCCESS_GREEN};
                color: white;
                border: none;
                border-radius: 5px;
            }}
        """)
        save_btn.clicked.connect(self.save_deadline)

        cancel_btn = QPushButton("✕ Cancel")
        cancel_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        cancel_btn.setMinimumHeight(40)
        cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {DARK_TERTIARY};
                color: white;
                border: none;
                border-radius: 5px;
            }}
        """)
        cancel_btn.clicked.connect(self.reject)

        buttons_layout.addWidget(save_btn)
        buttons_layout.addWidget(cancel_btn)
        layout.addLayout(buttons_layout)
        self.setLayout(layout)

    def save_deadline(self):
        qt_time = self.time_edit.time()
        self.deadline_time = time(qt_time.hour(), qt_time.minute())
        self.accept()


class EnrollmentCameraThread(QThread):
    frame_ready = pyqtSignal(object, object, float)
    camera_error = pyqtSignal(str)

    def __init__(self, extractor):
        super().__init__()
        self.extractor = extractor
        self.running = False
        self.camera = None

    def run(self):
        self.camera = cv2.VideoCapture(0, cv2.CAP_MSMF)
        if not self.camera.isOpened():
            self.camera.release()
            self.camera = None
            self.camera_error.emit("Camera could not be opened. Check that it is not being used by another app.")
            return

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.running = True
        frame_count = 0
        last_bbox = None
        last_score = 0.0

        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                self.msleep(30)
                continue

            frame_count += 1
            if frame_count % 3 == 0:
                try:
                    faces = self.extractor.app.get(frame)
                    if faces:
                        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                        last_bbox = tuple(int(v) for v in face.bbox)
                        last_score = float(face.det_score)
                    else:
                        last_bbox = None
                        last_score = 0.0
                except Exception:
                    last_bbox = None
                    last_score = 0.0

            self.frame_ready.emit(frame.copy(), last_bbox, last_score)
            self.msleep(30)

        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def stop(self):
        self.running = False
        if self.isRunning():
            self.wait(4000)


class AddEmployeeDialog(QDialog):

    def __init__(self, parent=None, db_helper=None, extractor=None):
        super().__init__(parent)
        self.setWindowTitle("Live Face Enrollment - Add Employee")
        self.setMinimumSize(760, 650)
        self.setStyleSheet(f"""
            QDialog {{ background-color: {DARK_BG}; }}
            QMessageBox {{ background-color: {DARK_BG}; }}
            QMessageBox QLabel {{ color: {TEXT_PRIMARY}; }}
        """)

        self.db_helper = db_helper
        self.extractor = extractor
        self.camera_thread = None
        self.latest_frame = None
        self.latest_bbox = None
        self.latest_score = 0.0
        self.captured_frames = []
        self.capture_stage = 0
        self.stage_prompts = [
            "1/3 — Look straight at the camera",
            "2/3 — Turn your head slightly to the LEFT",
            "3/3 — Turn your head slightly to the RIGHT"
        ]
        self.stage_timer = QTimer(self)
        self.stage_timer.setSingleShot(True)
        self.stage_timer.timeout.connect(self.capture_current_stage)

        self._setup_ui()
        QTimer.singleShot(300, self.start_camera)

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(20, 16, 20, 16)

        title = QLabel("Register New Employee")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {TEXT_PRIMARY};")
        layout.addWidget(title)

        form_layout = QFormLayout()
        form_layout.setSpacing(10)

        self.employee_id_input = QLineEdit()
        self.employee_id_input.setPlaceholderText("Example: EMP002")
        self.employee_name_input = QLineEdit()
        self.employee_name_input.setPlaceholderText("Employee full name")

        field_style = f"""
            QLineEdit {{
                background-color: {DARK_SECONDARY};
                color: {TEXT_PRIMARY};
                border: 1px solid {DARK_TERTIARY};
                border-radius: 5px;
                padding: 8px;
            }}
        """
        self.employee_id_input.setStyleSheet(field_style)
        self.employee_name_input.setStyleSheet(field_style)

        id_label = QLabel("Employee ID:")
        name_label = QLabel("Employee Name:")
        id_label.setStyleSheet(f"color: {TEXT_PRIMARY};")
        name_label.setStyleSheet(f"color: {TEXT_PRIMARY};")
        form_layout.addRow(id_label, self.employee_id_input)
        form_layout.addRow(name_label, self.employee_name_input)
        layout.addLayout(form_layout)

        self.preview_label = QLabel("Opening camera...")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFixedSize(640, 360)
        self.preview_label.setStyleSheet(
            f"background-color: {DARK_SECONDARY}; color: {TEXT_SECONDARY}; "
            f"border: 1px solid {DARK_TERTIARY}; border-radius: 8px;"
        )
        preview_row = QHBoxLayout()
        preview_row.addStretch()
        preview_row.addWidget(self.preview_label)
        preview_row.addStretch()
        layout.addLayout(preview_row)

        self.status_label = QLabel("Keep one face visible in the camera.")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {TEXT_PRIMARY}; padding: 5px;")
        layout.addWidget(self.status_label)

        self.progress_label = QLabel("Captured samples: 0 / 3")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        layout.addWidget(self.progress_label)

        action_layout = QHBoxLayout()
        self.camera_button = QPushButton("Restart Camera")
        self.camera_button.clicked.connect(self.start_camera)
        self.scan_button = QPushButton("Start Face Scan")
        self.scan_button.setEnabled(False)
        self.scan_button.clicked.connect(self.begin_enrollment)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        self.camera_button.setMinimumHeight(40)
        self.scan_button.setMinimumHeight(40)
        self.cancel_button.setMinimumHeight(40)
        self.camera_button.setStyleSheet(
            f"QPushButton {{ background-color: {ACCENT_BLUE}; color: white; border: none; "
            f"border-radius: 5px; padding: 8px; }}"
        )
        self.scan_button.setStyleSheet(
            f"QPushButton {{ background-color: {SUCCESS_GREEN}; color: white; border: none; "
            f"border-radius: 5px; padding: 8px; }}"
        )
        self.cancel_button.setStyleSheet(
            f"QPushButton {{ background-color: {DARK_TERTIARY}; color: white; border: none; "
            f"border-radius: 5px; padding: 8px; }}"
        )
        action_layout.addWidget(self.camera_button)
        action_layout.addWidget(self.scan_button)
        action_layout.addWidget(self.cancel_button)
        layout.addLayout(action_layout)

        self.setLayout(layout)

    def start_camera(self):
        if self.camera_thread is not None and self.camera_thread.isRunning():
            self.camera_thread.stop()

        if self.extractor is None or not hasattr(self.extractor, "app"):
            self.status_label.setText("Face recognition model is not available.")
            return

        self.latest_frame = None
        self.camera_thread = EnrollmentCameraThread(self.extractor)
        self.camera_thread.frame_ready.connect(self.update_camera_frame)
        self.camera_thread.camera_error.connect(self.show_camera_error)
        self.camera_thread.start()
        self.status_label.setText("Camera starting — position your face in the frame.")

    def update_camera_frame(self, frame, bbox, score):
        self.latest_frame = frame
        self.latest_bbox = bbox
        self.latest_score = score

        display_frame = frame.copy()
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 220, 100), 2)
            cv2.putText(
                display_frame, f"Face detected ({score:.2f})",
                (max(10, x1), max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 100), 2
            )

        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(
            rgb.data, width, height, channels * width, QImage.Format_RGB888
        ).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.preview_label.setPixmap(pixmap)

        if bbox is not None and score >= 0.60:
            self.scan_button.setEnabled(not self.stage_timer.isActive() and len(self.captured_frames) == 0)
            if len(self.captured_frames) == 0 and not self.stage_timer.isActive():
                self.status_label.setText("Face detected. Enter employee details, then start the scan.")
        else:
            self.scan_button.setEnabled(False)
            if len(self.captured_frames) == 0 and not self.stage_timer.isActive():
                self.status_label.setText("No clear face detected. Face the camera with good lighting.")

    def show_camera_error(self, message):
        self.scan_button.setEnabled(False)
        self.preview_label.setText("Camera unavailable")
        self.status_label.setText(message)

    def begin_enrollment(self):
        emp_id = self.employee_id_input.text().strip()
        emp_name = self.employee_name_input.text().strip()

        if not emp_id or not emp_name:
            QMessageBox.warning(self, "Missing details", "Please enter Employee ID and Employee Name first.")
            return

        if self.latest_frame is None or self.latest_bbox is None or self.latest_score < 0.60:
            QMessageBox.warning(self, "Face not detected", "Keep one clear face visible before starting enrollment.")
            return

        self.captured_frames = []
        self.capture_stage = 0
        self.scan_button.setEnabled(False)
        self.camera_button.setEnabled(False)
        self.employee_id_input.setEnabled(False)
        self.employee_name_input.setEnabled(False)
        self.capture_next_stage()

    def capture_next_stage(self):
        if self.capture_stage >= 3:
            self.save_employee()
            return

        self.status_label.setText(
            self.stage_prompts[self.capture_stage] +
            "\nHold this position. Capturing automatically in about 2 seconds..."
        )
        self.stage_timer.start(2200)

    def capture_current_stage(self):
        if self.latest_frame is None or self.latest_bbox is None or self.latest_score < 0.60:
            self.status_label.setText("Face not clear enough. Keep your face visible; trying again shortly...")
            self.stage_timer.start(700)
            return

        x1, y1, x2, y2 = self.latest_bbox
        if (x2 - x1) < 100 or (y2 - y1) < 100:
            self.status_label.setText("Move a little closer to the camera. Retrying...")
            self.stage_timer.start(700)
            return

        frame = self.latest_frame.copy()
        embedding, _ = self.extractor.extract_face_embedding(frame)
        if embedding is None:
            self.status_label.setText("Could not extract a clear face sample. Please hold still...")
            self.stage_timer.start(700)
            return

        self.captured_frames.append(frame)
        self.capture_stage += 1
        self.progress_label.setText(f"Captured samples: {len(self.captured_frames)} / 3")

        if self.capture_stage < 3:
            self.capture_next_stage()
        else:
            self.status_label.setText("All 3 samples captured. Saving employee and creating face embeddings...")
            QTimer.singleShot(200, self.save_employee)

    def save_employee(self):
        emp_id = self.employee_id_input.text().strip()
        emp_name = self.employee_name_input.text().strip()

        if len(self.captured_frames) != 3:
            QMessageBox.warning(self, "Enrollment incomplete", "Three face samples are required.")
            return

        temp_dir = None
        temp_paths = []
        try:
            temp_dir = tempfile.mkdtemp(prefix="attendance_enrollment_")
            for index, frame in enumerate(self.captured_frames, start=1):
                image_path = os.path.join(temp_dir, f"face_{index}.jpg")
                if not cv2.imwrite(image_path, frame):
                    raise RuntimeError(f"Could not save captured face sample {index}.")
                temp_paths.append(image_path)

            added = self.db_helper.add_employee(
                emp_id, emp_name, temp_paths[0], temp_paths[1], temp_paths[2]
            )
            if not added:
                raise RuntimeError("Employee could not be added. Check whether this Employee ID already exists.")

            self.extractor.embeddings_data = self.extractor.extract_embeddings_for_all_employees()
            self.extractor.save_embeddings(self.extractor.embeddings_data)

            QMessageBox.information(
                self, "Enrollment successful",
                f"{emp_name} ({emp_id}) has been registered using live camera samples."
            )
            self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Enrollment error", str(e))
            self.scan_button.setEnabled(True)
            self.camera_button.setEnabled(True)
            self.employee_id_input.setEnabled(True)
            self.employee_name_input.setEnabled(True)
        finally:
            for path in temp_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass
            if temp_dir:
                try:
                    os.rmdir(temp_dir)
                except OSError:
                    pass

    def _stop_camera(self):
        self.stage_timer.stop()
        if self.camera_thread is not None and self.camera_thread.isRunning():
            self.camera_thread.stop()
        self.camera_thread = None

    def accept(self):
        self._stop_camera()
        super().accept()

    def reject(self):
        self._stop_camera()
        super().reject()

    def closeEvent(self, event):
        self._stop_camera()
        super().closeEvent(event)

class AttendanceSystemGUI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Employee Attendance System - Face Recognition")
        self.setGeometry(50, 50, 1400, 950)
        self.setStyleSheet(f"QMainWindow {{ background-color: {DARK_BG}; }}")

        self.present_label = None
        self.absent_label = None
        self.clock_label = None
        self.countdown_label = None
        self.camera_label = None
        self.recognition_label = None
        self.start_btn = None
        self.stop_btn = None
        self.admin_btn = None

        self.db_helper = DatabaseHelper()
        if not self.db_helper.connect():
            msg = QMessageBox(self)
            msg.setWindowTitle("Error")
            msg.setText("Failed to connect to database")
            msg.setStyleSheet(f"""
                QMessageBox {{ background-color: {DARK_BG}; }}
                QMessageBox QLabel {{ color: {TEXT_PRIMARY}; }}
                QPushButton {{ 
                    color: {TEXT_PRIMARY}; 
                    background-color: {ACCENT_BLUE};
                    border: none;
                    border-radius: 3px;
                    padding: 5px;
                    min-width: 50px;
                }}
            """)
            msg.exec_()
            sys.exit(1)

        print("Loading face recognition model...")
        self.extractor = InsightFaceEmbeddingExtractor(self.db_helper)
        
        self.embeddings_data = self.extractor.load_embeddings()
        if self.embeddings_data is None or len(self.embeddings_data) == 0:
            print("Extracting embeddings from database...")
            self.embeddings_data = self.extractor.extract_embeddings_for_all_employees()
            if self.embeddings_data and len(self.embeddings_data) > 0:
                self.extractor.save_embeddings(self.embeddings_data)
                print(f"Successfully extracted embeddings for {len(self.embeddings_data)} employees")
            else:
                print("Warning: No embeddings could be extracted from database!")
                self.embeddings_data = {}
        else:
            print(f"Loaded embeddings for {len(self.embeddings_data)} employees")
        
        self.extractor.embeddings_data = self.embeddings_data

        self.session_daily_records = {}
        self.camera_thread = None
        self.status_timer = None
        self.schedule = self.db_helper.get_attendance_schedule()
        self.checkin_time = datetime.strptime(self.schedule['checkin_time'], '%H:%M:%S').time()
        self.checkin_cutoff = datetime.strptime(self.schedule['checkin_cutoff'], '%H:%M:%S').time()
        self.checkout_start = datetime.strptime(self.schedule['checkout_start'], '%H:%M:%S').time()
        self.camera_mode = None
        self.checkout_auto_started_date = None
        self.checkout_completed = False
        self.deadline_time = self.checkin_cutoff
        self.deadline_set = True
        self.admin_dialog = None
        self.current_table = None

        self.setup_ui()
        self.start_btn.setEnabled(datetime.now().time() <= self.checkin_cutoff)
        self.setup_timers()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        header_layout = QHBoxLayout()
        self.clock_label = QLabel()
        self.clock_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.clock_label.setStyleSheet(f"color: {TEXT_PRIMARY};")

        self.countdown_label = QLabel()
        self.countdown_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.countdown_label.setStyleSheet(f"color: {TEXT_SECONDARY};")

        self.update_clock()
        header_layout.addWidget(self.clock_label)
        header_layout.addStretch()
        header_layout.addWidget(self.countdown_label)
        main_layout.addLayout(header_layout)

        title = QLabel("Employee Attendance System")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {TEXT_PRIMARY};")
        main_layout.addWidget(title)

        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(800, 600)
        self.camera_label.setStyleSheet(f"border: 2px solid {ACCENT_BLUE}; border-radius: 5px;")
        self.update_camera_icon()
        main_layout.addWidget(self.camera_label)

        self.recognition_label = QLabel("")
        self.recognition_label.setAlignment(Qt.AlignCenter)
        self.recognition_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.recognition_label.setStyleSheet(f"color: {SUCCESS_GREEN}; padding: 10px;")
        main_layout.addWidget(self.recognition_label)

        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(10)

        self.present_label = QLabel("Present: 0")
        self.present_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.present_label.setAlignment(Qt.AlignCenter)
        self.present_label.setMinimumHeight(50)
        self.present_label.setStyleSheet(f"""
            background-color: {DARK_SECONDARY};
            border: 2px solid {SUCCESS_GREEN};
            border-radius: 5px;
            padding: 10px;
            color: {TEXT_PRIMARY};
        """)

        self.absent_label = QLabel("Absent: 0")
        self.absent_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.absent_label.setAlignment(Qt.AlignCenter)
        self.absent_label.setMinimumHeight(50)
        self.absent_label.setStyleSheet(f"""
            background-color: {DARK_SECONDARY};
            border: 2px solid {ERROR_RED};
            border-radius: 5px;
            padding: 10px;
            color: {TEXT_PRIMARY};
        """)

        stats_layout.addWidget(self.present_label)
        stats_layout.addWidget(self.absent_label)
        main_layout.addLayout(stats_layout)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)

        self.start_btn = QPushButton("▶ Start Recognition")
        self.start_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.start_btn.setMinimumHeight(50)
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {SUCCESS_GREEN};
                color: white;
                border: none;
                border-radius: 8px;
            }}
            QPushButton:hover:enabled {{ background-color: #2A7F3B; }}
            QPushButton:disabled {{ background-color: {DARK_TERTIARY}; }}
        """)
        self.start_btn.clicked.connect(self.start_recognition)

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.stop_btn.setMinimumHeight(50)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ERROR_RED};
                color: white;
                border: none;
                border-radius: 8px;
            }}
        """)
        self.stop_btn.clicked.connect(self.stop_recognition)

        self.admin_btn = QPushButton("⚙ Admin Panel")
        self.admin_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.admin_btn.setMinimumHeight(50)
        self.admin_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_BLUE};
                color: white;
                border: none;
                border-radius: 8px;
            }}
        """)
        self.admin_btn.clicked.connect(self.open_admin_panel)

        buttons_layout.addWidget(self.start_btn)
        buttons_layout.addWidget(self.stop_btn)
        buttons_layout.addWidget(self.admin_btn)
        main_layout.addLayout(buttons_layout)

        central_widget.setLayout(main_layout)

    def update_camera_icon(self):
        icon_pixmap = create_camera_icon(600)
        self.camera_label.setPixmap(icon_pixmap)

    def setup_timers(self):
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)

    def is_deadline_passed(self):
        return datetime.now().time() > self.checkin_cutoff

    def get_time_remaining(self):
        now = datetime.now()
        cutoff_dt = datetime.combine(now.date(), self.checkin_cutoff)
        if now >= cutoff_dt:
            return "Check-in closed"
        remaining = cutoff_dt - now
        hours, remainder = divmod(int(remaining.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def update_clock(self):
        now = datetime.now()
        self.clock_label.setText(f"{now.strftime('%H:%M:%S')} - {now.strftime('%d/%m/%Y')}")
        if now.time() <= self.checkin_cutoff:
            self.countdown_label.setText(f"Check-in closes in: {self.get_time_remaining()}")
            self.countdown_label.setStyleSheet(f"color: {SUCCESS_GREEN};")
        else:
            self.countdown_label.setText("Check-in closed | Check-out starts at " +
                                         self.checkout_start.strftime('%H:%M'))
            self.countdown_label.setStyleSheet(f"color: {WARNING_ORANGE};")
        if self.camera_thread and self.camera_thread.running and self.camera_mode == "checkin":
            if now.time() > self.checkin_cutoff:
                self.stop_recognition()
        if (hasattr(self, 'start_btn') and self.start_btn is not None
                and now.time() >= self.checkout_start
                and self.checkout_auto_started_date != now.date()):
            self.checkout_auto_started_date = now.date()
            self.start_checkout_scan(silent=True)
        if now.second == 0:
            self.update_stats()


    def update_stats(self):
        # Check if UI elements are initialized
        if not hasattr(self, 'present_label') or self.present_label is None:
            return
            
        employees = self.db_helper.get_all_employees()
        today = date.today()

        present_count = 0
        absent_count = 0

        for emp in employees:
            try:
                cursor = self.db_helper.connection.cursor()
                cursor.execute("""SELECT arrival_time FROM attendance 
                               WHERE employee_id = %s AND attendance_date = %s""", 
                             (emp['employee_id'], today))
                result = cursor.fetchone()
                cursor.close()

                if result:
                    present_count += 1
                else:
                    if self.is_deadline_passed():
                        absent_count += 1
            except:
                pass

        self.present_label.setText(f"Present: {present_count}")
        if self.is_deadline_passed():
            self.absent_label.setText(f"Absent: {absent_count}")
        else:
            self.absent_label.setText(f"Absent: 0")

    def start_recognition(self):
        if datetime.now().time() > self.checkin_cutoff:
            QMessageBox.information(self, "Check-in Closed",
                "Check-in is closed after 9:30 AM. Evening check-out starts automatically at 6:00 PM.")
            return
        self._start_camera_scan("checkin")

    def start_checkout_scan(self, silent=False):
        pending_ids = self.db_helper.get_pending_checkouts(date.today())
        if not pending_ids:
            if not silent:
                QMessageBox.information(self, "No Pending Check-outs",
                                        "There are no employees waiting for check-out today.")
            return
        if self.camera_thread and self.camera_thread.running:
            if self.camera_mode == "checkout":
                return
            self.stop_recognition()
        self._start_camera_scan("checkout", pending_ids=pending_ids)

    def _start_camera_scan(self, mode, pending_ids=None):
        if self.camera_thread and self.camera_thread.running:
            return
        self.camera_mode = mode
        self.checkout_completed = False
        eligible = pending_ids if mode == "checkout" else None
        self.camera_thread = InsightFaceCameraThread(
            self.extractor, self.embeddings_data, mode=mode, eligible_ids=eligible
        )
        self.camera_thread.frame_ready.connect(self.update_frame)
        self.camera_thread.face_recognized.connect(self.record_attendance)
        self.camera_thread.unknown_face.connect(self.show_unknown_person)
        self.camera_thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.recognition_label.setText("Scanning for check-ins..." if mode == "checkin" else
                                       f"Evening check-out scan active | {len(eligible or [])} pending")
        self.recognition_label.setStyleSheet(f"color: {SUCCESS_GREEN}; padding: 10px;")

    def stop_recognition(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None

        self.update_camera_icon()
        self.camera_mode = None
        self.start_btn.setEnabled(datetime.now().time() <= self.checkin_cutoff)
        self.stop_btn.setEnabled(False)
        self.recognition_label.setText("Camera Stopped")
        self.recognition_label.setStyleSheet(f"color: {ERROR_RED}; padding: 10px;")

    def update_frame(self, frame):
        if frame is None:
            self.recognition_label.setText("Camera could not be opened. Check camera connection.")
            self.recognition_label.setStyleSheet(f"color: {ERROR_RED}; padding: 10px;")
            self.stop_recognition()
            return
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_label.setPixmap(scaled)

    def record_attendance(self, employee_id, employee_name, similarity, face_info):
        print(f"Face recognized: {employee_name} ({employee_id}) similarity={similarity:.3f}")
        if self.camera_mode == "checkout":
            result = self.db_helper.record_check_out(employee_id)
            if result.get('ok'):
                message = (f"✓ Check-out Registered\n{employee_name}\n"
                           f"{result['time']} | Working: {result['duration']}")
            else:
                message = f"Check-out not saved: {result.get('reason', 'unknown')}"
                if result.get('reason') == 'no_checkin':
                    message = f"No check-in found for {employee_name}"
        else:
            if datetime.now().time() > self.checkin_cutoff:
                message = "Check-in closed after 9:30 AM"
            else:
                result = self.db_helper.record_check_in(employee_id)
                if result.get('ok'):
                    message = (f"✓ {result['status'].upper()} Check-in\n"
                               f"{employee_name}\n{result['time']}")
                else:
                    message = {
                        'checkin_closed': "Check-in closed after 9:30 AM",
                        'already_checked_in': f"{employee_name} already checked in",
                        'unknown_employee': "UNKNOWN PERSON / NOT REGISTERED"
                    }.get(result.get('reason'), "Attendance could not be saved")
        color = SUCCESS_GREEN if message.startswith("✓") else WARNING_ORANGE
        self.recognition_label.setText(message)
        self.recognition_label.setStyleSheet(
            f"color: {color}; padding: 10px; font-weight: bold; font-size: 14px;"
        )
        if self.status_timer:
            self.status_timer.stop()
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.clear_recognition)
        self.status_timer.start(4000)
        self.update_stats()
        if self.current_table:
            self.populate_table_data(self.current_table)
        if self.camera_mode == "checkout":
            remaining = self.db_helper.get_pending_checkouts(date.today())
            if not remaining:
                # Keep camera preview live after the scan completes; recognition is disabled
                # for further check-outs by clearing the eligible employee IDs.
                self.checkout_completed = True
                if self.camera_thread:
                    self.camera_thread.eligible_ids = set()
                self.recognition_label.setText(
                    f"✓ All check-outs completed for today\nLast: {employee_name} | {result['time']}"
                )
                self.recognition_label.setStyleSheet(
                    f"color: {SUCCESS_GREEN}; padding: 10px; font-weight: bold; font-size: 14px;"
                )
                if self.status_timer:
                    self.status_timer.stop()
            elif self.camera_thread:
                self.camera_thread.eligible_ids = set(remaining)


    def show_unknown_person(self):
        self.recognition_label.setText("UNKNOWN PERSON / NOT REGISTERED")
        self.recognition_label.setStyleSheet(
            f"color: {WARNING_ORANGE}; padding: 10px; font-weight: bold; font-size: 14px;"
        )
        if self.status_timer:
            self.status_timer.stop()
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.clear_recognition)
        self.status_timer.start(2500)

    def clear_recognition(self):
        if self.checkout_completed and self.camera_mode == "checkout":
            self.recognition_label.setText("✓ All check-outs completed for today | Camera live")
        else:
            self.recognition_label.setText("Recognizing faces...")
        self.recognition_label.setStyleSheet(f"color: {SUCCESS_GREEN}; padding: 10px;")
        if self.status_timer:
            self.status_timer.stop()

    def open_admin_panel(self):
        dialog = AdminLoginDialog(self)
        if dialog.exec_() == QDialog.Accepted and dialog.authenticated:
            self.show_admin_panel()

    def show_admin_panel(self):
        self.admin_dialog = QDialog(self)
        self.admin_dialog.setWindowTitle("Admin Panel")
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        width = int(screen_geometry.width() * 0.95)
        height = int(screen_geometry.height() * 0.90)
        self.admin_dialog.setGeometry(50, 50, width, height)
        self.admin_dialog.setStyleSheet(f"""
            QDialog {{ background-color: {DARK_BG}; }}
            QMessageBox {{ background-color: {DARK_BG}; }}
            QMessageBox QLabel {{ color: {TEXT_PRIMARY}; }}
        """)

        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        title = QLabel("Admin Management Panel")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {TEXT_PRIMARY};")
        layout.addWidget(title)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)

        add_btn = QPushButton("➕ Add Employee")
        add_btn.setFont(QFont("Segoe UI", 10, QFont.Bold))
        add_btn.setMinimumHeight(45)
        add_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {SUCCESS_GREEN};
                color: white;
                border: none;
                border-radius: 8px;
            }}
            QPushButton:hover {{ background-color: #2A7F3B; }}
        """)
        add_btn.clicked.connect(self.add_new_employee)

        view_btn = QPushButton("👥 View All Employees")
        view_btn.setFont(QFont("Segoe UI", 10, QFont.Bold))
        view_btn.setMinimumHeight(45)
        view_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_BLUE};
                color: white;
                border: none;
                border-radius: 8px;
            }}
            QPushButton:hover {{ background-color: {ACCENT_BLUE_HOVER}; }}
        """)
        view_btn.clicked.connect(self.view_all_employees)

        delete_btn = QPushButton("🗑 Delete Attendance")
        delete_btn.setFont(QFont("Segoe UI", 10, QFont.Bold))
        delete_btn.setMinimumHeight(45)
        delete_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ERROR_RED};
                color: white;
                border: none;
                border-radius: 8px;
            }}
            QPushButton:hover {{ background-color: #D41830; }}
        """)
        delete_btn.clicked.connect(self.delete_selected_records)

        deadline_btn = QPushButton("⏰ Set Deadline")
        deadline_btn.setFont(QFont("Segoe UI", 10, QFont.Bold))
        deadline_btn.setMinimumHeight(45)
        deadline_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {WARNING_ORANGE};
                color: white;
                border: none;
                border-radius: 8px;
            }}
            QPushButton:hover {{ background-color: #E65C00; }}
        """)
        deadline_btn.clicked.connect(self.open_deadline_settings)

        buttons_layout.addWidget(add_btn)
        buttons_layout.addWidget(view_btn)
        buttons_layout.addWidget(delete_btn)
        buttons_layout.addWidget(deadline_btn)
        layout.addLayout(buttons_layout)

        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        search_label.setStyleSheet(f"color: {TEXT_PRIMARY};")
        search_input = QLineEdit()
        search_input.setPlaceholderText("Search by name or ID...")
        search_input.setFont(QFont("Segoe UI", 10))
        search_input.setMinimumHeight(35)
        search_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {DARK_SECONDARY};
                color: {TEXT_PRIMARY};
                border: 2px solid {DARK_TERTIARY};
                border-radius: 8px;
                padding: 8px;
            }}
            QLineEdit:focus {{ border: 2px solid {ACCENT_BLUE}; }}
        """)
        search_input.textChanged.connect(lambda: self.search_attendance_records(table, search_input))

        clear_btn = QPushButton("✕ Clear")
        clear_btn.setFont(QFont("Segoe UI", 10, QFont.Bold))
        clear_btn.setMaximumWidth(110)
        clear_btn.setMinimumHeight(35)
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ERROR_RED};
                color: white;
                border: none;
                border-radius: 5px;
            }}
            QPushButton:hover {{ background-color: #D41830; }}
        """)
        clear_btn.clicked.connect(lambda: self.clear_search_attendance(table, search_input))

        search_layout.addWidget(search_label)
        search_layout.addWidget(search_input, 1)
        search_layout.addWidget(clear_btn)
        layout.addLayout(search_layout)

        report_title = QLabel("Daily Attendance Report")
        report_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        report_title.setStyleSheet(f"color: {TEXT_PRIMARY};")
        layout.addWidget(report_title)

        date_label = QLabel(f"Date: {date.today().strftime('%d/%m/%Y')}")
        date_label.setFont(QFont("Segoe UI", 10))
        date_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        layout.addWidget(date_label)

        table = QTableWidget()
        table.setColumnCount(7)
        table.setHorizontalHeaderLabels([
            "Employee ID", "Name", "Check-in", "Check-out",
            "Working Duration", "Status", "Select"
        ])
        table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {DARK_SECONDARY};
                color: {TEXT_PRIMARY};
                gridline-color: {DARK_TERTIARY};
            }}
            QHeaderView::section {{
                background-color: {DARK_TERTIARY};
                color: {TEXT_PRIMARY};
                padding: 8px;
                font-weight: bold;
            }}
            QTableWidget::item {{
                padding: 8px;
            }}
        """)
        table.horizontalHeader().setStretchLastSection(True)
        for column in range(6):
            table.horizontalHeader().setSectionResizeMode(column, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeToContents)

        self.current_table = table
        self.populate_table_data(table)
        layout.addWidget(table, 1)

        stats_layout = QHBoxLayout()
        present_label = QLabel("Present: 0")
        present_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        present_label.setStyleSheet(f"""
            background-color: {DARK_SECONDARY};
            border: 2px solid {SUCCESS_GREEN};
            border-radius: 5px;
            padding: 10px;
            color: {TEXT_PRIMARY};
        """)

        absent_label = QLabel("Absent: 0")
        absent_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        absent_label.setStyleSheet(f"""
            background-color: {DARK_SECONDARY};
            border: 2px solid {ERROR_RED};
            border-radius: 5px;
            padding: 10px;
            color: {TEXT_PRIMARY};
        """)

        stats_layout.addWidget(present_label)
        stats_layout.addWidget(absent_label)
        layout.addLayout(stats_layout)

        self.update_admin_stats(table, present_label, absent_label)

        export_layout = QHBoxLayout()
        
        clear_all_btn = QPushButton("🗑️ Clear All Data")
        clear_all_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        clear_all_btn.setMinimumHeight(40)
        clear_all_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {WARNING_ORANGE};
                color: white;
                border: none;
                border-radius: 8px;
            }}
            QPushButton:hover {{ background-color: #E65C00; }}
        """)
        clear_all_btn.clicked.connect(lambda: self.clear_all_attendance(table, present_label, absent_label))

        export_btn = QPushButton("💾 Export to CSV")
        export_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        export_btn.setMinimumHeight(40)
        export_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_BLUE};
                color: white;
                border: none;
                border-radius: 8px;
            }}
            QPushButton:hover {{ background-color: {ACCENT_BLUE_HOVER}; }}
        """)
        export_btn.clicked.connect(self.export_to_csv)

        export_layout.addWidget(clear_all_btn)
        export_layout.addStretch()
        export_layout.addWidget(export_btn)
        layout.addLayout(export_layout)

        close_btn = QPushButton("✕ Close")
        close_btn.setFont(QFont("Segoe UI", 10, QFont.Bold))
        close_btn.setMinimumHeight(40)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {DARK_TERTIARY};
                color: white;
                border: none;
                border-radius: 8px;
            }}
        """)
        close_btn.clicked.connect(self.admin_dialog.reject)
        layout.addWidget(close_btn)

        self.admin_dialog.setLayout(layout)
        self.admin_dialog.exec_()

    def populate_table_data(self, table):
        records = self.db_helper.get_daily_attendance(date.today())
        self._populate_attendance_rows(table, records)

    def _populate_attendance_rows(self, table, records):
        table.setRowCount(len(records))
        for row, record in enumerate(records):
            values = [
                record['employee_id'], record['employee_name'],
                record['arrival_time'], record['departure_time'],
                record['working_duration'], record['status']
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 5:
                    if value == "PRESENT":
                        item.setBackground(QColor(49, 162, 76))
                    elif value == "LATE":
                        item.setBackground(QColor(245, 124, 0))
                    elif value == "ABSENT":
                        item.setBackground(QColor(228, 22, 58))
                table.setItem(row, column, item)
            checkbox = QCheckBox()
            checkbox.setProperty("employee_id", record['employee_id'])
            table.setCellWidget(row, 6, checkbox)

    def update_admin_stats(self, table, present_label, absent_label):
        records = self.db_helper.get_daily_attendance(date.today())
        present_count = sum(1 for row in records if row['status'] in ("PRESENT", "LATE"))
        absent_count = sum(1 for row in records if row['status'] == "ABSENT")
        present_label.setText(f"Present: {present_count}")
        absent_label.setText(f"Absent: {absent_count}")

    def search_attendance_records(self, table, search_input):
        search_text = search_input.text().lower().strip()
        records = self.db_helper.get_daily_attendance(date.today())
        if search_text:
            records = [
                row for row in records
                if search_text in row['employee_name'].lower()
                or search_text in row['employee_id'].lower()
            ]
        self._populate_attendance_rows(table, records)

    def clear_search_attendance(self, table, search_input):
        search_input.clear()
        self.populate_table_data(table)

    def delete_selected_records(self):
        if not self.current_table:
            return

        selected = []
        employee_names = []
        
        for row in range(self.current_table.rowCount()):
            checkbox = self.current_table.cellWidget(row, 6)
            if checkbox and checkbox.isChecked():
                emp_id = self.current_table.item(row, 0).text()
                emp_name = self.current_table.item(row, 1).text()
                selected.append(emp_id)
                employee_names.append(emp_name)

        if not selected:
            msg = QMessageBox(self.admin_dialog)
            msg.setWindowTitle("Warning")
            msg.setText("Select at least one employee")
            msg.setStyleSheet(f"""
                QMessageBox {{ background-color: {DARK_BG}; }}
                QMessageBox QLabel {{ color: {TEXT_PRIMARY}; }}
                QPushButton {{ 
                    color: {TEXT_PRIMARY}; 
                    background-color: {ACCENT_BLUE};
                    border: none;
                    border-radius: 3px;
                    padding: 5px;
                    min-width: 50px;
                }}
            """)
            msg.exec_()
            return

        names_text = "\n".join(employee_names)
        msg = QMessageBox(self.admin_dialog)
        msg.setWindowTitle("Confirm Delete")
        msg.setText(f"Are you sure you want to delete attendance records for:\n{names_text}?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setStyleSheet(f"""
            QMessageBox {{ background-color: {DARK_BG}; }}
            QMessageBox QLabel {{ color: {TEXT_PRIMARY}; }}
            QPushButton {{ 
                color: {TEXT_PRIMARY}; 
                background-color: {ACCENT_BLUE};
                border: none;
                border-radius: 3px;
                padding: 5px;
                min-width: 50px;
            }}
        """)

        if msg.exec_() != QMessageBox.Yes:
            return

        try:
            cursor = self.db_helper.connection.cursor()
            today = date.today()
            for emp_id in selected:
                cursor.execute("DELETE FROM attendance WHERE employee_id = %s AND attendance_date = %s", 
                             (emp_id, today))
            self.db_helper.connection.commit()
            cursor.close()

            msg = QMessageBox(self.admin_dialog)
            msg.setWindowTitle("Success")
            msg.setText("Attendance records deleted successfully!")
            msg.setStyleSheet(f"""
                QMessageBox {{ background-color: {DARK_BG}; }}
                QMessageBox QLabel {{ color: {TEXT_PRIMARY}; }}
                QPushButton {{ 
                    color: {TEXT_PRIMARY}; 
                    background-color: {ACCENT_BLUE};
                    border: none;
                    border-radius: 3px;
                    padding: 5px;
                    min-width: 50px;
                }}
            """)
            msg.exec_()
            
            self.populate_table_data(self.current_table)
            
            for i in range(self.admin_dialog.layout().count()):
                item = self.admin_dialog.layout().itemAt(i)
                if isinstance(item, QHBoxLayout):
                    if item.count() >= 2:
                        widget1 = item.itemAt(0).widget()
                        widget2 = item.itemAt(1).widget()
                        if isinstance(widget1, QLabel) and "Present" in widget1.text() and isinstance(widget2, QLabel) and "Absent" in widget2.text():
                            self.update_admin_stats(self.current_table, widget1, widget2)
                            break
            
            self.update_stats()
                
        except Exception as e:
            msg = QMessageBox(self.admin_dialog)
            msg.setWindowTitle("Error")
            msg.setText(str(e))
            msg.setStyleSheet(f"""
                QMessageBox {{ background-color: {DARK_BG}; }}
                QMessageBox QLabel {{ color: {TEXT_PRIMARY}; }}
                QPushButton {{ 
                    color: {TEXT_PRIMARY}; 
                    background-color: {ACCENT_BLUE};
                    border: none;
                    border-radius: 3px;
                    padding: 5px;
                    min-width: 50px;
                }}
            """)
            msg.exec_()
            
        

    def clear_all_attendance(self, table, present_label, absent_label):
        msg = QMessageBox(self.admin_dialog)
        msg.setWindowTitle("Clear All Data")
        msg.setText("Are you sure you want to clear ALL attendance data?\nThis will:\n• Delete all attendance records\n• Reset deadline\n• Keep employees data")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setStyleSheet(f"""
            QMessageBox {{ background-color: {DARK_BG}; }}
            QMessageBox QLabel {{ color: {TEXT_PRIMARY}; }}
            QPushButton {{ 
                color: {TEXT_PRIMARY}; 
                background-color: {ACCENT_BLUE};
                border: none;
                border-radius: 3px;
                padding: 5px;
                min-width: 50px;
            }}
        """)
    
        if msg.exec_() != QMessageBox.Yes:
            return
        
        try:
            cursor = self.db_helper.connection.cursor()
            cursor.execute("DELETE FROM attendance")
            self.db_helper.connection.commit()
            cursor.close()
        
            # Clearing today's attendance must not reset the saved office schedule.
            self.deadline_set = True
            self.deadline_time = self.checkin_cutoff
            self.start_btn.setEnabled(datetime.now().time() <= self.checkin_cutoff)
        
            self.update_clock()
            self.update_stats()
            
            self.update_admin_stats(table, present_label, absent_label)
            
            self.populate_table_data(table)
        
            msg = QMessageBox(self.admin_dialog)
            msg.setWindowTitle("Success")
            msg.setText("All attendance data cleared successfully!\nDeadline has been reset.")
            msg.setStyleSheet(f"""
                QMessageBox {{ background-color: {DARK_BG}; }}
                QMessageBox QLabel {{ color: {TEXT_PRIMARY}; }}
                QPushButton {{ 
                    color: {TEXT_PRIMARY}; 
                    background-color: {ACCENT_BLUE};
                    border: none;
                    border-radius: 3px;
                    padding: 5px;
                    min-width: 50px;
                }}
            """)
            msg.exec_()
            
        except Exception as e:
            msg = QMessageBox(self.admin_dialog)
            msg.setWindowTitle("Error")
            msg.setText(f"Error clearing data: {str(e)}")
            msg.setStyleSheet(f"""
                QMessageBox {{ background-color: {DARK_BG}; }}
                QMessageBox QLabel {{ color: {TEXT_PRIMARY}; }}
                QPushButton {{ 
                    color: {TEXT_PRIMARY}; 
                    background-color: {ACCENT_BLUE};
                    border: none;
                    border-radius: 3px;
                    padding: 5px;
                    min-width: 50px;
                }}
            """)
            msg.exec_()

    def restart_application(self):
        try:
            # Clean up resources properly
            if hasattr(self, 'camera_thread') and self.camera_thread:
                self.camera_thread.stop()
                self.camera_thread = None
            
            if hasattr(self, 'clock_timer') and self.clock_timer:
                self.clock_timer.stop()
            
            if hasattr(self, 'db_helper') and self.db_helper:
                self.db_helper.disconnect()
            
            # Create new instance
            new_window = AttendanceSystemGUI()
            new_window.show()
            
        except Exception as e:
            print(f"Error during restart: {e}")
            # Fallback: show error message and exit
            msg = QMessageBox()
            msg.setWindowTitle("Restart Error")
            msg.setText(f"Failed to restart application: {str(e)}")
            msg.exec_()

    def add_new_employee(self):
        dialog = AddEmployeeDialog(self.admin_dialog, self.db_helper, self.extractor)
        if dialog.exec_() == QDialog.Accepted:
            self.embeddings_data = self.extractor.embeddings_data
            self.populate_table_data(self.current_table)

    def view_all_employees(self):
        dialog = ViewAllEmployeesDialog(self.admin_dialog, self.db_helper, self.extractor, self.update_stats)
        dialog.exec_()

    def open_deadline_settings(self):
        dialog = DeadlineSettingsDialog(self.admin_dialog, self.checkin_cutoff)
        if dialog.exec_() == QDialog.Accepted:
            new_cutoff = dialog.deadline_time
            if new_cutoff <= self.checkin_time:
                QMessageBox.warning(
                    self.admin_dialog, "Invalid Check-in Cutoff",
                    "The check-in closing time must be later than the 9:00 AM on-time threshold."
                )
                return
            if not self.db_helper.save_checkin_cutoff(new_cutoff.strftime('%H:%M:%S')):
                QMessageBox.critical(
                    self.admin_dialog, "Schedule Save Failed",
                    "Could not save the check-in cutoff to the database."
                )
                return
            self.checkin_cutoff = new_cutoff
            self.deadline_time = new_cutoff
            self.deadline_set = True
            self.start_btn.setEnabled(datetime.now().time() <= self.checkin_cutoff)
            self.update_clock()
            self.update_stats()

    def export_to_csv(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self.admin_dialog, "Save Report",
            f"attendance_{date.today()}.csv", "CSV Files (*.csv)"
        )
        if not file_path:
            return
        try:
            records = self.db_helper.get_daily_attendance(date.today())
            with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Employee ID", "Name", "Check-in", "Check-out",
                    "Working Duration", "Status"
                ])
                for record in records:
                    writer.writerow([
                        record['employee_id'], record['employee_name'],
                        record['arrival_time'], record['departure_time'],
                        record['working_duration'], record['status']
                    ])
            QMessageBox.information(self.admin_dialog, "Success",
                                    "Attendance report exported successfully.")
        except Exception as e:
            QMessageBox.critical(self.admin_dialog, "Export Error", str(e))

    def closeEvent(self, event):
        if self.camera_thread:
            self.camera_thread.stop()
        self.clock_timer.stop()
        self.db_helper.disconnect()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = AttendanceSystemGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()