import calendar
from datetime import date, datetime, time
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QTabWidget, QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox,
    QHeaderView
)
from PyQt5.QtCore import Qt
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


class MonthlyPayrollDialog(QDialog):
    """Monthly attendance and payroll summary; Sunday is the only weekly off."""

    def __init__(self, parent, db_helper):
        super().__init__(parent)
        self.db = db_helper
        self.setWindowTitle("Monthly Payroll & Attendance")
        self.resize(1100, 700)
        root = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Month:"))
        self.month = QComboBox()
        self.month.addItems([calendar.month_name[m] for m in range(1, 13)])
        self.month.setCurrentIndex(date.today().month - 1)
        controls.addWidget(self.month)
        controls.addWidget(QLabel("Year:"))
        self.year = QComboBox()
        years = list(range(date.today().year - 5, date.today().year + 2))
        self.year.addItems([str(y) for y in years])
        self.year.setCurrentText(str(date.today().year))
        controls.addWidget(self.year)
        load = QPushButton("Load Report")
        load.clicked.connect(self.load_report)
        controls.addWidget(load)
        root.addLayout(controls)

        self.tabs = QTabWidget()
        self.summary = QTableWidget(0, 7)
        self.summary.setHorizontalHeaderLabels([
            "Employee ID", "Employee Name", "Present", "Absent",
            "Late", "Early Departure", "Total Working Hours"
        ])
        self.details = QTableWidget(0, 7)
        self.details.setHorizontalHeaderLabels([
            "Date", "Employee ID", "Employee Name", "Check-in",
            "Check-out", "Working Hours", "Daily Status"
        ])
        for table in (self.summary, self.details):
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            table.horizontalHeader().setStretchLastSection(True)
            table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tabs.addTab(self.summary, "Monthly Totals")
        self.tabs.addTab(self.details, "Date-wise Details")
        root.addWidget(self.tabs)

        exports = QHBoxLayout()
        excel = QPushButton("Export Excel")
        excel.clicked.connect(self.export_excel)
        pdf = QPushButton("Export PDF")
        pdf.clicked.connect(self.export_pdf)
        exports.addStretch()
        exports.addWidget(excel)
        exports.addWidget(pdf)
        root.addLayout(exports)
        self.summary_rows = []
        self.detail_rows = []
        self.load_report()

    @staticmethod
    def _as_time(value):
        if value is None:
            return None
        if isinstance(value, time):
            return value
        return datetime.strptime(str(value), "%H:%M:%S").time()

    def load_report(self):
        year, month = int(self.year.currentText()), self.month.currentIndex() + 1
        last_day = calendar.monthrange(year, month)[1]
        first = date(year, month, 1)
        last = date(year, month, last_day)
        today = date.today()
        effective_last = min(last, today) if (year, month) == (today.year, today.month) else last
        working_dates = [
            date(year, month, day) for day in range(1, effective_last.day + 1)
            if date(year, month, day).weekday() != 6
        ]
        cursor = self.db.connection.cursor(dictionary=True)
        cursor.execute(
            """SELECT e.employee_id, e.employee_name, a.attendance_date,
                      a.arrival_time, a.departure_time, a.working_duration_seconds, a.status
               FROM employees e
               LEFT JOIN attendance a
                 ON a.employee_id=e.employee_id
                AND a.attendance_date BETWEEN %s AND %s
               WHERE e.is_active=TRUE
               ORDER BY e.employee_name, a.attendance_date""",
            (first, last)
        )
        rows = cursor.fetchall()
        cursor.close()
        employees = {}
        for row in rows:
            key = row["employee_id"]
            employees.setdefault(key, {"name": row["employee_name"], "days": {}})
            if row["attendance_date"]:
                employees[key]["days"][row["attendance_date"]] = row

        schedule = self.db.get_attendance_schedule()
        checkin_threshold = self._as_time(schedule.get("checkin_time", "09:00:00"))
        checkout_threshold = self._as_time(schedule.get("checkout_start", "18:00:00"))
        self.summary_rows, self.detail_rows = [], []
        for employee_id, item in employees.items():
            present = absent = late = early = total_seconds = 0
            for day in working_dates:
                record = item["days"].get(day)
                if not record or not record["arrival_time"]:
                    absent += 1
                    status, arrival, departure, duration = "ABSENT", "-", "-", "-"
                else:
                    present += 1
                    arrival_time = self._as_time(record["arrival_time"])
                    departure_time = self._as_time(record["departure_time"])
                    arrival = arrival_time.strftime("%H:%M:%S")
                    departure = departure_time.strftime("%H:%M:%S") if departure_time else "NOT CHECKED OUT"
                    seconds = int(record["working_duration_seconds"] or 0)
                    total_seconds += seconds
                    duration = f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}"
                    is_late = str(record["status"]).lower() == "late" or arrival_time > checkin_threshold
                    is_early = departure_time is not None and departure_time < checkout_threshold
                    if is_late:
                        late += 1
                    if is_early:
                        early += 1
                    status_parts = []
                    status_parts.append("LATE" if is_late else "PRESENT")
                    if is_early:
                        status_parts.append("EARLY DEPARTURE")
                    elif departure_time is None:
                        status_parts.append("CHECK-OUT PENDING")
                    status = " / ".join(status_parts)
                self.detail_rows.append([
                    day.isoformat(), str(employee_id), item["name"],
                    arrival, departure, duration, status
                ])
            total_hours = f"{total_seconds // 3600:02d}:{(total_seconds % 3600) // 60:02d}"
            self.summary_rows.append([
                str(employee_id), item["name"], present, absent, late, early, total_hours
            ])

        self._fill_table(self.summary, self.summary_rows)
        self._fill_table(self.details, self.detail_rows)

    @staticmethod
    def _fill_table(table, rows):
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                table.setItem(r, c, QTableWidgetItem(str(value)))

    def export_excel(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Monthly Payroll", "monthly_payroll.xlsx", "Excel Files (*.xlsx)"
        )
        if not path:
            return
        try:
            book = Workbook()
            summary = book.active
            summary.title = "Monthly Totals"
            summary.append(["Employee ID", "Employee Name", "Present", "Absent",
                            "Late", "Early Departure", "Total Working Hours"])
            for row in self.summary_rows:
                summary.append(row)
            detail = book.create_sheet("Date-wise Details")
            detail.append(["Date", "Employee ID", "Employee Name", "Check-in",
                           "Check-out", "Working Hours", "Daily Status"])
            for row in self.detail_rows:
                detail.append(row)
            for sheet in book.worksheets:
                sheet.freeze_panes = "A2"
                sheet.auto_filter.ref = sheet.dimensions
                for cell in sheet[1]:
                    cell.font = Font(bold=True, color="FFFFFF")
                    cell.fill = PatternFill("solid", fgColor="244062")
                    cell.alignment = Alignment(horizontal="center")
                for column in sheet.columns:
                    letter = column[0].column_letter
                    sheet.column_dimensions[letter].width = min(
                        32, max(14, max(len(str(c.value or "")) for c in column) + 2)
                    )
            book.save(path)
            QMessageBox.information(self, "Export Complete", "Excel payroll report saved.")
        except Exception as exc:
            QMessageBox.critical(self, "Excel Export Error", str(exc))

    def export_pdf(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Monthly Payroll", "monthly_payroll.pdf", "PDF Files (*.pdf)"
        )
        if not path:
            return
        try:
            styles = getSampleStyleSheet()
            story = [Paragraph("Monthly Payroll & Attendance", styles["Title"]),
                     Spacer(1, 12)]
            data = [["Employee ID", "Name", "Present", "Absent", "Late", "Early", "Hours"]]
            data.extend(self.summary_rows)
            table = Table(data, repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#244062")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.extend([Paragraph("Monthly Totals", styles["Heading2"]), table,
                          Spacer(1, 16),
                          Paragraph("Date-wise Details", styles["Heading2"])])
            detail_data = [["Date", "ID", "Name", "Check-in", "Check-out", "Hours", "Status"]]
            detail_data.extend(self.detail_rows)
            detail_table = Table(detail_data, repeatRows=1)
            detail_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#244062")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(detail_table)
            SimpleDocTemplate(path, pagesize=landscape(A4)).build(story)
            QMessageBox.information(self, "Export Complete", "PDF payroll report saved.")
        except Exception as exc:
            QMessageBox.critical(self, "PDF Export Error", str(exc))
