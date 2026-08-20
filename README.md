# Employee Attendance System

Face-recognition attendance tracking for a workplace: employees are enrolled
once from reference photographs, then check in by looking at a webcam. Built
with **InsightFace**, **OpenCV**, **MySQL**, and a **PyQt5** desktop
interface.

## Overview

Badge- and PIN-based attendance systems share one weakness: the credential can
be handed to somebody else. Face recognition ties the attendance record to the
person rather than to a token they carry.

Each employee is enrolled with three reference images stored in MySQL. At
check-in the system extracts an embedding from the live camera frame, compares
it against the enrolled set, and writes a dated attendance row. Repeat
check-ins on the same day update the existing row instead of creating
duplicates.

## Features

- Employee enrolment with three reference photographs per person
- Live webcam recognition with InsightFace embeddings
- One attendance row per employee per day, enforced in the schema
- Daily report showing present and absent employees side by side
- CSV export of any day's report
- Admin dialog for managing enrolled employees
- Embedding cache on disk so recognition does not re-encode every start

## Architecture

```mermaid
flowchart LR
    A[Webcam frame] --> B[OpenCV capture]
    B --> C[InsightFace<br/>embedding]
    C --> D{Match against<br/>enrolled set}
    D -->|match| E[Record attendance]
    E --> F[(MySQL)]
    F --> G[Daily report / CSV]
    H[Enrolment photos] --> C
    C --> I[embeddings_insightface.pkl<br/>cache]
    I --> D
```

## Project structure

```text
attendance_app_enhanced.py   PyQt5 application, camera loop, admin dialogs
database_helper.py           MySQL connection, employees and attendance tables
insightface_embeddings.py    Embedding extraction, matching, and disk cache
schema.sql                   Database schema
requirements.txt             Python dependencies
```

## Requirements

- Python 3.8 or higher
- MySQL Server 8.0 or higher
- Webcam
- 4 GB RAM minimum

## Installation

Create and activate a virtual environment:

```bash
python -m venv attendance_env
```

```bash
attendance_env\Scripts\activate
```

On Linux or macOS use `source attendance_env/bin/activate` instead.

Install the dependencies:

```bash
pip install -r requirements.txt
```

Create the database and tables:

```bash
mysql -u root -p < schema.sql
```

## Configuration

The application reads its database settings from the environment and falls
back to local development defaults:

| Variable | Default |
|---|---|
| `ATTENDANCE_DB_HOST` | `localhost` |
| `ATTENDANCE_DB_USER` | `root` |
| `ATTENDANCE_DB_PASSWORD` | `1234` |
| `ATTENDANCE_DB_NAME` | `attend` |

Set a real password before running this anywhere other than a local machine:

```bash
export ATTENDANCE_DB_PASSWORD='your-mysql-password'
```

On Windows PowerShell use `$env:ATTENDANCE_DB_PASSWORD = 'your-mysql-password'`.

## Usage

```bash
python attendance_app_enhanced.py
```

InsightFace downloads its model weights on first run. Enrol employees from the
admin dialog before starting recognition — the system can only match faces it
has been given reference images for.

## Limitations

- Recognition thresholds are fixed constants rather than being calibrated
  against a labelled evaluation set.
- No liveness detection: a printed photograph or a phone screen held to the
  camera is not distinguished from a real face. This is a meaningful
  limitation for an attendance system and would need addressing before real
  deployment.
- Reference images are stored as BLOBs in MySQL with no encryption at rest.
- The admin dialog has no authentication of its own; it assumes a trusted
  operator on a trusted machine.
- Tested on Windows with a local MySQL instance.

## License

[MIT](LICENSE)
