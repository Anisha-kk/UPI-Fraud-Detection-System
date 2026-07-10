#!/bin/bash
set -e

echo "Waiting for database..."

python - <<'PY'
import time
import os
from sqlalchemy import create_engine

url = os.environ["DATABASE_URL"]

for i in range(30):
    try:
        engine = create_engine(url)
        engine.connect()
        print("Database ready")
        break
    except Exception:
        print("Waiting...")
        time.sleep(2)
else:
    raise Exception("Database unavailable")
PY

echo "Creating database tables..."
python -m db.init_db

if [ $? -ne 0 ]; then
    echo "Database initialization failed"
    exit 1
fi

echo "Database initialization completed"

echo "Starting application..."
exec "$@"