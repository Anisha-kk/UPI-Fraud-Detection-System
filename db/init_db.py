from db.database import engine
from db.schema import Base

Base.metadata.create_all(bind=engine)

print("Database tables created.")