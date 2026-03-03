from sqlalchemy import create_engine, text

engine = create_engine(
    "postgresql+psycopg2://road_user:road_pw@localhost:5432/road_accidents", future=True
)

with engine.begin() as conn:
    result = conn.execute(text("SELECT 1"))
    print(result.fetchall())
