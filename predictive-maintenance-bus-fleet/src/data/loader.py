
import pandas as pd
from sqlalchemy import create_engine, text
from src.utils.config import Config

class DataLoader:
    def __init__(self):
        self.engine = create_engine(
            f"postgresql+psycopg2://{Config.PG_USER}:{Config.PG_PASS}@"
            f"{Config.PG_HOST}:{Config.PG_PORT}/{Config.PG_DB}"
        )

    def load(self, sql: str) -> pd.DataFrame:
        with self.engine.connect() as con:
            return pd.read_sql_query(text(sql), con, parse_dates=["date"])
