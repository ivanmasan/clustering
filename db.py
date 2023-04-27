from pathlib import Path
from typing import Union

import yaml
from sshtunnel import SSHTunnelForwarder
from sqlalchemy import create_engine

from sqlalchemy import text
import pandas as pd
from string import Template


class TelemetryConnector:
    SSH_HOST = 'ai-cloud.photoneo.com'
    SSH_PORT = 22
    POSTGRES_REMOTE_PORT = 5432
    DATABASE_NAME = "telemetry"

    def __init__(self, user_name: str, password: str, ssh_user_name: str) -> None:
        self._user_name = user_name
        self._password = password
        self._ssh_user_name = ssh_user_name

    def fetch(self, query: str) -> pd.DataFrame:
        with SSHTunnelForwarder(
                (self.SSH_HOST, self.SSH_PORT),
                ssh_username=self._ssh_user_name,
                remote_bind_address=('127.0.0.1', self.POSTGRES_REMOTE_PORT)
        ) as server:
            conn_str = (f"postgresql://{self._user_name}:{self._password}@"
                        f"127.0.0.1:{server.local_bind_port}/{self.DATABASE_NAME}")
            engine = create_engine(conn_str)

            with engine.connect() as conn:
                return pd.read_sql(text(query), conn)


class QueryFetcher:
    def __init__(self, file_path: Union[Path, str], connector: TelemetryConnector) -> None:
        self._connector = connector

        with open(str(file_path), 'r') as f:
            self._queries = yaml.safe_load(f)

    def fetch(self, query_name: str, **kwargs):
        try:
            query = self._queries[query_name]
        except KeyError:
            raise ValueError(f"No query under name {query_name}. "
                             f"Available queries {list(self._queries.keys())}")

        template = Template(query)
        query = template.safe_substitute(kwargs)

        return self._connector.fetch(query)


def get_query_fetcher():
    connector = TelemetryConnector(
        user_name="postgres",
        password="Photoneo2022",
        ssh_user_name="root")

    return QueryFetcher(
        file_path=Path('clustering_queries.yaml'),
        connector=connector
    )
