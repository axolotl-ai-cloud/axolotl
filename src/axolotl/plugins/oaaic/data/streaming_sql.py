import os
from typing import Callable, Generator, Tuple

import psycopg
import psycopg.conninfo


def pgsql(pgsql_table=None, id_field="id", **kwargs) -> Callable:
    pgsql_conn = os.environ.get("PGSQL_CONN", None)
    if not pgsql_conn:
        raise ValueError("missing PGSQL_CONN environment variable")
    conn_dict = psycopg.conninfo.conninfo_to_dict(pgsql_conn)

    def data_generator() -> Generator[Tuple, None, None]:
        with psycopg.connect(**conn_dict) as conn:
            with conn.cursor() as cur:
                page_size = 10
                last_id = None
                while True:
                    if last_id:
                        where_clause = f" WHERE {id_field} > {last_id}"
                    cur.execute(
                        f"SELECT * FROM {pgsql_table}{where_clause} ORDER BY {id_field} ASC LIMIT {page_size}"
                    )
                    for row in cur.fetchall():
                        yield row[id_field], dict(row)

    return data_generator
