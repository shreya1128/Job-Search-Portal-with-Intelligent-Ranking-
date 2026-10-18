"""Export the MySQL DDL and a Mermaid ER diagram from the models (keeps docs in sync with code)."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy.dialects import mysql
from sqlalchemy.schema import CreateIndex, CreateTable

from jobintel.db.models import Base


def mysql_ddl() -> str:
    d = mysql.dialect()
    parts = ["-- Auto-generated from jobintel/db/models.py  (MySQL / MariaDB)", "SET NAMES utf8mb4;", ""]
    for t in Base.metadata.sorted_tables:
        parts.append(str(CreateTable(t).compile(dialect=d)).strip() + " ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;")
        for ix in sorted(t.indexes, key=lambda i: i.name):
            parts.append(str(CreateIndex(ix).compile(dialect=d)).strip() + ";")
        parts.append("")
    return "\n".join(parts)


def mermaid_er() -> str:
    lines = ["erDiagram"]
    for t in Base.metadata.sorted_tables:
        for fk in t.foreign_keys:
            lines.append(f'    {fk.column.table.name} ||--o{{ {t.name} : "{fk.parent.name}"')
    for t in Base.metadata.sorted_tables:
        lines.append(f"    {t.name} {{")
        for c in t.columns:
            kind = (str(c.type).split("(")[0].split(" ")[0].lower() or "col")
            tag = " PK" if c.primary_key else (" FK" if c.foreign_keys else "")
            lines.append(f"        {kind} {c.name}{tag}")
        lines.append("    }")
    return "\n".join(lines)


def export(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "schema.sql").write_text(mysql_ddl(), encoding="utf-8")
    (out_dir / "er_diagram.mmd").write_text(mermaid_er(), encoding="utf-8")
