"""Split markdown error-code tables into per-row chunks."""

import re

ERROR_CODE_RE = re.compile(r"^\d+\.\d+[A-Z]*$")
TABLE_SEPARATOR_RE = re.compile(r"^\|[-:|\s]+\|$")


def _parse_table_cells(line: str) -> list[str] | None:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return None
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def _is_error_table_header(cells: list[str]) -> bool:
    return any("error code" in cell.lower() for cell in cells)


def extract_error_table_rows(content: str) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    table_started = False

    for line in content.splitlines():
        cells = _parse_table_cells(line)
        if cells is None:
            continue

        if TABLE_SEPARATOR_RE.match(line.strip()):
            table_started = True
            continue

        if not table_started and _is_error_table_header(cells):
            table_started = True
            continue

        if not table_started:
            continue

        if len(cells) < 3:
            continue

        code, description, action = cells[0], cells[1], cells[2]
        if not ERROR_CODE_RE.match(code):
            continue

        rows.append((code, description, action))

    return rows


def format_error_row_content(
    *,
    section_title: str,
    error_code: str,
    description: str,
    action: str,
) -> str:
    return (
        f"Error code: {error_code}\n"
        f"Section: {section_title}\n"
        f"Description: {description}\n"
        f"Action: {action}"
    )
