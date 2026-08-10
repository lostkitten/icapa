"""Private constants shared by report-builder services."""

from openpyxl.styles import Font, PatternFill
import re



_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_FORMULA_PREFIXES = ("=", "+", "-", "@")
_SENSITIVE_PARTS = {
    "connection",
    "credential",
    "database",
    "dsn",
    "host",
    "password",
    "path",
    "privatekey",
    "query",
    "schema",
    "secret",
    "server",
    "sql",
    "token",
    "url",
    "user",
    "warehouse",
}
_KEY_COLUMN_NAMES = {
    "attribute",
    "configurationkey",
    "field",
    "key",
    "metric",
    "parameter",
    "property",
    "setting",
}
_DATABASE_URI_VALUE = re.compile(
    r"\b(?:"
    r"postgres(?:ql)?|mysql|mariadb|mssql|sqlserver|oracle|snowflake|"
    r"redshift|mongodb(?:\+srv)?|redis|sqlite|duckdb"
    r")(?:\+[A-Za-z0-9_.-]+)?://",
    flags=re.IGNORECASE,
)
_CREDENTIAL_URI_VALUE = re.compile(
    r"\b[A-Za-z][A-Za-z0-9+.-]*://[^/\s@:]+:[^/@\s]+@",
    flags=re.IGNORECASE,
)
_CREDENTIAL_ASSIGNMENT_VALUE = re.compile(
    r"(?:^|[?&;,\s])"
    r"(?:password|passwd|pwd|secret|token|api[_ -]?key|private[_ -]?key|"
    r"client[_ -]?secret|access[_ -]?key)"
    r"\s*[:=]\s*[^\s;&,]+",
    flags=re.IGNORECASE,
)
_CONNECTION_ASSIGNMENT_VALUE = re.compile(
    r"(?:^|;)\s*"
    r"(?:driver|dsn|server|data\s+source|host|database|dbname|"
    r"initial\s+catalog|uid|user\s+id|username|warehouse|account)"
    r"\s*=\s*[^;]+(?:;|$)",
    flags=re.IGNORECASE,
)
_BEARER_CREDENTIAL_VALUE = re.compile(
    r"(?:authorization\s*:\s*)?bearer\s+[A-Za-z0-9._~+/=-]+",
    flags=re.IGNORECASE,
)
_PRIVATE_KEY_VALUE = re.compile(
    r"-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----",
    flags=re.IGNORECASE,
)
_SQL_STATEMENT_VALUE = re.compile(
    r"(?:^|[\s;(])(?:"
    r"select\s+.+?\s+from\s+|"
    r"insert\s+into\s+|"
    r"update\s+\S+\s+set\s+|"
    r"delete\s+from\s+|"
    r"merge\s+into\s+|"
    r"(?:create|alter|drop|truncate)\s+"
    r"(?:table|view|schema|database)\s+|"
    r"with\s+[A-Za-z_][A-Za-z0-9_]*\s+as\s*\("
    r")",
    flags=re.IGNORECASE | re.DOTALL,
)
_EXCEL_MAX_ROWS = 1_048_576
_V1_HEADER_ROWS = 3
_HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
_HEADER_FONT = Font(color="FFFFFF", bold=True)
_PAYLOAD_FIELD_BY_SHEET = {
    "Overview": "overview",
    "Review Schedule": "review_schedule",
    "Latest Holdings": "latest_holdings",
    "All Review Weights": "all_review_weights",
    "Performance": "performance",
    "Exposures": "exposures",
    "Turnover": "turnover",
    "Attribution": "attribution",
    "Methodology Parameters": "methodology_parameters",
    "Data Sources": "data_sources",
    "Validation": "validation",
}


__all__ = [
    "_BEARER_CREDENTIAL_VALUE",
    "_CONNECTION_ASSIGNMENT_VALUE",
    "_CREDENTIAL_ASSIGNMENT_VALUE",
    "_CREDENTIAL_URI_VALUE",
    "_DATABASE_URI_VALUE",
    "_EXCEL_MAX_ROWS",
    "_FORMULA_PREFIXES",
    "_HEADER_FILL",
    "_HEADER_FONT",
    "_KEY_COLUMN_NAMES",
    "_PAYLOAD_FIELD_BY_SHEET",
    "_PRIVATE_KEY_VALUE",
    "_SAFE_NAME",
    "_SENSITIVE_PARTS",
    "_SQL_STATEMENT_VALUE",
    "_V1_HEADER_ROWS",
]
