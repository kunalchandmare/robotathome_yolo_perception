import argparse

NULL_TOKENS = {"", "null", "none"}


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value!r}")


def parse_optional_str(value):
    if value is None:
        return None
    text = str(value)
    if text.strip().lower() in NULL_TOKENS:
        return None
    return text


def parse_optional_int(value):
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in NULL_TOKENS:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected an integer value, got: {value!r}") from exc


def parse_optional_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in NULL_TOKENS:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected a float value, got: {value!r}") from exc


def parse_optional_bool(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in NULL_TOKENS:
        return None
    return parse_bool(text)
