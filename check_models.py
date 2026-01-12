def can_cast_to_float(x) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def is_truthy_string(s: str) -> bool:
    return s.strip().lower() in {"true", "yes", "1"}


def is_falsy_string(s: str) -> bool:
    return s.strip().lower() in {"false", "no", "0"}
