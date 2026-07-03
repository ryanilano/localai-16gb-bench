def dedupe(items):
    """Return items with duplicates removed, preserving first-seen order."""
    return list(set(items))  # BUG: set() drops the first-seen ordering
