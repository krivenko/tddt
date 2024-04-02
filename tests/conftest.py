# Ignore unit tests depending on realevol if module realevol cannot be imported
collect_ignore = []
try:
    import realevol  # noqa: F401
except ImportError:
    collect_ignore.append("test_realevol.py")
