class PolyscopeSession:
    """This is a context manager for Polyscope sessions."""
    def __init__(self):
        self._handles = []

    def __enter__(self):
        ps.init()
        return self

    def register(self, handle):
        self._handles.append(handle)
        return handle

    def __exit__(self, exc_type, exc, tb):
        # Remove objects so successive runs don’t stack
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        # Optional: ps.shutdown() if you want to fully close viewer
        # ps.shutdown()
        return False  # don’t suppress exceptions
