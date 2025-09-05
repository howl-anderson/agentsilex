from contextvars import ContextVar


class Context:
    def __init__(self, page):
        self.page = page


_current_context: ContextVar[Context] = ContextVar("current_context")


def set_context(ctx: Context):
    _current_context.set(ctx)


def get_context() -> Context:
    return _current_context.get()
