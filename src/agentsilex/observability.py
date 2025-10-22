import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource

from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)

from contextlib import contextmanager

_tracer: trace.Tracer = None


def setup_tracer_provider():
    """Setup TracerProvider with OTLP exporter if endpoint is configured"""
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        exporter = OTLPSpanExporter(endpoint, insecure=True)
        provider = TracerProvider(
            resource=Resource.create({"service.name": "agentsilex"})
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)


def initialize_tracer():
    """Initialize the global tracer instance"""
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer("agentsilex")


@contextmanager
def span(name: str, **attrs):
    with _tracer.start_as_current_span(name, attributes=attrs) as s:
        yield s


class ManagedSpan:
    def __init__(self, name: str, **attributes):
        self.name = name
        self.attributes = attributes
        self._span = None
        self._context = None

    def start(self):
        self._span = _tracer.start_span(self.name, attributes=self.attributes)
        self._context = trace.use_span(self._span, end_on_exit=False)
        self._context.__enter__()
        return self

    def end(self):
        if self._context:
            self._context.__exit__(None, None, None)
            self._context = None
        if self._span:
            self._span.end()
            self._span = None


class SpanManager:
    def __init__(self):
        self.current: ManagedSpan | None = None

    def switch_to(self, name: str, **attributes):
        if self.current:
            self.current.end()

        self.current = ManagedSpan(name, **attributes).start()
        return self.current

    def end_current(self):
        if self.current:
            self.current.end()
            self.current = None

    def __del__(self):
        self.end_current()
