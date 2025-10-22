import pytest
import time
from agentsilex import observability
from agentsilex.observability import span, ManagedSpan, SpanManager

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture(scope="session")
def tracer_provider():
    """Setup TracerProvider once for all tests"""
    provider = TracerProvider()
    trace.set_tracer_provider(provider)

    # Initialize the tracer with the test provider
    observability.initialize_tracer()

    return provider


@pytest.fixture
def span_exporter(tracer_provider):
    """Setup in-memory span exporter for each test"""
    exporter = InMemorySpanExporter()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    yield exporter

    # Clean up: remove the processor and clear exporter
    exporter.clear()


def test_basic_span(span_exporter):
    """Test basic span creation and usage"""
    with span("test_operation", foo="bar"):
        time.sleep(0.01)

    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    assert spans[0].name == "test_operation"
    assert spans[0].attributes["foo"] == "bar"


def test_nested_spans(span_exporter):
    """Test nested spans"""
    with span("parent"):
        with span("child1"):
            time.sleep(0.01)
        with span("child2"):
            time.sleep(0.01)

    spans = span_exporter.get_finished_spans()

    assert len(spans) == 3

    parent_span = next(s for s in spans if s.name == "parent")
    child1_span = next(s for s in spans if s.name == "child1")
    child2_span = next(s for s in spans if s.name == "child2")

    assert child1_span.parent.span_id == parent_span.context.span_id
    assert child2_span.parent.span_id == parent_span.context.span_id


def test_managed_span_manual_control(span_exporter):
    """Test ManagedSpan manual control"""
    managed = ManagedSpan("manual_span", key="value")
    managed.start()

    time.sleep(0.01)

    managed.end()

    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    assert spans[0].name == "manual_span"
    assert spans[0].attributes["key"] == "value"


def test_managed_span_with_nested(span_exporter):
    """Test ManagedSpan as parent with nested static span"""
    managed = ManagedSpan("parent_managed").start()

    with span("nested_child"):
        time.sleep(0.01)

    managed.end()

    spans = span_exporter.get_finished_spans()

    assert len(spans) == 2

    parent = next(s for s in spans if s.name == "parent_managed")
    child = next(s for s in spans if s.name == "nested_child")

    assert child.parent.span_id == parent.context.span_id


def test_span_manager_switch(span_exporter):
    """Test SpanManager switch functionality"""
    manager = SpanManager()

    manager.switch_to("agent_1", agent="first")
    time.sleep(0.01)

    manager.switch_to("agent_2", agent="second")
    time.sleep(0.01)

    manager.end_current()

    spans = span_exporter.get_finished_spans()

    assert len(spans) == 2

    agent1_span = next(s for s in spans if s.name == "agent_1")
    agent2_span = next(s for s in spans if s.name == "agent_2")

    assert agent1_span.attributes["agent"] == "first"
    assert agent2_span.attributes["agent"] == "second"

    assert agent1_span.end_time < agent2_span.start_time


def test_span_manager_sequential_execution(span_exporter):
    """Test SpanManager sequential execution (simulating agent handoff)"""
    manager = SpanManager()

    agents = ["triage", "weather", "summary"]

    for agent_name in agents:
        manager.switch_to(f"agent.{agent_name}", agent=agent_name)
        time.sleep(0.01)

    manager.end_current()

    spans = span_exporter.get_finished_spans()

    assert len(spans) == 3

    triage = next(s for s in spans if s.name == "agent.triage")
    weather = next(s for s in spans if s.name == "agent.weather")
    summary = next(s for s in spans if s.name == "agent.summary")

    assert triage.end_time < weather.start_time
    assert weather.end_time < summary.start_time


def test_span_attributes(span_exporter):
    """Test span attribute setting"""
    with span("test_attrs", initial="value") as s:
        if s:
            s.set_attribute("dynamic", "added")
            s.set_attribute("number", 42)

    spans = span_exporter.get_finished_spans()

    assert len(spans) == 1
    assert spans[0].attributes["initial"] == "value"
    assert spans[0].attributes["dynamic"] == "added"
    assert spans[0].attributes["number"] == 42


def test_span_manager_with_nested_operations(span_exporter):
    """Test SpanManager with nested operations (simulating real scenario)"""
    manager = SpanManager()

    manager.switch_to("agent.main")

    with span("llm.call", model="gpt-4"):
        time.sleep(0.01)

    with span("tool.search"):
        time.sleep(0.01)

    manager.switch_to("agent.weather")

    with span("llm.call", model="gpt-4"):
        time.sleep(0.01)

    manager.end_current()

    spans = span_exporter.get_finished_spans()

    assert len(spans) == 5

    llm_calls = [s for s in spans if s.name == "llm.call"]
    assert len(llm_calls) == 2

    main_agent = next(s for s in spans if s.name == "agent.main")
    first_llm = llm_calls[0]
    assert first_llm.parent.span_id == main_agent.context.span_id


def test_empty_span_manager(span_exporter):
    """Test empty SpanManager"""
    manager = SpanManager()

    manager.end_current()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 0


def test_multiple_switches_without_end(span_exporter):
    """Test multiple switches without manual end"""
    manager = SpanManager()

    manager.switch_to("span1")
    manager.switch_to("span2")
    manager.switch_to("span3")

    del manager

    spans = span_exporter.get_finished_spans()

    assert len(spans) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
