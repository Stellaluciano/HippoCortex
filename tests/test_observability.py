import logging

from hippocortex import DummyEmbedder, HippoConfig, HippoCortex
from hippocortex.config import ModelConfig, StorageConfig
from hippocortex.telemetry import InMemoryTelemetry


def _build_sdk(tmp_path):
    db_path = tmp_path / "memory.db"
    telemetry = InMemoryTelemetry()
    sdk = HippoCortex.default(
        config=HippoConfig(
            storage=StorageConfig(db_path=str(db_path)),
            model=ModelConfig(embedding_dim=8),
        ),
        embedder=DummyEmbedder(dimension=8),
    )
    sdk.telemetry = telemetry
    sdk.hippo.telemetry = telemetry
    return sdk, telemetry


def test_router_explain_contains_keywords_and_budget():
    decision = HippoCortex.default().router.route("summarize latest history", max_tokens=100)
    assert "matched_keywords" in decision.explain
    assert "budget_allocation" in decision.explain


def test_observability_logs_include_required_fields(tmp_path):
    sdk, _ = _build_sdk(tmp_path)
    records = []

    class CaptureHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = CaptureHandler()
    logger = logging.getLogger("hippocortex")
    obs_logger = logging.getLogger("hippocortex.observability")
    logger.addHandler(handler)
    obs_logger.addHandler(handler)

    try:
        sdk.hippo.add_event("agent-1", "session-1", "user", "I like tea", request_id="req-1")
        sdk.consolidate(agent_id="agent-1", session_id="session-1", request_id="req-1")
        sdk.build_context(
            agent_id="agent-1",
            session_id="session-1",
            user_message="summarize history",
            max_tokens=100,
            request_id="req-1",
        )
    finally:
        logger.removeHandler(handler)
        obs_logger.removeHandler(handler)

    assert records
    for record in records:
        assert hasattr(record, "agent_id")
        assert hasattr(record, "session_id")
        assert hasattr(record, "request_id")


def test_telemetry_metrics_emitted(tmp_path):
    sdk, telemetry = _build_sdk(tmp_path)

    sdk.hippo.add_event("agent-1", "session-1", "user", "I like tea", request_id="req-1")
    sdk.consolidate(agent_id="agent-1", session_id="session-1", request_id="req-1")
    sdk.cortex.search(agent_id="agent-1", session_id="session-1", query="tea", k=5, request_id="req-1")

    counter_names = [name for name, _, _ in telemetry.counters]
    assert "hippocortex.events.write" in counter_names

    observation_names = [name for name, _, _ in telemetry.observations]
    assert "hippocortex.consolidation.duration_ms" in observation_names
    assert "hippocortex.search.duration_ms" in observation_names
    assert "hippocortex.search.hit_rate" in observation_names
