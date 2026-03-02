from hippocortex.router import MemoryRouter


def test_router_prefers_semantic_keywords():
    router = MemoryRouter()
    decision = router.route("Can you summarize my history and patterns?", max_tokens=1000)
    assert decision.intent == "semantic"
    assert decision.semantic_tokens > decision.working_memory_tokens
