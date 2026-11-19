from types import SimpleNamespace

from ours_memory.baselines.builders import _point_amem_at_base


def test_amem_deepseek_disables_thinking_without_changing_other_options(monkeypatch):
    captured = {}
    monkeypatch.setenv("OPENAI_BASE_URL", "https://placeholder.invalid")

    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, *args, **kwargs):
            captured.update(kwargs)
            return "ok"

    monkeypatch.setattr("openai.OpenAI", Client)
    llm = SimpleNamespace(model="deepseek-flash", client=None)
    system = SimpleNamespace(llm_controller=SimpleNamespace(llm=llm))
    _point_amem_at_base(system, "placeholder", "https://api.deepseek.com")
    assert llm.client.chat.completions.create(max_tokens=1000, extra_body={"test": 1}) == "ok"
    assert captured == {"max_tokens": 1000, "extra_body": {"test": 1, "thinking": {"type": "disabled"}}}
