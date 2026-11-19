"""Regression: public native ranking must survive ingestion and note generation."""
from types import SimpleNamespace

import pytest
from test_multiuser import actor, env, ingest  # noqa: F401

from ours_memory.memos_backend import MemOSBackend
from ours_memory.models import Caller, Event, ValidationError, stable_id
from ours_memory.multiuser import MultiUserMemory
from ours_memory.store import Store


class NativeMemory:
    def __init__(self):
        self.rows = {}
        self.searches = []

    def extract(self, messages):
        text = messages[0]['content']
        return [SimpleNamespace(model_dump=lambda: {'memory': text, 'metadata': {}})]

    def add(self, rows):
        self.rows.update({r['id']: r for r in rows})

    def search(self, query, top_k):
        self.searches.append((query, top_k))
        # Deliberate rank differs from ID order; both best hits are from one person.
        ranked = sorted(self.rows.values(), key=lambda r: ('EEG' not in r['memory'],
                                                          'MRI' not in r['memory']))
        return [SimpleNamespace(id=r['id'], memory=r['memory']) for r in ranked[:top_k]]


def test_public_index_one_search_keeps_best_same_author_cards_and_restart(env, tmp_path):  # noqa: F811
    m, model, clock, config = env
    m.control.access_mode = 'shared_pool'
    m.settings.candidate_limit = 2
    memory, opened = NativeMemory(), []

    def factory(policy, user):
        opened.append((policy, user))
        return SimpleNamespace(text_mem=memory)

    backend = MemOSBackend({'directory': str(tmp_path / 'native')}, m.store, factory)
    m.backend = m.core.backend = backend
    for i in range(24):
        person = ['marcus', 'martha', 'patel'][i % 3]
        ingest(m, text=f'unrelated {i}', event_id=f'noise{i}', who=actor(person), subject=person,
               category=['clinical', 'logistics'][i % 2])
    ingest(m, text='EEG Friday', event_id='eeg', who=actor('patel'), subject='patel')
    ingest(m, text='MRI Tuesday', event_id='mri', who=actor('patel'), subject='patel', category='clinical')
    policy = stable_id('shared_pool', 'care')
    assert opened == [(policy, policy)]
    assert len(memory.rows) == 26
    assert {c['policy'] for c in m.store.all('card')} == {policy}
    assert {c['subject'] for c in m.store.all('card')} == {'marcus', 'martha', 'patel'}

    original = model.complete
    model.complete = lambda stage, payload: ({'card_ids': [], 'uses': {}} if stage == 'select'
                                             else original(stage, payload))
    result = m.read(actor(), subject_ids=['marcus'], query='tests', purpose='current')
    assert [c['claim_text'] for c in result['cards']] == ['EEG Friday', 'MRI Tuesday']
    assert memory.searches == [('marcus tests', 2)]
    assert all(c['usage_note'] == (c.get('use_hint') or '') for c in result['cards'])

    # Reload the service and backend against the persisted public layout.
    config['control']['access_mode'] = 'shared_pool'
    backend2 = MemOSBackend({'directory': str(tmp_path / 'native')}, m.store, factory)
    # These cards were written using the historical schema. A new shared-pool
    # service must reject them instead of silently treating them as new cards.
    with pytest.raises(ValidationError, match='新运行目录'):
        MultiUserMemory(m.settings, backend2, model, m.store, config['control'], clock)


def test_old_partitioned_store_requires_explicit_rebuild(env):  # noqa: F811
    m, model, clock, config = env
    ingest(m)
    config['control']['access_mode'] = 'shared_pool'
    with pytest.raises(ValidationError, match='新运行目录'):
        MultiUserMemory(m.settings, m.backend, model, m.store, config['control'], clock)
    assert len(m.store.all('card')) == 1


def test_shared_sleep_sees_all_people_once(env):  # noqa: F811
    m, _, _, _ = env
    m.control.access_mode = 'shared_pool'
    ingest(m, text='100', event_id='one', who=actor('marcus'), subject='marcus')
    ingest(m, text='200', event_id='two', who=actor('martha'), subject='martha')
    result = m.sleep(actor('marcus', 'clinical'))
    assert len(result['jobs']) == 1
    assert result['jobs'][0]['processed'] == 2
    assert not m.store.all('pending_state')
    assert {s['subject'] for s in m.store.states()} == {'marcus', 'martha'}


def test_native_public_pool_isolates_spaces_and_preserves_order(tmp_path):
    store = Store(':memory:')
    memories = {}

    def factory(policy, user):
        return SimpleNamespace(text_mem=memories.setdefault(policy, NativeMemory()))

    backend = MemOSBackend({'directory': str(tmp_path)}, store, factory)
    for space, text in [('one', 'EEG one'), ('two', 'MRI two')]:
        event = Event('e', 'author', space, 'agent', 'task', text, '2026-10-20T00:00:00Z', ('agent',))
        backend.index_card(event, stable_id('shared_pool', space), space, text)
    hits = backend.search_shared('tests', Caller('someone_else', 'one', 'agent'),
                                 stable_id('shared_pool', 'one'), 20)
    assert [h.text for h in hits] == ['EEG one']
    with pytest.raises(ValidationError):
        backend.search_shared('tests', Caller('u', 'one', 'agent'), stable_id('shared_pool', 'two'), 20)
    store.close()
