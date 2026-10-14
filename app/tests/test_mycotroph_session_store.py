import json
from pathlib import Path

import pytest

from megaglitch.mycotroph_session_store import SessionStore, SessionStoreError


def graph(label="Invert"):
    return {
        "version": "1.0",
        "nodes": [
            {
                "id": "InvertColorNode:stable",
                "type": "InvertColorNode",
                "label": label,
                "position": [100, 50],
                "params": {"enabled": True},
                "future_node": {"kept": True},
            }
        ],
        "links": [
            {
                "from_node": "SOURCE",
                "to_node": label,
                "to_node_id": "InvertColorNode:stable",
                "from_attr": "image_out",
                "to_attr": "image_in",
            },
            {
                "from_node": label,
                "from_node_id": "InvertColorNode:stable",
                "to_node": "DISPLAY",
                "from_attr": "image_out",
                "to_attr": "image_in",
            },
        ],
    }


def session():
    return {
        "format": "megaglitch.session",
        "format_version": 2,
        "decks": {"a": {"graph": graph()}, "b": {"graph": graph("Invert B")}},
        "future_top": {"kept": [1, 2, 3]},
    }


def test_atomic_save_and_load_preserve_unknown_fields(tmp_path):
    destination = tmp_path / "machine.json"
    store = SessionStore(session())
    assert store.snapshot().generation == 0
    assert store.replace(session()).dirty is True
    saved = store.save(destination)
    assert saved.dirty is False
    assert saved.path == str(destination.resolve())
    assert not list(tmp_path.glob(".machine.json.*.tmp"))

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["future_top"] == {"kept": [1, 2, 3]}
    assert payload["decks"]["a"]["graph"]["nodes"][0]["future_node"] == {"kept": True}

    reloaded = SessionStore({"version": "1.0", "nodes": [], "links": []})
    snapshot = reloaded.load(destination)
    assert snapshot.dirty is False
    assert reloaded.document() == store.document()


def test_failed_load_preserves_last_committed_document(tmp_path):
    store = SessionStore(session())
    before = store.document()
    broken = tmp_path / "broken.json"
    broken.write_text("{not-json", encoding="utf-8")
    with pytest.raises(SessionStoreError) as caught:
        store.load(broken)
    assert caught.value.code == "session_read_failed"
    assert store.document() == before
    assert store.snapshot().generation == 0


def test_invalid_replace_is_transactional():
    store = SessionStore(session())
    before = store.document()
    with pytest.raises(SessionStoreError) as caught:
        store.replace({"format": "megaglitch.session", "format_version": 99})
    assert caught.value.code == "invalid_session"
    assert store.document() == before
    assert store.snapshot().generation == 0


def test_save_requires_existing_parent_and_reports_stable_error(tmp_path):
    store = SessionStore(session())
    with pytest.raises(SessionStoreError) as caught:
        store.save(tmp_path / "missing" / "machine.json")
    assert caught.value.code == "session_directory_not_found"
    assert Path(caught.value.path).name == "machine.json"


def test_document_and_snapshot_are_detached():
    store = SessionStore(session())
    detached = store.document()
    detached["future_top"]["kept"].append(999)
    assert store.document()["future_top"] == {"kept": [1, 2, 3]}
    assert len(store.snapshot().sha256) == 64
