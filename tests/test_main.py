import importlib
import os
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

os.environ['USE_GEMINI'] = 'false'
os.environ['PINECONE_API_KEY'] = 'test-only'
chatbot = importlib.import_module('main')
client = TestClient(chatbot.app)


def test_health():
    assert client.get('/health').json() == {'status': 'ok'}


def test_missing_message():
    assert client.post('/chat', json={'history': []}).status_code == 422


def test_pinecone_shapes():
    assert chatbot.extract_rag_answer({'message': {'content': 'RAG answer'}}) == 'RAG answer'
    assert chatbot.extract_rag_answer({'choices': [{'message': {'content': 'Alternative'}}]}) == 'Alternative'


def test_chat_includes_history(monkeypatch):
    mock = AsyncMock(return_value=('Source-grounded answer', ['knowledge.pdf']))
    monkeypatch.setattr(chatbot, 'ask_pinecone', mock)
    monkeypatch.setattr(chatbot, 'USE_GEMINI', False)
    first = client.post('/chat', json={'chatInput': 'hello', 'history': []})
    assert first.status_code == 200, first.text
    j = first.json()
    assert j['answer'] == 'Source-grounded answer'
    assert j['output'] == j['answer']
    assert j['sources'] == ['knowledge.pdf']
    assert len(j['history']) == 2
    second = client.post('/chat', json={'message': 'follow up', 'history': j['history'], 'session_id': j['session_id']})
    assert second.status_code == 200
    assert second.json()['session_id'] == j['session_id']
    assert len(second.json()['history']) == 4
    assert len(mock.call_args.args[0]) == 3  # 2 prior + new user question
