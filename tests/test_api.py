import pytest
from unittest.mock import MagicMock, patch
from app import query_vectorstore, process_uploaded_file, save_feedback, embeddings
import os

@pytest.fixture
def mock_vectorstore():
    mock = MagicMock()
    mock.similarity_search.return_value = []
    return mock

def test_query_vectorstore_empty(mock_vectorstore):
    results = query_vectorstore(mock_vectorstore, query="", top_k=3)
    assert results == []

def test_save_feedback(tmp_path):
    feedback_file = tmp_path / "feedback.csv"

    # Patch the feedback file path inside save_feedback if needed
    with patch("app.feedback_file", str(feedback_file), create=True):
        save_feedback("query", "result", "Yes")

    # Verify feedback was saved
    assert feedback_file.exists()
    content = feedback_file.read_text()
    assert "query" in content
    assert "result" in content
    assert "Yes" in content

def test_process_uploaded_file_txt(tmp_path, mock_vectorstore):
    # Mock uploaded file
    uploaded_file = MagicMock()
    uploaded_file.name = "test.txt"
    uploaded_file.getbuffer.return_value = b"Some content"

    # If needed, patch `os.remove` or handle temp dir changes
    with patch("app.os.remove") as mock_remove, patch("app.os.getcwd", return_value=str(tmp_path)):
        num_chunks = process_uploaded_file(uploaded_file, mock_vectorstore, embeddings)
        assert num_chunks > 0
        mock_vectorstore.add_documents.assert_called_once()
        mock_remove.assert_called()