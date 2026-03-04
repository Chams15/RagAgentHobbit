"""
Comprehensive test suite for the RAG Chatbot using PyTest.
Tests cover positive and negative cases for all major functions.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from collections import deque

from langchain_core.documents import Document

# Import the modules to test
import vector
from vector import (
    detect_chapter,
    get_book_metadata,
    _load_raw_text,
    get_embeddings,
    _get_parent_docs,
    ChapterAwareRetriever,
    cleanup,
    _check_source_file_freshness,
)

import main
from main import (
    format_history,
    format_docs,
    chat_history,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_source_file():
    """Create a temporary Hobbit source file for testing."""
    temp_dir = tempfile.mkdtemp()
    source_file = Path(temp_dir) / "TheHobbit.md"
    
    # Create a minimal markdown structure
    content = """# The Hobbit

## II
## ROAST MUTTON

Once upon a time, Bilbo Baggins found himself in a predicament. The dwarves were hungry and tired.

Several paragraphs of content would follow here about roast mutton and the party.

## III
## A SHORT REST

After eating, the company took shelter in a cave. Gandalf warned them about goblins.

More content about their rest and preparations for the journey ahead.

## IV
## OVER HILL AND UNDER HILL

The mountains loomed ahead as they traveled. Trolls appeared on the path.

The trolls captured them and the story continued into the mountain journey.
"""
    source_file.write_text(content)
    
    original_source = vector.SOURCE_FILE
    vector.SOURCE_FILE = str(source_file)
    
    yield source_file, temp_dir
    
    vector.SOURCE_FILE = original_source
    vector._load_raw_text.cache_clear()
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def clear_caches():
    """Clear all LRU caches before and after each test."""
    vector._load_raw_text.cache_clear()
    vector.get_embeddings.cache_clear()
    vector._get_parent_docs.cache_clear()
    vector.get_book_metadata.cache_clear()
    yield
    vector._load_raw_text.cache_clear()
    vector.get_embeddings.cache_clear()
    vector._get_parent_docs.cache_clear()
    vector.get_book_metadata.cache_clear()


@pytest.fixture
def clear_chat_history():
    """Clear chat history before each test."""
    main.chat_history.clear()
    yield
    main.chat_history.clear()


# ============================================================================
# POSITIVE TEST CASES - detect_chapter()
# ============================================================================

class TestDetectChapter:
    """Tests for the detect_chapter() function."""
    
    def test_detect_chapter_with_numeric_valid(self):
        """Positive: Should detect valid chapter numbers (1-19)."""
        assert detect_chapter("What happens in chapter 1?") == "AN UNEXPECTED PARTY"
        assert detect_chapter("Tell me about chapter 5") == "RIDDLES IN THE DARK"
        assert detect_chapter("Chapter 19 is about what?") == "THE LAST STAGE"
    
    def test_detect_chapter_with_roman_numerals(self):
        """Positive: Should detect Roman numeral chapter references."""
        assert detect_chapter("What about Chapter I?") == "AN UNEXPECTED PARTY"
        assert detect_chapter("Chapter V details") == "RIDDLES IN THE DARK"
        assert detect_chapter("In Chapter XIX, what happens?") == "THE LAST STAGE"
    
    def test_detect_chapter_mixed_case(self):
        """Positive: Should handle mixed case."""
        assert detect_chapter("What happens in CHAPTER 2?") == "ROAST MUTTON"
        assert detect_chapter("Tell me about ChApTeR 3") == "A SHORT REST"
    
    def test_detect_chapter_abbreviated(self):
        """Positive: Should accept abbreviated 'ch'."""
        assert detect_chapter("What's in ch 1?") == "AN UNEXPECTED PARTY"
        assert detect_chapter("ch. 5 is about?") == "RIDDLES IN THE DARK"
    
    def test_detect_chapter_no_chapter_mentioned(self):
        """Negative: Should return None when no chapter is mentioned."""
        assert detect_chapter("What is the book about?") is None
        assert detect_chapter("Tell me about Bilbo") is None
        assert detect_chapter("") is None
    
    def test_detect_chapter_invalid_number(self):
        """Negative: Should return None for invalid chapter numbers."""
        assert detect_chapter("Chapter 0 is what?") is None  # Chapter 0 doesn't exist
        assert detect_chapter("Chapter 20 is about?") is None  # Only 19 chapters
        assert detect_chapter("Chapter 100 details") is None
    
    def test_detect_chapter_invalid_roman(self):
        """Negative: Should return None for invalid Roman numerals."""
        assert detect_chapter("Chapter XX is?") is None  # Not in _ROMAN_MAP
        assert detect_chapter("Chapter XXX") is None
    
    def test_detect_chapter_false_positive_avoidance(self):
        """Positive: Should not trigger on false positives with word boundaries."""
        # With the updated regex using \b, these shouldn't match incorrectly
        assert detect_chapter("this chapter discusses") is None
        assert detect_chapter("the chapter is long") is None
        assert detect_chapter("multiple chapters in") is None


# ============================================================================
# POSITIVE TEST CASES - format_docs()
# ============================================================================

class TestFormatDocs:
    """Tests for the format_docs() function."""
    
    def test_format_docs_single_document(self):
        """Positive: Should format a single document correctly."""
        doc = Document(page_content="This is content", metadata={"source": "test"})
        result = format_docs([doc])
        assert result == "This is content"
    
    def test_format_docs_multiple_documents(self):
        """Positive: Should format multiple documents with separator."""
        doc1 = Document(page_content="Content 1", metadata={})
        doc2 = Document(page_content="Content 2", metadata={})
        doc3 = Document(page_content="Content 3", metadata={})
        result = format_docs([doc1, doc2, doc3])
        expected = "Content 1\n\n---\n\nContent 2\n\n---\n\nContent 3"
        assert result == expected
    
    def test_format_docs_deduplication(self):
        """Positive: Should deduplicate identical contents while preserving order."""
        doc1 = Document(page_content="Content A", metadata={})
        doc2 = Document(page_content="Content A", metadata={})  # Duplicate
        doc3 = Document(page_content="Content B", metadata={})
        result = format_docs([doc1, doc2, doc3])
        expected = "Content A\n\n---\n\nContent B"
        assert result == expected
    
    def test_format_docs_empty_list(self):
        """Negative: Should handle empty document list."""
        result = format_docs([])
        assert result == ""
    
    def test_format_docs_whitespace_handling(self):
        """Positive: Should preserve whitespace in content."""
        doc1 = Document(page_content="Line 1\nLine 2", metadata={})
        doc2 = Document(page_content="Line 3", metadata={})
        result = format_docs([doc1, doc2])
        assert "Line 1\nLine 2" in result
        assert "Line 3" in result


# ============================================================================
# POSITIVE TEST CASES - format_history()
# ============================================================================

class TestFormatHistory:
    """Tests for the format_history() function."""
    
    def test_format_history_empty(self, clear_chat_history):
        """Positive: Should return default message for empty history."""
        result = format_history()
        assert result == "No previous conversation."
    
    def test_format_history_single_turn(self, clear_chat_history):
        """Positive: Should format single Q&A turn."""
        main.chat_history.append(("What is Bilbo?", "Bilbo Baggins is a hobbit."))
        result = format_history()
        assert "User: What is Bilbo?" in result
        assert "Assistant: Bilbo Baggins is a hobbit." in result
    
    def test_format_history_multiple_turns(self, clear_chat_history):
        """Positive: Should format multiple Q&A turns."""
        main.chat_history.append(("Q1", "A1"))
        main.chat_history.append(("Q2", "A2"))
        main.chat_history.append(("Q3", "A3"))
        result = format_history()
        assert "User: Q1" in result
        assert "Assistant: A1" in result
        assert "User: Q3" in result
        assert "Assistant: A3" in result
    
    def test_format_history_truncates_long_answers(self, clear_chat_history):
        """Positive: Should truncate answers longer than MAX_HISTORY_ANSWER_CHARS."""
        long_answer = "x" * (main.MAX_HISTORY_ANSWER_CHARS + 100)
        main.chat_history.append(("Question", long_answer))
        result = format_history()
        # Should be truncated with ellipsis
        assert "..." in result
        assert len(result) < len(long_answer) + 100
    
    def test_format_history_respects_max_turns(self, clear_chat_history):
        """Positive: Should only keep latest MAX_HISTORY_TURNS."""
        for i in range(main.MAX_HISTORY_TURNS + 2):
            main.chat_history.append((f"Q{i}", f"A{i}"))
        
        result = format_history()
        # Should not contain first turn (oldest)
        assert "Q0" not in result
        # Should contain latest turns
        assert f"Q{main.MAX_HISTORY_TURNS}" in result


# ============================================================================
# POSITIVE TEST CASES - Vector Store and Retrieval
# ============================================================================

class TestVectorStoreOperations:
    """Tests for vector store initialization and retrieval."""
    
    @patch('vector.OllamaEmbeddings')
    def test_get_embeddings_success(self, mock_ollama):
        """Positive: Should initialize embeddings successfully."""
        mock_embedding = MagicMock()
        mock_ollama.return_value = mock_embedding
        
        result = get_embeddings()
        assert result is mock_embedding
        mock_ollama.assert_called_once_with(model="mxbai-embed-large")
    
    @patch('vector.OllamaEmbeddings')
    def test_get_embeddings_failure(self, mock_ollama):
        """Negative: Should raise RuntimeError on embedding failure."""
        mock_ollama.side_effect = Exception("Ollama not running")
        
        with pytest.raises(RuntimeError, match="Failed to initialize Ollama embeddings"):
            get_embeddings()
    
    def test_load_raw_text_success(self, temp_source_file, clear_caches):
        """Positive: Should load raw text from file."""
        source_file, _ = temp_source_file
        text = _load_raw_text()
        assert "# The Hobbit" in text
        assert "ROAST MUTTON" in text
    
    def test_load_raw_text_caching(self, temp_source_file, clear_caches):
        """Positive: Should cache raw text and not reload."""
        source_file, _ = temp_source_file
        text1 = _load_raw_text()
        text2 = _load_raw_text()
        # Both should be identical from cache
        assert text1 is text2
    
    def test_get_parent_docs_parsing(self, temp_source_file, clear_caches):
        """Positive: Should parse documents by chapter headers."""
        source_file, _ = temp_source_file
        docs = _get_parent_docs()
        assert len(docs) == 4  # 1 title doc + 3 chapters in our test file
        # Check metadata contains chapter info for the 3 chapter docs
        chapter_docs = [doc for doc in docs if "Chapter" in doc.metadata]
        assert len(chapter_docs) == 3
    
    def test_check_source_file_freshness_no_change(self, temp_source_file, clear_caches):
        """Positive: Should not warn if file hasn't changed."""
        source_file, _ = temp_source_file
        _load_raw_text()  # Load and set mtime
        
        with pytest.warns(None) as warning_list:
            _check_source_file_freshness()
        
        assert len([w for w in warning_list if issubclass(w.category, UserWarning)]) == 0
    
    def test_check_source_file_freshness_warns_on_change(self, temp_source_file, clear_caches):
        """Positive: Should warn if source file is modified."""
        source_file, _ = temp_source_file
        _load_raw_text()  # Load and record mtime
        
        # Modify file
        source_file.write_text(source_file.read_text() + "\nModified content")
        
        with pytest.warns(UserWarning, match="has been modified"):
            _check_source_file_freshness()


# ============================================================================
# POSITIVE TEST CASES - ChapterAwareRetriever
# ============================================================================

class TestChapterAwareRetriever:
    """Tests for the ChapterAwareRetriever class."""
    
    def test_chapter_aware_retriever_no_chapter_specified(self):
        """Positive: Should return all docs when no chapter specified."""
        mock_base_retriever = MagicMock()
        doc1 = Document(page_content="Content 1", metadata={"Chapter": "ROAST MUTTON"})
        doc2 = Document(page_content="Content 2", metadata={"Chapter": "A SHORT REST"})
        mock_base_retriever.invoke.return_value = [doc1, doc2]
        
        retriever = ChapterAwareRetriever(base_retriever=mock_base_retriever)
        result = retriever.invoke("What about Bilbo?")
        
        assert len(result) == 2
        assert result[0] is doc1
        assert result[1] is doc2
    
    def test_chapter_aware_retriever_filters_by_chapter(self):
        """Positive: Should filter docs by chapter when specified."""
        mock_base_retriever = MagicMock()
        doc1 = Document(page_content="Content 1", metadata={"Chapter": "ROAST MUTTON"})
        doc2 = Document(page_content="Content 2", metadata={"Chapter": "A SHORT REST"})
        doc3 = Document(page_content="Content 3", metadata={"Chapter": "ROAST MUTTON"})
        mock_base_retriever.invoke.return_value = [doc1, doc2, doc3]
        
        retriever = ChapterAwareRetriever(base_retriever=mock_base_retriever)
        result = retriever.invoke("What happens in chapter 2?")  # ROAST MUTTON is chapter 2
        
        assert len(result) == 2
        assert all(d.metadata["Chapter"] == "ROAST MUTTON" for d in result)
    
    def test_chapter_aware_retriever_fallback_to_all(self):
        """Positive: Should fallback to all docs if filtering removes everything."""
        mock_base_retriever = MagicMock()
        doc1 = Document(page_content="Content 1", metadata={"Chapter": "ROAST MUTTON"})
        doc2 = Document(page_content="Content 2", metadata={"Chapter": "ROAST MUTTON"})
        mock_base_retriever.invoke.return_value = [doc1, doc2]
        
        retriever = ChapterAwareRetriever(base_retriever=mock_base_retriever)
        # Request chapter 3 but only chapter 2 docs available
        result = retriever.invoke("What's in chapter 3?")  # A SHORT REST
        
        # Should fallback to all docs
        assert len(result) == 2


# ============================================================================
# NEGATIVE TEST CASES - Input Validation
# ============================================================================

class TestInputValidation:
    """Tests for input validation and edge cases."""
    
    def test_detect_chapter_with_extra_whitespace(self):
        """Positive: Should handle extra whitespace."""
        assert detect_chapter("  chapter   5   ") == "RIDDLES IN THE DARK"
    
    def test_detect_chapter_at_start_and_middle(self):
        """Positive: Should find chapter reference anywhere."""
        assert detect_chapter("I want to know about chapter 1") == "AN UNEXPECTED PARTY"
        assert detect_chapter("Previously in chapter 7") == "QUEER LODGINGS"
    
    def test_format_history_with_special_characters(self, clear_chat_history):
        """Positive: Should handle special characters in history."""
        main.chat_history.append(("What's up?", "It's going well!"))
        result = format_history()
        assert "What's up?" in result
        assert "It's going well!" in result
    
    def test_get_book_metadata_returns_string(self, temp_source_file, clear_caches):
        """Positive: Should return valid metadata string."""
        source_file, _ = temp_source_file
        metadata = get_book_metadata()
        assert isinstance(metadata, str)
        assert "The Hobbit" in metadata
        assert "Total Chapters" in metadata
        assert "Total Words" in metadata


# ============================================================================
# NEGATIVE TEST CASES - Resource Cleanup
# ============================================================================

class TestCleanup:
    """Tests for the cleanup function."""
    
    def test_cleanup_with_empty_resources(self):
        """Positive: Should handle cleanup when no resources exist."""
        vector._active_resources.clear()
        # Should not raise any exception
        cleanup()
    
    def test_cleanup_clears_caches(self, clear_caches):
        """Positive: Should clear all cached data."""
        # Populate some cache
        with patch('vector.OllamaEmbeddings'):
            try:
                get_embeddings()
            except:
                pass
        
        cleanup()
        
        # Cache should be cleared
        assert get_embeddings.cache_info().currsize == 0
    
    @patch('vector._active_resources', {'vectorstore': MagicMock(), 'docstore': MagicMock()})
    def test_cleanup_handles_vectorstore_errors(self):
        """Positive: Should handle errors during vectorstore cleanup."""
        mock_vs = vector._active_resources['vectorstore']
        mock_vs.delete_collection.side_effect = Exception("Delete failed")
        
        # Should not raise exception
        cleanup()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for the complete RAG system."""
    
    @patch('vector.OllamaEmbeddings')
    @patch('vector.Chroma')
    @patch('vector.ParentDocumentRetriever')
    @patch('vector.BM25Retriever')
    @patch('vector.EnsembleRetriever')
    @patch('vector.ContextualCompressionRetriever')
    def test_build_retriever_end_to_end(self, mock_comp_retr, mock_ensemble, mock_bm25, 
                                       mock_parent, mock_chroma, mock_embeddings, 
                                       temp_source_file, clear_caches):
        """Integration: Should build complete retriever chain."""
        source_file, _ = temp_source_file
        
        # Mock the chain components
        mock_bm25.from_documents.return_value = MagicMock()
        mock_ensemble.return_value = MagicMock()
        mock_comp_retr.return_value = MagicMock()
        
        # Build retriever
        retriever = vector.get_advanced_retriever()
        
        # Should be a ChapterAwareRetriever
        assert isinstance(retriever, ChapterAwareRetriever)
    
    def test_multiple_chapter_detections_in_query(self):
        """Positive: Should handle query with multiple chapter mentions (uses first)."""
        # Regex finds first match
        result = detect_chapter("Chapter 1 and Chapter 5")
        # Should match the first occurrence
        assert result == "AN UNEXPECTED PARTY"
    
    def test_format_history_consistency(self, clear_chat_history):
        """Positive: Should produce consistent formatting."""
        main.chat_history.append(("Q1", "A1"))
        main.chat_history.append(("Q2", "A2"))
        
        result1 = format_history()
        result2 = format_history()
        
        assert result1 == result2


# ============================================================================
# EDGE CASES AND ERROR CONDITIONS
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_detect_chapter_boundary_values(self):
        """Positive: Should handle boundary chapter numbers."""
        assert detect_chapter("Chapter 1") == "AN UNEXPECTED PARTY"  # First
        assert detect_chapter("Chapter 19") == "THE LAST STAGE"  # Last
    
    def test_detect_chapter_numeric_edge_cases(self):
        """Negative: Should reject out-of-range chapters."""
        assert detect_chapter("Chapter 0") is None
        assert detect_chapter("Chapter -1") is None
        assert detect_chapter("Chapter 20") is None
    
    def test_format_docs_with_mixed_metadata(self):
        """Positive: Should handle docs with different metadata."""
        doc1 = Document(page_content="C1", metadata={"Chapter": "One"})
        doc2 = Document(page_content="C2", metadata={})
        doc3 = Document(page_content="C3", metadata={"Chapter": "Three", "page": 5})
        
        result = format_docs([doc1, doc2, doc3])
        assert "C1" in result
        assert "C2" in result
        assert "C3" in result
    
    def test_format_history_max_turns_enforcement(self, clear_chat_history):
        """Positive: Deque should enforce max length."""
        # Add more turns than MAX_HISTORY_TURNS
        for i in range(main.MAX_HISTORY_TURNS + 5):
            main.chat_history.append((f"Q{i}", f"A{i}"))
        
        # Should only keep latest MAX_HISTORY_TURNS
        assert len(main.chat_history) == main.MAX_HISTORY_TURNS
        assert main.chat_history[0][0] == f"Q5"  # First of 5 remaining


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
