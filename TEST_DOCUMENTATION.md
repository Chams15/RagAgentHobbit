# RAG Chatbot Test Suite Documentation

## Overview

This document describes the comprehensive test suite for the RAG (Retrieval-Augmented Generation) Chatbot built with LangChain, Ollama, and ChromaDB. The tests are organized using PyTest and cover both positive and negative test cases.

## Running the Tests

### Run all tests:
```bash
pytest test_rag_chatbot.py -v
```

### Run specific test class:
```bash
pytest test_rag_chatbot.py::TestDetectChapter -v
```

### Run specific test:
```bash
pytest test_rag_chatbot.py::TestDetectChapter::test_detect_chapter_with_numeric_valid -v
```

### Run with coverage report:
```bash
pytest test_rag_chatbot.py --cov=. --cov-report=html -v
```

### Run integration tests (requires Ollama):
```bash
pytest test_rag_integration.py -v
```

### Run all tests (unit + integration):
```bash
pytest test_rag_chatbot.py test_rag_integration.py -v
```

### Run all tests with coverage:
```bash
pytest --cov=. --cov-report=html test_rag_chatbot.py test_rag_integration.py -v
```

---

## Test Coverage Summary

### 1. **TestDetectChapter** (9 tests)
**Purpose**: Validate chapter detection from user queries

**Positive Cases**:
- `test_detect_chapter_with_numeric_valid`: Valid chapter numbers 1-19
- `test_detect_chapter_with_roman_numerals`: Roman numeral detection (I-XIX)
- `test_detect_chapter_mixed_case`: Case-insensitive matching
- `test_detect_chapter_abbreviated`: Abbreviated forms ("ch", "ch.")
- `test_detect_chapter_false_positive_avoidance`: Word boundary protection

**Negative Cases**:
- `test_detect_chapter_no_chapter_mentioned`: Query without chapter reference
- `test_detect_chapter_invalid_number`: Out-of-range chapter numbers (0, 20+)
- `test_detect_chapter_invalid_roman`: Invalid Roman numerals (XX, XXX)

**Key Assertions**:
- Returns correct chapter title for valid inputs
- Returns `None` for invalid/missing chapter references
- Handles mixed case and abbreviated formats

---

### 2. **TestFormatDocs** (5 tests)
**Purpose**: Validate document formatting and deduplication

**Positive Cases**:
- `test_format_docs_single_document`: Single document formatting
- `test_format_docs_multiple_documents`: Multiple documents with separators
- `test_format_docs_deduplication`: Duplicate content removal
- `test_format_docs_whitespace_handling`: Preserves whitespace in content

**Negative Cases**:
- `test_format_docs_empty_list`: Empty document list handling

**Key Assertions**:
- Proper separator placement (`\n\n---\n\n`)
- Deduplication maintains order
- Handles empty input gracefully

---

### 3. **TestFormatHistory** (5 tests)
**Purpose**: Validate conversation history formatting

**Positive Cases**:
- `test_format_history_empty`: Empty history returns default message
- `test_format_history_single_turn`: Single Q&A formatting
- `test_format_history_multiple_turns`: Multiple turns with proper formatting
- `test_format_history_truncates_long_answers`: Long answers truncated to `MAX_HISTORY_ANSWER_CHARS`
- `test_format_history_respects_max_turns`: Enforces `MAX_HISTORY_TURNS` limit

**Key Assertions**:
- Format: "User: {question}\nAssistant: {answer}"
- Truncates answers > 500 characters
- Keeps only last 4 conversation turns
- Returns "No previous conversation." for empty history

---

### 4. **TestVectorStoreOperations** (7 tests)
**Purpose**: Validate vector store initialization and data loading

**Positive Cases**:
- `test_get_embeddings_success`: Successful embedding initialization
- `test_load_raw_text_success`: File reading and parsing
- `test_load_raw_text_caching`: LRU cache functionality
- `test_get_parent_docs_parsing`: Document parsing by chapter headers
- `test_check_source_file_freshness_no_change`: No warning when file unchanged
- `test_check_source_file_freshness_warns_on_change`: Warns on file modification

**Negative Cases**:
- `test_get_embeddings_failure`: Handles Ollama connection failures

**Key Assertions**:
- Embeddings initialized with correct model name
- Raw text loaded with correct encoding
- Cache provides same object reference
- Metadata contains chapter information
- Modification detection working

---

### 5. **TestChapterAwareRetriever** (3 tests)
**Purpose**: Validate chapter-aware filtering

**Positive Cases**:
- `test_chapter_aware_retriever_no_chapter_specified`: Returns all docs when no chapter specified
- `test_chapter_aware_retriever_filters_by_chapter`: Filters results by detected chapter
- `test_chapter_aware_retriever_fallback_to_all`: Falls back to all when filtering finds nothing

**Key Assertions**:
- Filtering works when chapter is detected
- Fallback mechanism prevents empty results
- Metadata matching is case-insensitive

---

### 6. **TestInputValidation** (4 tests)
**Purpose**: Validate input handling and edge cases

**Positive Cases**:
- `test_detect_chapter_with_extra_whitespace`: Handles extra whitespace
- `test_detect_chapter_at_start_and_middle`: Finds chapter reference anywhere
- `test_format_history_with_special_characters`: Handles special characters
- `test_get_book_metadata_returns_string`: Returns valid metadata

**Key Assertions**:
- Robust whitespace handling
- Chapter detection location-independent
- Special characters preserved
- Metadata contains required fields

---

### 7. **TestCleanup** (3 tests)
**Purpose**: Validate resource cleanup and cache clearing

**Positive Cases**:
- `test_cleanup_with_empty_resources`: Handles empty resource dict
- `test_cleanup_clears_caches`: Successfully clears all LRU caches
- `test_cleanup_handles_vectorstore_errors`: Handles vectorstore deletion errors

**Key Assertions**:
- No exceptions raised during cleanup
- All caches cleared
- Graceful error handling

---

### 8. **TestIntegration** (3 tests)
**Purpose**: Integration tests for the complete system

**Positive Cases**:
- `test_build_retriever_end_to_end`: Complete retriever chain construction
- `test_multiple_chapter_detections_in_query`: Handles multiple chapter mentions
- `test_format_history_consistency`: Consistent formatting across calls

**Key Assertions**:
- All components properly integrated
- Returns ChapterAwareRetriever instance
- Deterministic output

---

### 9. **TestEdgeCases** (5 tests)
**Purpose**: Boundary conditions and edge cases

**Positive Cases**:
- `test_detect_chapter_boundary_values`: First and last chapters (1, 19)
- `test_format_docs_with_mixed_metadata`: Inconsistent metadata handling
- `test_format_history_max_turns_enforcement`: Deque max length enforcement

**Negative Cases**:
- `test_detect_chapter_numeric_edge_cases`: Out-of-range values (-1, 0, 20+)

**Key Assertions**:
- Boundary chapters work correctly
- Invalid boundaries rejected
- Collection enforcement working

---

## Fixtures

### `temp_source_file`
Creates a temporary Hobbit markdown file with 3 chapters for testing without modifying the actual source file.

### `clear_caches`
Clears all LRU caches (`_load_raw_text`, `get_embeddings`, `_get_parent_docs`, `get_book_metadata`) before and after each test to ensure test isolation.

### `clear_chat_history`
Clears the conversation history for tests that modify the chat history deque.

---

## Mocking Strategy

- **Ollama**: Mocked to avoid dependency on running Ollama service
- **ChromaDB**: Mocked for vector store operations
- **Retrievers**: Mocked to test logic without building actual retrieval chains

---

## Test Statistics

| Category | Count |
|----------|-------|
| Total Tests | 44 |
| Positive Tests | 31 |
| Negative Tests | 13 |
| Test Classes | 9 |

---

## Integration Tests (test_rag_integration.py)

Separate integration test suite that validates the **actual RAG pipeline** with real Hobbit-based responses and queries.

### Test Coverage:

**Note**: Integration tests require Ollama to be running. Use `-m "not slow"` to skip slow tests.

### 1. **TestRAGAnswerQuality** (3 tests - @pytest.mark.slow)
**Purpose**: Validate answer quality for Hobbit-specific questions

- `test_rag_query_about_bilbo`: Responses about Bilbo contain hobbit-related keywords
- `test_rag_query_about_gandalf`: Gandalf responses are accurate
- `test_rag_query_about_dragon`: Smaug/dragon information is presented correctly

### 2. **TestHobbitChapters** (3 tests - @pytest.mark.slow)
**Purpose**: Test chapter-specific Hobbit content retrieval

- `test_chapter_1_unexpected_party`: Chapter 1 returns party-related content
- `test_chapter_5_riddles`: Chapter 5 returns riddle/underground content
- `test_quest_and_journey`: Quest purpose questions return treasure/mountain references

### 3. **TestHobbitPlotElements** (3 tests - @pytest.mark.slow)
**Purpose**: Key Hobbit plot elements are retrieved correctly

- `test_magic_ring_arc`: Ring/invisibility queries return relevant content
- `test_dwarves_and_quest`: Dwarf and Thorin references are accurate
- `test_mountain_and_treasure`: Mountain/treasure queries return content

### 4. **TestRAGPipeline** (3 tests - @pytest.mark.slow)
**Purpose**: Full Hobbit RAG pipeline execution

- `test_rag_chain_builds_successfully`: Complete chain builds without errors
- `test_rag_retrieval_returns_hobbit_docs`: Retriever returns Hobbit documents
- `test_rag_streaming_hobbit_query`: Streaming works for Bilbo character queries

### 5. **TestConversationContinuity** (1 test - @pytest.mark.slow)
**Purpose**: Multi-turn Hobbit discussion

- `test_hobbit_discussion_with_follow_up`: Bilbo discussion maintains conversation context

### 6. **TestRetrievalQuality** (2 tests - @pytest.mark.slow)
**Purpose**: Hobbit document retrieval quality

- `test_retriever_deduplicates_results`: No excessive duplicates in Bilbo results
- `test_retriever_respects_chapter_filter`: Chapter 1 filtering works correctly

### 7. **TestIntegrationErrorHandling** (2 tests - @pytest.mark.slow)
**Purpose**: Error handling with Hobbit queries

- `test_rag_handles_hobbit_typos`: Handles misspelled names (e.g., "Bagins")
- `test_rag_handles_out_of_scope_hobbit_query`: Handles non-Hobbit queries (e.g., LOTR)

### 8. **TestPerformance** (1 test - @pytest.mark.slow)
**Purpose**: Hobbit query response time

- `test_rag_hobbit_query_response_time`: Hobbit journey queries complete in <30s

### 9. **TestMockedRAGPipeline** (2 tests - **NO Ollama needed**)
**Purpose**: Mocked Hobbit tests for CI/CD

- `test_rag_chain_with_mock_hobbit_components`: Mock LLM with Hobbit documents
- `test_format_hobbit_retrieval_results`: Hobbit document formatting integration

---

## Integration Test Differences

| Feature | Unit Tests | Integration Tests |
|---------|-----------|-------------------|
| Ollama Required | ❌ No | ✅ Yes (for most) |
| Mocked Components | ✅ Yes | ❌ No (real components) |
| Answer Quality | ❌ Not tested | ✅ Yes (Hobbit-focused) |
| Full Chain | ❌ Not tested | ✅ Yes |
| Response Streaming | ❌ Not tested | ✅ Yes |
| Performance | ❌ Not tested | ✅ Yes |
| Conversation History | ❌ Not tested | ✅ Yes |
| Hobbit Content Accuracy | ❌ Not tested | ✅ Yes |
| Chapter Filtering | ❌ Not tested | ✅ Yes |
| Execution Time | Fast (< 1 sec) | Slow (5-30 sec per test) |

---

## Running Integration Tests

### All Hobbit RAG integration tests (requires Ollama):
```bash
pytest test_rag_integration.py -v
```

### Only Hobbit chapter tests:
```bash
pytest test_rag_integration.py::TestHobbitChapters -v
```

### Hobbit plot elements tests:
```bash
pytest test_rag_integration.py::TestHobbitPlotElements -v
```

### Mocked integration tests (no Ollama):
```bash
pytest test_rag_integration.py -m "not slow" -v
```

### Specific Hobbit character query test:
```bash
pytest test_rag_integration.py::TestRAGAnswerQuality::test_rag_query_about_bilbo -v
```

### CI/CD pipeline (mocked Hobbit tests only, no Ollama):
```bash
pytest test_rag_integration.py -m "not slow" -v
```

### All tests with performance check:
```bash
pytest test_rag_integration.py::TestPerformance -v
```

---

## Key Testing Principles

1. **Isolation**: Each test is independent via fixtures
2. **Mocking**: External dependencies mocked to isolate logic
3. **Coverage**: Positive and negative cases for all major functions
4. **Edge Cases**: Boundary values and error conditions tested
5. **Integration**: End-to-end system tests verify component interaction

---

## Common Test Patterns

### Testing with Mocks
```python
@patch('vector.OllamaEmbeddings')
def test_function(self, mock_ollama):
    mock_ollama.return_value = MagicMock()
    # Test code
```

### Testing Exceptions
```python
with pytest.raises(RuntimeError, match="error message"):
    function_that_raises()
```

### Testing Warnings
```python
with pytest.warns(UserWarning, match="warning text"):
    function_that_warns()
```

### Testing with Fixtures
```python
def test_function(self, temp_source_file, clear_caches):
    # temp_source_file and clear_caches are fixtures
    pass
```

---

## Extending the Test Suite

To add new tests:

1. Choose appropriate test class or create new class
2. Follow naming convention: `test_<function>_<scenario>`
3. Use descriptive docstrings
4. Add both positive and negative cases
5. Use appropriate fixtures for setup/teardown

---

## Troubleshooting

### Import Errors
Ensure `main.py` and `vector.py` are in the same directory as `test_rag_chatbot.py`

### Cache-Related Failures
Use `clear_caches` fixture to reset LRU caches between tests

### Mocking Issues
Verify mock paths match actual import statements in source files

### File Not Found
`temp_source_file` fixture creates temporary files; ensure cleanup happens

---

## Future Test Additions

Consider adding tests for:
- User acceptance testing with domain experts
- Semantic similarity scoring for answer validation
- Multi-language query support
- Concurrent query handling (stress testing)
- Large document processing (>100 chapters)
- Memory profiling during extended sessions
- Cost analysis for different embedding models
- A/B testing different retrieval strategies
