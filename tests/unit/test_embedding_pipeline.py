import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from api.chunker import (
    MAX_CHUNK_CHARACTERS,
    MAX_CHUNK_TOKENS,
    _approx_tokens,
    _chunk_text,
)
from api.config import _embedding_dimension
from api.data_pipeline import _cache_matches_fingerprint, embed_documents
from api.db.chroma_store import collection_name_for_repo
from api.db.meta_store import MetaStore
from api.types import Document


class ChunkingTests(unittest.TestCase):
    def assert_bounded(self, chunks):
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), MAX_CHUNK_CHARACTERS)
            self.assertLessEqual(_approx_tokens(chunk.text), MAX_CHUNK_TOKENS)

    def test_large_json_is_split_without_missing_content(self):
        text = '{"items":[' + ",".join(
            f'{{"id":{index},"value":"marker-{index}-' + "x" * 80 + '"}'
            for index in range(500)
        ) + "]}"

        chunks = _chunk_text(text, "large.json", "json")

        self.assertGreater(len(chunks), 1)
        self.assert_bounded(chunks)
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_oversized_single_line_is_split_without_missing_content(self):
        text = "".join(f"marker-{index:05d};" for index in range(4000))

        chunks = _chunk_text(text, "large.txt", "txt")

        self.assertGreater(len(chunks), 1)
        self.assert_bounded(chunks)
        self.assertEqual("".join(chunk.text for chunk in chunks), text)

    def test_large_function_and_source_preamble_are_preserved(self):
        assignments = [f'    marker_{index} = "' + "y" * 120 + '"' for index in range(300)]
        text = "import os\n\ndef first():\n" + "\n".join(assignments) + "\n\ndef second():\n    return 2\n"

        chunks = _chunk_text(text, "large.py", "py")
        combined = "\n".join(chunk.text for chunk in chunks)

        self.assertGreater(len(chunks), 1)
        self.assert_bounded(chunks)
        self.assertIn("import os", combined)
        self.assertIn("marker_0", combined)
        self.assertIn("marker_299", combined)
        self.assertIn("def second", combined)

    def test_normal_source_remains_bounded(self):
        text = "import os\n\ndef first():\n    return 1\n\ndef second():\n    return 2\n"

        chunks = _chunk_text(text, "normal.py", "py")

        self.assert_bounded(chunks)
        self.assertIn("import os", "\n".join(chunk.text for chunk in chunks))


class EmbeddingValidationTests(unittest.TestCase):
    def test_invalid_qwen_dimension_is_rejected(self):
        with patch.dict(os.environ, {"EMBEDDING_DIM": "2560"}):
            with self.assertRaisesRegex(ValueError, "not supported"):
                _embedding_dimension("Qwen/Qwen3-Embedding-8B")

    def test_response_count_mismatch_contains_chunk_context(self):
        class FakeEmbedder:
            model_kwargs = {"model": "test-model", "dimensions": 3}

            def __call__(self, input):
                return SimpleNamespace(
                    data=[SimpleNamespace(embedding=[1.0, 2.0, 3.0])]
                )

        documents = [
            Document(text="first", meta_data={"file_path": "a.py", "chunk_index": 0}),
            Document(text="second", meta_data={"file_path": "b.py", "chunk_index": 1}),
        ]

        with self.assertRaisesRegex(ValueError, "response count mismatch") as error:
            embed_documents(documents, FakeEmbedder(), batch_size=2)

        self.assertIn("a.py", str(error.exception))
        self.assertIn("b.py", str(error.exception))


class IndexCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.fingerprint = {
            "embedding_base_url": "https://api.siliconflow.cn/v1",
            "embedding_model": "Qwen/Qwen3-Embedding-8B",
            "embedding_dim": 1024,
            "embedding_normalize": True,
            "chunker_version": "bounded-v2",
            "index_schema_version": 1,
            "embedder_type": "direct",
        }

    def test_metadata_reports_changed_fingerprint_and_resets_index_state(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MetaStore(os.path.join(directory, "metadata.sqlite3"))
            store.upsert_repo("repo-id", directory, None, self.fingerprint)
            store.update_repo_sha("repo-id", "old-sha")
            store.replace_file_chunks(
                "repo-id",
                "src/a.py",
                "file-hash",
                [
                    {
                        "chunk_id": "chunk-1",
                        "content_hash": "content-hash",
                        "start_line": 1,
                        "end_line": 2,
                    }
                ],
            )

            self.assertEqual(store.incompatible_fields("repo-id", self.fingerprint), [])
            changed = dict(self.fingerprint, embedding_dim=4096)
            self.assertEqual(store.incompatible_fields("repo-id", changed), ["embedding_dim"])

            store.reset_index_state("repo-id")
            self.assertIsNone(store.get_repo("repo-id")["last_indexed_sha"])
            with store.connect() as connection:
                file_count = connection.execute(
                    "SELECT COUNT(*) FROM file_index WHERE repo_id = ?", ("repo-id",)
                ).fetchone()[0]
                chunk_count = connection.execute(
                    "SELECT COUNT(*) FROM chunk_index WHERE repo_id = ?", ("repo-id",)
                ).fetchone()[0]
            self.assertEqual(file_count, 0)
            self.assertEqual(chunk_count, 0)

    def test_pickle_cache_requires_exact_fingerprint(self):
        compatible_cache = {"embedding_fingerprint": dict(self.fingerprint)}
        legacy_cache = {"split_and_embed": []}
        changed_cache = {
            "embedding_fingerprint": dict(self.fingerprint, embedding_model="old-model")
        }

        self.assertTrue(_cache_matches_fingerprint(compatible_cache, self.fingerprint))
        self.assertFalse(_cache_matches_fingerprint(legacy_cache, self.fingerprint))
        self.assertFalse(_cache_matches_fingerprint(changed_cache, self.fingerprint))

    def test_chroma_collection_is_isolated_by_fingerprint(self):
        original = collection_name_for_repo("git@example.test:group/repo.git", self.fingerprint)
        changed = collection_name_for_repo(
            "git@example.test:group/repo.git",
            dict(self.fingerprint, embedding_dim=4096),
        )

        self.assertNotEqual(original, changed)


if __name__ == "__main__":
    unittest.main()