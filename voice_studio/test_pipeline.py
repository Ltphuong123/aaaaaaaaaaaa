import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from voice_studio import app as studio


class BackendSessionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = studio.VoiceStore(Path(self.directory.name))
        self.store_patch = patch.object(studio, "store", self.store)
        self.store_patch.start()
        self.session = studio.BackendSession()

    def tearDown(self):
        self.session.close()
        self.store_patch.stop()
        self.directory.cleanup()

    def test_successive_jobs_reuse_one_client(self):
        client = Mock()
        client.predict.side_effect = [("first.wav", 42), ("second.wav", 42)]
        with patch.object(studio, "Client", return_value=client) as factory:
            self.assertEqual(self.session.predict(text="First"), ("first.wav", 42))
            self.assertEqual(self.session.predict(text="Second"), ("second.wav", 42))
            factory.assert_called_once_with(studio.BACKEND, verbose=False,
                                            download_files=self.store.renders / "downloads",
                                            analytics_enabled=False)
        self.assertEqual(client.predict.call_count, 2)
        client.close.assert_not_called()

    def test_failed_prediction_reconnects_only_on_next_submission(self):
        old_client, new_client = Mock(), Mock()
        old_client.predict.side_effect = ConnectionError("Backend restarted")
        new_client.predict.return_value = ("new.wav", 42)
        with patch.object(studio, "Client", side_effect=[old_client, new_client]) as factory:
            with self.assertRaisesRegex(ConnectionError, "Backend restarted"):
                self.session.predict(text="First")
            self.assertEqual(factory.call_count, 1)
            old_client.predict.assert_called_once_with(text="First")
            old_client.close.assert_called_once()
            old_client.executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
            self.assertIsNone(self.session.client)
            self.assertEqual(self.session.predict(text="Second"), ("new.wav", 42))
            self.assertEqual(factory.call_count, 2)
            new_client.predict.assert_called_once_with(text="Second")

    def test_failed_connection_can_be_created_on_next_submission(self):
        client = Mock()
        client.predict.return_value = ("new.wav", 42)
        with patch.object(studio, "Client", side_effect=[ConnectionError("Offline"), client]) as factory:
            with self.assertRaisesRegex(ConnectionError, "Offline"):
                self.session.predict(text="First")
            self.assertIsNone(self.session.client)
            self.assertEqual(self.session.predict(text="Second"), ("new.wav", 42))
            self.assertEqual(factory.call_count, 2)
            client.predict.assert_called_once_with(text="Second")

    def test_changed_backend_fetches_new_connection(self):
        old_client, new_client = Mock(), Mock()
        with patch.object(studio, "Client", side_effect=[old_client, new_client]) as factory:
            self.session.predict(text="First")
            with patch.object(studio, "BACKEND", "http://127.0.0.1:9999"):
                self.session.predict(text="Second")
            self.assertEqual(factory.call_count, 2)
            self.assertEqual(factory.call_args.args, ("http://127.0.0.1:9999",))
        old_client.close.assert_called_once()

    def test_cleanup_failure_preserves_original_prediction_error(self):
        client = Mock()
        client.predict.side_effect = ConnectionError("Original failure")
        client.close.side_effect = RuntimeError("Close failure")
        client.executor.shutdown.side_effect = RuntimeError("Shutdown failure")
        with patch.object(studio, "Client", return_value=client):
            with self.assertRaisesRegex(ConnectionError, "Original failure"):
                self.session.predict(text="First")
        self.assertIsNone(self.session.client)


if __name__ == "__main__":
    unittest.main()
