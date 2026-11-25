import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from voice_commands import Command, ToolControl, VoiceEvent, VoiceListener, parse_command


class CommandTests(unittest.TestCase):
    def test_colors_and_aliases(self):
        for color in ('white', 'green', 'blue', 'pink'):
            for prefix in ('change color to ', 'change colour to ', 'color ', 'change color '):
                self.assertEqual(parse_command(prefix + color), Command('color', color))
        self.assertEqual(parse_command('Canvas, please change color to BLUE!'), Command('color', 'blue'))

    def test_whole_phrases_only(self):
        for text in ('do not clear screen', 'I said highlight mode yesterday',
                     'clear screen confirm clear', 'blue', 'change color to orange'):
            self.assertIsNone(parse_command(text))

    def test_tool_commands(self):
        for phrase, action in [('highlight mode', 'highlight'), ('drawing mode', 'draw'),
                               ('eraser mode', 'erase')]:
            self.assertEqual(parse_command(phrase), Command('tool', action))


class ControlTests(unittest.TestCase):
    def test_original_gestures_preserved(self):
        c = ToolControl()
        for mode in ('draw', 'highlight', 'erase', 'clear', 'pause'):
            self.assertEqual(c.resolve(mode), mode)

    def test_voice_tool_persists(self):
        c = ToolControl()
        c.apply(parse_command('highlight mode'), 0)
        self.assertEqual(c.resolve('draw'), 'highlight')
        self.assertEqual(c.resolve('highlight'), 'highlight')
        self.assertEqual(c.resolve('pause'), 'pause')
        self.assertEqual(c.resolve('erase'), 'erase')
        self.assertEqual(c.resolve('clear'), 'clear')
        self.assertEqual(c.resolve('draw', False), 'pause')
        c.apply(parse_command('drawing mode'), 1)
        self.assertEqual(c.resolve('highlight'), 'draw')
        c.apply(parse_command('gesture mode'), 2)
        self.assertEqual(c.resolve('highlight'), 'highlight')

    def test_clear_requires_fresh_confirmation_and_pauses(self):
        c = ToolControl()
        self.assertFalse(c.apply(parse_command('clear screen'), 10)[1])
        self.assertEqual(c.resolve('draw'), 'pause')
        self.assertFalse(c.apply(parse_command('confirm clear'), 11, captured_at=9)[1])
        self.assertTrue(c.apply(parse_command('confirm clear'), 12, captured_at=11)[1])
        self.assertFalse(c.apply(parse_command('confirm clear'), 13)[1])

    def test_clear_expiration_and_cancel(self):
        c = ToolControl()
        c.apply(parse_command('clear screen'), 0)
        self.assertFalse(c.apply(parse_command('confirm clear'), 11)[1])
        c.apply(parse_command('clear screen'), 12)
        c.apply(parse_command('cancel'), 13)
        self.assertFalse(c.apply(parse_command('confirm clear'), 14)[1])

    def test_other_command_cancels_pending_clear(self):
        c = ToolControl()
        c.apply(parse_command('highlight mode'), 0)
        c.apply(parse_command('clear screen'), 1)
        c.apply(parse_command('color blue'), 2)
        self.assertFalse(c.pending_clear)
        self.assertEqual(c.tool, 'highlight')


class ListenerTests(unittest.TestCase):
    def test_worker_transcribes_without_ui_calls_and_reports_device_failure(self):
        microphone = MagicMock()
        recognizer = MagicMock()
        audio = SimpleNamespace(frame_data=b'0' * 3200, sample_rate=16000, sample_width=2)
        recognizer.listen.side_effect = [audio, OSError('device disconnected')]
        recognizer.recognize_google.return_value = 'highlight mode'
        fake_sr = SimpleNamespace(
            Recognizer=lambda: recognizer, Microphone=lambda **kw: microphone,
            WaitTimeoutError=type('WaitTimeoutError', (Exception,), {}),
            UnknownValueError=type('UnknownValueError', (Exception,), {}),
            RequestError=type('RequestError', (Exception,), {}))
        with patch.dict(sys.modules, {'speech_recognition': fake_sr}):
            listener = VoiceListener()
            listener.start()
            listener._thread.join(timeout=2)
            events = listener.poll()
        self.assertTrue(any(e.kind == 'text' and e.text == 'highlight mode' for e in events))
        self.assertTrue(any(e.kind == 'error' for e in events))
        recognizer.recognize_google.assert_called_once_with(audio, language='en-US')
        self.assertFalse(listener.enabled)

    def test_disabled_listener_does_not_import_audio(self):
        with patch.dict(sys.modules, {'speech_recognition': None}):
            listener = VoiceListener()
            self.assertFalse(listener.enabled)
            self.assertEqual(listener.poll(), [])

    def test_missing_dependency_does_not_crash(self):
        with patch.dict(sys.modules, {'speech_recognition': None}):
            listener = VoiceListener()
            listener.start()
            listener._thread.join(timeout=2)
            events = listener.poll()
            self.assertEqual(events[-1].kind, 'error')
            self.assertFalse(listener.enabled)

    def test_stopped_listener_discards_late_results(self):
        listener = VoiceListener()
        listener._post('text', 'clear screen')
        listener.stop()
        listener._post('text', 'confirm clear')
        self.assertEqual(listener.poll(), [])


class AppIntegrationTests(unittest.TestCase):
    def setUp(self):
        from main import GesturePainting
        self.app = GesturePainting()
        self.app.board.auto_shapes = False

    def send(self, text, now, captured_at=None):
        self.app.voice.events.put(VoiceEvent('text', text,
            now if captured_at is None else captured_at, now))
        self.app.handle_voice(now)

    def test_voice_color_and_mode_change(self):
        self.send('highlight mode', 10)
        self.send('change color to blue', 11)
        self.assertEqual(self.app.control.resolve('draw'), 'highlight')
        self.assertEqual(self.app.board.color, (255, 170, 60))

    def test_clear_changes_canvas_only_after_confirmation(self):
        self.app.board.stroke((300, 300), 'draw')
        self.app.board.stop()
        self.send('clear screen', 10)
        self.assertTrue(self.app.board.ink_mask.any())
        self.send('confirm clear', 12, captured_at=11)
        self.assertFalse(self.app.board.ink_mask.any())

    def test_stale_recognition_is_ignored(self):
        self.app.voice.events.put(VoiceEvent('text', 'highlight mode', 0, 1))
        self.app.handle_voice(10)
        self.assertIsNone(self.app.control.tool)

    def test_voice_errors_leave_gestures_usable(self):
        self.app.voice.events.put(VoiceEvent('error', 'Test microphone error', 1, 1))
        self.app.handle_voice(1)
        self.assertEqual(self.app.control.resolve('draw'), 'draw')

    def test_full_frame_loop_and_cleanup_without_devices(self):
        import cv2
        import numpy as np
        self.app.camera = MagicMock()
        self.app.camera.read.return_value = (True, np.zeros((720, 1280, 3), np.uint8))
        self.app.tracker = MagicMock()
        self.app.tracker.trackFingers.return_value = (None, [])
        with patch.object(self.app, 'start_webcam'), patch('cv2.imshow') as show, \
                patch('cv2.waitKey', return_value=ord('q')), patch('cv2.destroyAllWindows'):
            self.app.run()
        self.assertEqual(show.call_args.args[1].shape, (785, 1280, 3))
        self.app.camera.release.assert_called_once()
        self.app.tracker.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
