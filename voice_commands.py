"""Optional microphone input and testable voice/gesture control rules.

SpeechRecognition sends audio to Google when voice input is enabled. Imports
and microphone access are deferred, so gesture-only use needs no audio packages.
"""
from dataclasses import dataclass
from queue import Empty, Queue
import re
from threading import Event, Thread
import time


COLOR_NAMES = ('white', 'green', 'blue', 'pink')


@dataclass(frozen=True)
class Command:
    action: str
    value: str = ''


def parse_command(text):
    """Match whole commands, never action words inside ordinary conversation."""
    text = re.sub(r'[^a-z ]', ' ', text.lower())
    text = ' '.join(text.split())
    if text.startswith('canvas '):
        text = text[7:]
    if text.startswith('please '):
        text = text[7:]
    phrases = {
        'highlight mode': Command('tool', 'highlight'),
        'highlighter mode': Command('tool', 'highlight'),
        'drawing mode': Command('tool', 'draw'),
        'draw mode': Command('tool', 'draw'),
        'pen mode': Command('tool', 'draw'),
        'eraser mode': Command('tool', 'erase'),
        'erase mode': Command('tool', 'erase'),
        'gesture mode': Command('gesture'),
        'clear screen': Command('clear'),
        'clear the screen': Command('clear'),
        'clear board': Command('clear'),
        'clear the board': Command('clear'),
        'confirm clear': Command('confirm'),
        'cancel': Command('cancel'),
        'cancel clear': Command('cancel'),
    }
    if text in phrases:
        return phrases[text]
    match = re.fullmatch(
        r'(?:change (?:the )?(?:color|colour) to|(?:color|colour)|'
        r'change (?:color|colour)|set (?:the )?(?:color|colour) to) '
        r'(white|green|blue|pink)', text)
    return Command('color', match.group(1)) if match else None


class ToolControl:
    """Voice selects a persistent tool; fingers still start/stop its strokes."""
    def __init__(self, confirmation_seconds=10.0):
        self.tool = None
        self.confirmation_seconds = confirmation_seconds
        self.clear_requested_at = None
        self.clear_deadline = None

    def expire(self, now):
        if self.clear_deadline is not None and now >= self.clear_deadline:
            self.cancel()
            return True
        return False

    @property
    def pending_clear(self):
        return self.clear_deadline is not None

    def cancel(self):
        self.clear_requested_at = self.clear_deadline = None

    def apply(self, command, now, captured_at=None):
        """Return (message, clear_now). Mutations of the board stay in main.py."""
        expired = self.expire(now)
        if command.action == 'clear':
            self.clear_requested_at = now
            self.clear_deadline = now + self.confirmation_seconds
            return 'Say "confirm clear" or "cancel" within 10 seconds', False
        if command.action == 'confirm':
            # A confirmation must have been recorded AFTER the request was shown.
            fresh = captured_at is None or (
                self.clear_requested_at is not None and captured_at >= self.clear_requested_at)
            accepted = self.pending_clear and fresh
            if accepted:
                self.cancel()
                return 'Board cleared', True
            return ('Clear request expired' if expired else 'No fresh clear confirmation'), False
        if command.action == 'cancel':
            self.cancel()
            return 'Clear cancelled', False
        # A different intentional command also cancels a pending clear.
        self.cancel()
        if command.action == 'tool':
            self.tool = command.value
            return 'Selected tool: ' + command.value, False
        if command.action == 'gesture':
            self.tool = None
            return 'Gesture tool selection restored', False
        if command.action == 'color':
            return 'Pen color: ' + command.value, False
        return 'Command not supported', False

    def resolve(self, gesture, hand_present=True):
        if not hand_present or self.pending_clear:
            return 'pause'
        if self.tool is None:
            return gesture
        # Preserve palm wiping and two-palm clear, while pointing/two fingers
        # both use the voice-selected tool instead of overriding it.
        if gesture in ('draw', 'highlight'):
            return self.tool
        return gesture


@dataclass(frozen=True)
class VoiceEvent:
    kind: str
    text: str
    captured_at: float
    completed_at: float


class VoiceListener:
    """One background worker; all OpenCV/board updates happen on the UI thread."""
    def __init__(self, microphone_index=None, language='en-US'):
        self.microphone_index = microphone_index
        self.language = language
        self.events = Queue(maxsize=32)
        self._stop = Event()
        self._thread = None
        self._enabled = False

    @property
    def enabled(self):
        return self._enabled

    def _flush(self):
        while True:
            try:
                self.events.get_nowait()
            except Empty:
                return

    def start(self):
        if self._thread and self._thread.is_alive():
            return False
        self._flush()
        self._stop.clear()
        self._enabled = True
        self._thread = Thread(target=self._run, name='voice-listener', daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._enabled = False
        self._stop.set()
        self._flush()

    def poll(self):
        result = []
        while True:
            try:
                result.append(self.events.get_nowait())
            except Empty:
                return result

    def _post(self, kind, text, captured_at=None):
        if self._stop.is_set():
            return
        now = time.monotonic()
        event = VoiceEvent(kind, text, now if captured_at is None else captured_at, now)
        if self.events.full():
            try:
                self.events.get_nowait()
            except Empty:
                pass
        self.events.put_nowait(event)

    def _run(self):
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            recognizer.operation_timeout = 5
            recognizer.pause_threshold = 0.6
            with sr.Microphone(device_index=self.microphone_index) as source:
                self._post('status', 'Calibrating - keep quiet for one second')
                recognizer.adjust_for_ambient_noise(source, duration=1)
                while not self._stop.is_set():
                    self._post('status', 'Listening')
                    try:
                        audio = recognizer.listen(source, timeout=1, phrase_time_limit=4)
                    except sr.WaitTimeoutError:
                        continue
                    # Timestamp the captured phrase rather than the beginning
                    # of the silent wait, so a prompt can precede confirmation.
                    duration = len(audio.frame_data) / (audio.sample_rate * audio.sample_width)
                    captured_at = time.monotonic() - duration
                    if self._stop.is_set():
                        break
                    self._post('status', 'Recognizing')
                    try:
                        text = recognizer.recognize_google(audio, language=self.language)
                    except sr.UnknownValueError:
                        self._post('notice', 'Speech not understood - try again')
                    except sr.RequestError:
                        self._post('notice', 'Speech service unavailable - check internet')
                        self._stop.wait(1)
                    else:
                        self._post('text', text, captured_at)
        except (ImportError, AttributeError):
            self._post('error', 'Voice unavailable: install requirements-voice.txt')
        except Exception as exc:
            self._post('error', 'Microphone error: ' + str(exc)[:130])
        finally:
            self._enabled = False


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='List available microphone indexes')
    parser.add_argument('--list-microphones', action='store_true')
    args = parser.parse_args()
    if args.list_microphones:
        try:
            import speech_recognition as sr
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f'{index}: {name}')
        except (ImportError, AttributeError, OSError) as exc:
            parser.exit(1, f'Install voice dependencies and check microphone permissions: {exc}\n')
    else:
        parser.print_help()
