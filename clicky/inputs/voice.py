from ..voice import apply_spellings, transcribe


class VoiceInput:
    def __init__(self, recording, provider, spellings="", language="default"):
        self.recording = recording
        self.provider = provider
        self.spellings = spellings
        self.language = language

    def read(self) -> str:
        try:
            transcript = transcribe(self.recording.path, self.provider, self.language)
            return apply_spellings(transcript, self.spellings)
        finally:
            self.recording.close()
