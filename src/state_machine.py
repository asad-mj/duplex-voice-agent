from enum import Enum
from pydantic import BaseModel

class DuplexState(str, Enum):
    IDLE = "IDLE"
    USER_SPEAKING = "USER_SPEAKING"
    AGENT_SPEAKING = "AGENT_SPEAKING"
    INTERRUPTED = "INTERRUPTED"

class SessionStepResult(BaseModel):
    state: DuplexState
    barge_in_triggered: bool
    cancel_agent_playback: bool
    interruption_count: int
    consecutive_speech_frames: int

class DuplexSessionController:
    """
    Deterministic finite-state machine managing full-duplex conversational turns
    and mid-utterance user barge-in interruptions.
    """

    def __init__(self, barge_in_frame_trigger: int = 2, silence_hangover_frames: int = 3):
        self.state: DuplexState = DuplexState.IDLE
        self.barge_in_frame_trigger = barge_in_frame_trigger
        self.silence_hangover_frames = silence_hangover_frames

        self.consecutive_speech = 0
        self.consecutive_silence = 0
        self.interruption_count = 0

    def start_agent_response(self) -> None:
        """Transitions session into active agent audio output state."""
        self.state = DuplexState.AGENT_SPEAKING
        self.consecutive_speech = 0
        self.consecutive_silence = 0

    def process_vad_decision(self, is_speech: bool) -> SessionStepResult:
        barge_in = False
        cancel_playback = False

        if is_speech:
            self.consecutive_speech += 1
            self.consecutive_silence = 0

            if self.state == DuplexState.AGENT_SPEAKING:
                if self.consecutive_speech >= self.barge_in_frame_trigger:
                    self.state = DuplexState.INTERRUPTED
                    self.interruption_count += 1
                    barge_in = True
                    cancel_playback = True
            elif self.state in (DuplexState.IDLE, DuplexState.INTERRUPTED):
                self.state = DuplexState.USER_SPEAKING
        else:
            self.consecutive_silence += 1
            self.consecutive_speech = 0

            if self.state in (DuplexState.USER_SPEAKING, DuplexState.INTERRUPTED):
                if self.consecutive_silence >= self.silence_hangover_frames:
                    self.state = DuplexState.IDLE

        return SessionStepResult(
            state=self.state,
            barge_in_triggered=barge_in,
            cancel_agent_playback=cancel_playback,
            interruption_count=self.interruption_count,
            consecutive_speech_frames=self.consecutive_speech
        )
