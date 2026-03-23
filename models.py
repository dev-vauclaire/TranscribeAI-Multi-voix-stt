from dataclasses import dataclass
from typing import Optional

@dataclass
class Segment:
    segment_id: int
    start: float
    end: float
    speaker: str
    text: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end - self.start
    
    def same_speaker(self, segment) -> bool:
        return self.speaker == segment.speaker
        