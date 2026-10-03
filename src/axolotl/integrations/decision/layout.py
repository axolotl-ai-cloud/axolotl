from dataclasses import dataclass


@dataclass(frozen=True)
class CanvasLayout:
    width: int = 128
    thought_open: tuple[int, ...] = ()
    thought_close: tuple[int, ...] = ()
    turn_close: tuple[int, ...] = ()
    pad_id: int = 0

    def build(self, prompt_ids, answer_ids, slots=()):
        canvas = (
            list(self.thought_open)
            + list(slots)
            + list(self.thought_close)
            + list(answer_ids)
            + list(self.turn_close)
        )
        if len(canvas) > self.width:
            raise ValueError("decision canvas exceeds width")
        n = len(canvas)
        canvas += [self.pad_id] * (self.width - n)
        return canvas, n
