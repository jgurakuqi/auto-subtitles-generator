class VocalSeparatorConfig:

    overlap: float
    segment: float

    def __init__(self, overlap: float, segment: float) -> None:
        self.overlap = overlap
        self.segment = segment
