class TypedInput:
    def __init__(self, field):
        self.field = field

    def read(self) -> str:
        return self.field.stringValue().strip()
