class FakeMsg:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    def __init__(self, outputs):
        if isinstance(outputs, str):
            outputs = [outputs]
        self.outputs = outputs
        self.i = 0

    def invoke(self, prompt: str):
        out = self.outputs[min(self.i, len(self.outputs) - 1)]
        self.i += 1
        return FakeMsg(out)
