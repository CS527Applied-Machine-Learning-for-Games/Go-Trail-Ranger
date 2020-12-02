# Define an AI Class for dealing with the Game
class Agent:
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, inputs, hidden):
        output, (hx, cx) = self.brain(inputs, hidden)
        actions = self.body(output)
        return actions.data.cpu().numpy(), (hx, cx)