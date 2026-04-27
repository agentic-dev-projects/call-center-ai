from agents.base_agent import BaseAgent


class TestAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="TestAgent")

    def process(self, input_data):
        return {"message": f"Hello {input_data}"}


if __name__ == "__main__":
    agent = TestAgent()
    result = agent.run("Bijan")
    print(result)