from agents.intake_agent import CallIntakeAgent

if __name__ == "__main__":
    agent = CallIntakeAgent()

    # Test JSON input
    input_data = {
        "transcript": "Agent: Hello. Customer: My internet is not working.",
        "agent_name": "Sarah",
        "customer_id": "C123",
        "duration_seconds": 120
    }

    result = agent.run(input_data)

    print(result.model_dump())