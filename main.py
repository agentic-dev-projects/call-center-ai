from pipeline.orchestrator import run_pipeline

if __name__ == "__main__":

    input_audio = "data/sample_audio/sample.mp3"

    result = run_pipeline(input_audio)

    if isinstance(result, dict):
        print(result)
    else:
        print(result.model_dump(mode="json"))