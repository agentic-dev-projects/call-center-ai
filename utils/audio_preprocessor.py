import subprocess


def preprocess_audio(input_path: str, output_path: str):
    """
    Convert audio to 16kHz mono WAV using ffmpeg
    """

    command = [
        "ffmpeg",
        "-i", input_path,
        "-ac", "1",        # mono
        "-ar", "16000",    # 16kHz
        "-y",
        output_path
    ]

    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path