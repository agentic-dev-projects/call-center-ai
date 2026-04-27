from gtts import gTTS
import os

# Ensure directory exists
output_dir = "../data/sample_audio"
os.makedirs(output_dir, exist_ok=True)

file_path = os.path.join(output_dir, "sample.mp3")

text = """
Hello, thank you for calling customer support.
My internet has not been working for the past two days.
Can you please help me resolve this issue?
"""

tts = gTTS(text=text, lang='en')
tts.save(file_path)

print(f"Audio file generated at: {file_path}")