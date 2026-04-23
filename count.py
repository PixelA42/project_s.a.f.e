import os

human_path = r"Testing_data\ai"
ai_path = r"Testing_data\human"

human_count = len([f for f in os.listdir(human_path) if f.endswith(('.wav', '.mp3'))])
ai_count = len([f for f in os.listdir(ai_path) if f.endswith(('.wav', '.mp3'))])

print(f"ai samples: {human_count}")
print(f"human samples: {ai_count}")