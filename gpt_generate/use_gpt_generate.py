import base64
import os
import re
from datetime import datetime

from openai import OpenAI

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)

# 1. Initialize the client (reads OPENAI_API_KEY from your environment)
client = OpenAI()

# 2. Path to your clothing photograph (one folder up)
IMAGE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "test_images/white_mercedes_shirt.jpg"
)

# 3. Prompt for professional catalog style
catalog_prompt = (
    "Extract the clothing item from this photo and completely remove its original background. "
    "Place the clothing neatly flat-laid with a fully transparent background (no backdrop, no shadow, no floor). "
    "Enhance the lighting to look like professional e-commerce catalog photography. "
    "Keep the texture and color of the original clothing identical."
)

print("Processing image and generating professional catalog view...")

# 4. Send the image and the instructions to the GPT image model
try:
    with open(IMAGE_PATH, "rb") as image_file:
        response = client.images.edit(
            model="gpt-image-1-mini",
            image=image_file,
            prompt=catalog_prompt,
            quality="medium",
            size="1024x1536",  # portrait, closest to 3:4
            background="transparent",
            output_format="png",
        )
except FileNotFoundError:
    print("Error: Could not find the clothing photo. Check IMAGE_PATH!")
    exit()

# 5. Save the result to a new timestamped file
output_name = os.path.join(
    os.path.dirname(__file__), f"gpt_{re.sub(r'_\d{10}_.*$', '', os.path.splitext(os.path.basename(IMAGE_PATH))[0])}_{datetime.now():%Y%m%d_%H%M%S}.png"
)
with open(output_name, "wb") as f:
    f.write(base64.b64decode(response.data[0].b64_json))
print(f"Success! Your professional catalog image is saved as {output_name}")
