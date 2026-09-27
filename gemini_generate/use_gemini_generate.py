import os
import re

from google import genai
from google.genai import types
from PIL import Image
from datetime import datetime

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# 1. Initialize the client (reads your GEMINI_API_KEY)
client = genai.Client()

# 2. Load your self-taken clothing photograph
try:
    IMAGE_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..",
        "test_images/white_mercedes_shirt.jpg",
    )
    source_image = Image.open(IMAGE_PATH)
except FileNotFoundError:
    print("Error: Could not find the clothing photo. Place it in the same folder as this script!")
    exit()

# 3. Create a highly descriptive prompt for professional catalog style
catalog_prompt = (
    "Extract the clothing item from this photo and completely remove its original background. "
    "Place the clothing neatly flat-laid on a clean, softly lit, neutral light-gray studio backdrop. "
    "Enhance the lighting to look like professional e-commerce catalog photography with subtle, "
    "realistic soft shadows underneath the fabric. Keep the texture and color of the original clothing identical."
)

print("Processing image and generating professional catalog view...")

# 4. Call gemini-3.1-flash-image passing both the image and the text instructions
response = client.models.generate_content(
    model="gemini-3.1-flash-lite-image",
    contents=[source_image, catalog_prompt],
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio="3:4",  # Standard commercial/apparel catalog dimension
        ),
    ),
)

# 5. Extract and save the new asset
for part in response.parts:
    if part.inline_data is not None:
        catalog_image = part.as_image()
        output_name = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"gemini_{re.sub(r'_\d{10}_.*$', '', os.path.splitext(os.path.basename(IMAGE_PATH))[0])}_{datetime.now():%Y%m%d_%H%M%S}.png",
        )
        catalog_image.save(output_name)
        print(f"Success! Your professional catalog image is saved as {output_name}")
