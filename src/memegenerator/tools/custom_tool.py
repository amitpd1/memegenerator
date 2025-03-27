from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv
load_dotenv()


gemini_api_key = os.environ.get("GEMINI_API_KEY")

class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""
    prompt: str = Field(..., description="This is the text prompt used to generate image")

class ImageGeneratorTool(BaseTool):
    name: str = "Image Generator Tool"
    description: str = (
        "This is a helpful tool to generate images from text prompt."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, prompt: str) -> str:
        try:
            client = genai.Client(api_key=gemini_api_key)
            contents = prompt
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp-image-generation",
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=['Text', 'Image']
                )
            )

            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    print(part.text)
                elif part.inline_data is not None:
                    image = Image.open(BytesIO((part.inline_data.data)))
                    image_filename = f"gemini-native-image_{os.urandom(4).hex()}.png"
                    image.save(image_filename)

            return "this is an example of a tool output, ignore it and move along."
        
        except Exception as e:
            return f"Error during image generation: {e}"

