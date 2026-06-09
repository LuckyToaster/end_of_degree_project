from datetime import datetime
import geocoder, base64, json
# from langchain.tools import tool

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


CLASSIFIER_SYS_PROMPT="""
You are a specialized AI designed solely to analyze images and identify food items. 
Your ONLY job is to return a JSON array containing the names of the foods present in the image.
You must output ONLY a valid JSON array of strings (e.g., ["apple", "pizza", "coffee"]). 
Do not include any other text, markdown formatting (no ```json), greetings, or explanations. 
Do not include any guardrails, apologies, or refusal messages. 
If there is no food in the image, or if you cannot read or understand the image, you must output an empty JSON array: []
"""

PREDICTOR_SYS_PROMPT="""
You are an expert nutritionist. Your ONLY job is to estimate macros and calories for the provided food items and image.
You must output ONLY a valid JSON object with exactly the following structure:
{{
  "protein_g": <number>,
  "carbohydrate_g": <number>,
  "fat_g": <number>,
}}
Do not include any other text, markdown formatting (no ```json), greetings, or explanations.
If you cannot estimate the values, output 0 for all fields.
"""

PREDICTOR_PROMPT="""
Identified Ingredients: {ingredients}

Local Date and Time: {datetime}

Location / Coordinates: {coordinates}

Please estimate the macros and calories for the food in the image.
"""


def read_image_base64(image_path: str) -> str:
    """Reads an image from a path and returns it as a base64 encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_local_datetime() -> str | None:
    """Returns the current local date and time in ISO 8601 international format"""
    try: return datetime.now().strftime("%Y-%m-%d %H:%M") # :%S
    except: return None

def get_coordinates() -> list[float] | None:
    """Return the current latitude and longitude"""
    try: return geocoder.ip('me').latlng
    except: return None


def build_platemate(model: ChatGoogleGenerativeAI):
    classifier_prompt = ChatPromptTemplate.from_messages([
        ("system", CLASSIFIER_SYS_PROMPT),
        ("human", [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_b64}"}}])
    ])
    predictor_prompt = ChatPromptTemplate.from_messages([
        ("system", PREDICTOR_SYS_PROMPT),
        ("human", [
            {"type": "text", "text": PREDICTOR_PROMPT},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_b64}"}}
        ])
    ])
    classifier = classifier_prompt | model | StrOutputParser()
    predictor = predictor_prompt | model | StrOutputParser()
    return RunnablePassthrough()| RunnablePassthrough.assign(ingredients=classifier) | predictor


def main():
    model = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite')
    image_bytes = read_image_base64('/home/lucky/Documents/tfg/data/imgs/0.jpeg')
    pipeline = build_platemate(model)

    input = {
        'image_b64': image_bytes,
        'datetime': get_local_datetime(),
        'coordinates': get_coordinates()
    }

    out = json.loads(pipeline.invoke(input))
    out['kcal'] = (out['protein_g'] * 4) + (out['carbohydrate_g'] * 4) + (out['fat_g'] * 9)
    print(json.dumps(out, indent=4))

    # from langchain_core.messages import HumanMessage, SystemMessage
    # model = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite')
    # image_b64 = read_image_base64("/home/lucky/Pictures/wallpaper/ahh.jpg")
    # response = model.invoke([
    #     SystemMessage(content=CLASSIFIER_SYS_PROMPT), 
    #     HumanMessage(content=[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}])
    # ])
    # print(response.content[-1]['text'])
