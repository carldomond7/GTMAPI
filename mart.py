from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

groq_api_key = os.getenv("GROQ_API_KEY")

class UserRequest(BaseModel):
    information: str 

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "plswork!"}

@app.post("/route/")
async def process_request(request: UserRequest):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name='mixtral-8x7b-32768')

    information = request.information

    prompt_template =   f"""
        Based on the information provided use the following guidelines to generate the response:
        information: {information}

    guidelines:
    # Objective
    Provide a detailed explanation of the objective, including its immediate goals and long-term implications. Specify how the information will be utilized and its potential impact.

    # Background Context
    Offer an exhaustive background, touching on all relevant aspects of the topic. Include historical development, current status, and key players.

    # Instructions for Comprehensive Coverage
    - **Topic Exploration**: Direct the LLM to explore the topic from multiple dimensions, ensuring a rounded perspective.
    - **Case Studies and Examples**: Demand the inclusion of detailed case studies or examples relevant to the topic.
    - **Data-Driven Insights**: Stress the importance of including statistics, research findings, and trend analysis.
    - **Expert Opinions**: Request insights from experts or authoritative sources to lend credibility and depth.

    # Structuring for Clarity
    - **Complex Structuring**: Outline a complex structure with clear sections dedicated to different aspects of the topic.
    - **Analysis and Discussion**: Each section should offer not just information but also analysis, linking different pieces of data and discussing their implications.
    - **Future Directions**: Include a section on future trends, potential developments, and speculative insights based on current data.

    # Ethical and Responsible Use
    Emphasize ethical considerations, especially in handling sensitive information, predictions, and data interpretation.
    """

    # Define the prompt structure
    prompt = PromptTemplate(
        input_variables=["information"],
        template=prompt_template,
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Pass the context and question to the Langchain chain
    result_chain = llm_chain.invoke({"information": information})

    return result_chain
