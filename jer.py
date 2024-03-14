from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from langchain.chains import LLMChain  # Update with the correct import based on your langchain package
from langchain.prompts import PromptTemplate  # Update with the correct import based on your langchain package
from langchain_groq import ChatGroq  # Update with the correct import based on your langchain package


groq_api_key = os.getenv("GROQ_API_KEY")


class UserRequest(BaseModel):
    query: str


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "plswork!"}




@app.post("/route/")
async def process_request(request: UserRequest):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name='mixtral-8x7b-32768')


    query = request.query

    prompt_template = """
     Please carefully analyze the following request and create an enhanced prompt that addresses the key elements while providing additional context and instructions to optimize the output of a large language model (LLM). Consider the subject matter and intent of the original request, and preface your revised prompt with a relevant expert persona that the LLM should embody when responding. Aim to make the prompt clear, specific, and well-structured to elicit a high-quality and coherent response from the LLM.

      In your revised prompt, be sure to:

      Identify the core objective and any subtasks
      Provide necessary background information or definitions
      Specify the desired format, length, and style of the response
      Include any examples, constraints, or special instructions
      Adopt an appropriate expert tone based on the request's domain
      Once you have carefully constructed an improved version of the original prompt designed to optimize the LLM's comprehension and output quality, please provide the completed enhanced prompt, followed by the response that the enhanced prompt generates from the LLM.

      Response from updated prompt:

      Expert Prompt Engineer: Here is an enhanced prompt I've created based on your original request:

      Subject Matter Expert in [Relevant Domain]: You are an expert in [specific field related to the request]. Carefully review the following request and provide a thorough, well-structured response that addresses all key aspects of the query. Aim for a [desired length] response written in a [specified style/tone].

      In your response, be sure to:

    Clearly explain the main concepts, processes or recommendations relevant to answering the core question or completing the primary objective
    Break your response into logical sections with headers, bulleted lists, or numbered steps as appropriate
    Define any technical terms or provide brief background information as needed for clarity
    Include specific examples, evidence, or references to support your points
    Offer actionable advice, solutions or next steps if applicable
    Use your expert knowledge to address any implicit elements of the request
    Original QUERY: {query}

    Please provide your expert response adhering to the specifications outlined above.

    When you receive this enhanced prompt, adopt the role of the specified subject matter expert and provide a response tailored to the original request, following the structure and guidelines detailed in the prompt. The response should demonstrate deep subject knowledge while remaining clear and accessible.
    """


# Define the prompt structure
    prompt = PromptTemplate(
    input_variables=["query"],
    template=prompt_template,
)




    llm_chain = LLMChain(llm=llm, prompt=prompt)


    # Pass the context and question to the Langchain chain
    result_chain = llm_chain.invoke({"query": query})


    return result_chain


if __name__ == "__main__":
        uvicorn.run(app)
