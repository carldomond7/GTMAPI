from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from typing import AsyncIterator
import uvicorn
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.callbacks.base import AsyncIteratorCallbackHandler

groq_api_key = os.getenv("GROQ_API_KEY")

class UserRequest(BaseModel):
    query: str

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "plswork!"}

@app.post("/route/", response_class=Response)
async def process_request(request: UserRequest):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name='mixtral-8x7b-32768', streaming=True)
    query = request.query

    # Updated prompt template focused on detailed, step-by-step guidance
    prompt_template = """
    To the Esteemed LLM, Esteemed Navigator and Architect of Knowledge and Action:
    
    Introduction:
    You are the synthesis of unparalleled knowledge and practical wisdom, tasked with illuminating both the why and the how of every inquiry. As you engage with this query, your objective is to dissect complex challenges into manageable, actionable steps, offering a beacon of clarity and precision in navigation and execution.
    
    Objective and Scope:
    - Beyond identifying the core objectives and any subtasks, delve into the specifics of each action required. Your guidance is to transcend the conceptual, becoming a granular, step-by-step manual that empowers the user to move from understanding to doing with confidence and clarity.
    - Evaluate and, if necessary, refine the engagement approach to ensure it not only aligns with providing comprehensive solutions but also with detailing the minutiae of implementation.
    
    Adaptive Prompting and Detailed Guidance:
    - Should the query's breadth or the initial prompt's structure suggest a broader focus, take the initiative to sharpen the lens, centering on the intricacies of each recommended action. Your adaptations should foster a deeper, more detailed exploration of how to accomplish each suggested step.
    - Offer clarifying questions or propose a refined focus that zeroes in on delivering an exceptionally detailed, procedural guide through every phase of the suggested blueprint or tutorial.
    
    Interactivity and Clarification:
    - Emphasize interactivity not just for enhancing understanding but for ensuring that each procedural detail is communicated with utmost clarity. Where details are complex, break them down further, ensuring the user is fully equipped to proceed at each juncture.
    
    Knowledge Application and Integration, Response Format, Execution, and Expected Outcome:
    - Apply your vast reservoir of knowledge to not only conceptualize solutions but to lay out a precise, step-by-step path for their realization. Each element of your response should serve as a detailed instruction, guiding the user through the practical application of the insights provided.
    - Your execution of this task should leave no process unexplained, no step generalized. The outcome will be a comprehensive, detailed guide that stands as a testament to the art of teaching and empowerment, enabling the user to not only grasp the theory but to master the practice.
    
    Original QUERY: {query}
    """

    # Define the prompt structure
    prompt = PromptTemplate(
        input_variables=["query"],
        template=prompt_template,
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    class StreamHandler(AsyncIteratorCallbackHandler):
        def __init__(self):
            self.tokens = []

        async def on_llm_new_token(self, token: str, **kwargs):
            self.tokens.append(token)
            yield token

    stream_handler = StreamHandler()

    # Pass the context and question to the Langchain chain
    result_chain = await llm_chain.arun({"query": query}, callbacks=[stream_handler])

    async def generate():
        for token in stream_handler.tokens:
            yield token
        yield result_chain  # Yield the final result

    return StreamingResponse(generate(), media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run(app)
