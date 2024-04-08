from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

groq_api_key = os.getenv("GROQ_API_KEY")

class UserRequest(BaseModel):
    query: str
    result: str

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "plswork!"}

@app.post("/route/")
async def process_request(request: UserRequest):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name='mixtral-8x7b-32768')

    query = request.query
    result = request.result

    prompt_template = f"""
    
    Use the following query, result, and instructions to create the Template below:
    query: {query}
    result: {result}
    instructions: Approach this template with a focus on utilizing your expert judgment and analytical capabilities. Wherever the template requires information, endeavor to provide accurate and fact-based content. Make informed assumptions for aspects that are generally recognized or can be substantiated with factual evidence. 
    When faced with uncertainties:
    - If there's substantial evidence to support a reasoned assumption, proceed to include it in the response. Your aim is to fill gaps with educated guesses that align closely with available data or recognized patterns.
    - If evidence is insufficient to form a confident assumption, refrain from making speculative inferences. Instead, mark these areas clearly within the document.
    Exception to Inference:
    - Avoid making inferences or assumptions in situations where the required information is highly specific, unique, or cannot be reasonably deduced through available facts or general knowledge. This includes proprietary data, personal information, or context-specific details that are not publicly available or commonly understood.
    Reporting Nuances and Assumptions:
    - At the end of the document, include a dedicated section titled 'Areas Requiring Further Input'. In this section, list all areas where assumptions were made, along with a brief explanation of the rationale behind each assumption.
    - Additionally, identify areas where specific, nuanced information is necessary but could not be assumed. Provide a clear explanation for why these elements could not be inferred and highlight the need for direct input from relevant sources.
    Your goal is to produce a comprehensive and informative response that is as accurate and complete as possible, given the constraints of available information and the scope of general knowledge.
    
This is the Template you need to fill out using query, result, and instructions: 
Title Page
Report Title
Prepared for: [Client/Organization Name]
Prepared by: [Your Name/Team Name]
Date

Table of Contents
Automatically generated and includes all headings and subheadings with page numbers.

Executive Summary
Brief overview of the strategy, key action points, expected outcomes, and benefits.

Introduction
Purpose: Outline the purpose of the document and the specific task or problem it addresses.
Scope: Define the scope of the strategy, including what is covered and any limitations.
Objectives: List the objectives that the strategy aims to achieve.

Situation Analysis
Current Situation: Overview of the current status, highlighting relevant background information and the need for action.
SWOT Analysis: Strengths, Weaknesses, Opportunities, and Threats related to the task or project.
Stakeholder Analysis: Identification of all stakeholders and their roles, interests, and potential impact on the project.

Strategy Overview
Strategic Approach: High-level description of the chosen strategic approach to achieve the objectives.
Rationale: Explanation of the reasoning behind the strategy, including why it is expected to work.
Goals: Specific goals the strategy aims to achieve, aligned with the overall objectives.

Implementation Plan
Action Items: Detailed list of actions needed to implement the strategy, including:
  - Description: What needs to be done.
  - Responsibility: Who will do it.
  - Resources Required: Necessary resources (time, money, personnel).
  - Timeline: When it will be done.
Resource Allocation: Overview of the budget and resources allocated to each action item.
Risk Management Plan: Identification of potential risks and contingency plans.

Action Plan Execution
Timeline and Milestones: Detailed timeline with milestones to track progress.
Roles and Responsibilities: Detailed breakdown of roles and responsibilities in executing the plan.

Monitoring and Evaluation
Performance Metrics: Key performance indicators (KPIs) to measure success.
Evaluation Plan: How and when the implementation will be reviewed and evaluated.
Adjustment Mechanism: Process for making adjustments based on monitoring and evaluation results.

Conclusion
Summary of the strategy, expected outcomes, and the significance of the implementation plan.

Appendices
Any additional information, such as detailed data, technical descriptions, or related documents.

References
List of all sources cited in the preparation of the document.
"""
    # Define the prompt structure
    prompt = PromptTemplate(
        input_variables=["query", "result"],
        template=prompt_template,
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Pass the context and question to the Langchain chain
    result_chain = llm_chain.invoke({"query": query, "result": result})

    return result_chain
