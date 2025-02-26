"""Define a chatbot agent that uses an LLM to classify mortgage products and financing types."""

from typing import Any, Dict, List, Literal
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from agent.configuration import Configuration
from agent.state import State
from agent.constants import ProductType, FinancingType

class MortgageClassification(BaseModel):
    """Classification of mortgage product and financing type."""
    productCode: ProductType = Field(
        description="The type of mortgage product the user is interested in"
    )
    financingType: FinancingType = Field(
        description="The type of financing the user is seeking"
    )
    reasoning: str = Field(
        description="Explanation of why this classification was chosen"
    )


async def classify_with_llm(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Use an LLM to classify mortgage product and financing type based on user input."""
    configuration = Configuration.from_runnable_config(config)
    
    # System prompt to guide the LLM
    system_prompt = """You are a mortgage loan officer. Your job is to chat with the user to determine what kind of mortgage they're looking for:
    
    1. Product Code: What type of mortgage product they're looking for
        - Value Add: The user is looking for a short term loan which they may also call a "fix and flip" loan or a "bridge" loan.
        - Rental: The user is looking for a long term loan which takes into account the DSCR (Debt Service Coverage Ratio) of the property.
    
    2. Financing Type: What kind of financing they need
       - purchase: Buying a new home
       - refinance rate term: Refinancing an existing mortgage
       - refinance cashout: Refinancing an existing mortgage to cash out equity
    
    Analyze the user's message carefully and provide your classification in JSON format.
    """
    
    # Create messages for the LLM
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state.input if hasattr(state, "input") else "")
    ]
    
    # Initialize the LLM
    model = ChatOpenAI(temperature=0, model="gpt-4o")
    
    # Create a parser for structured output
    parser = JsonOutputParser(pydantic_object=MortgageClassification)
    
    # Create the chain
    chain = model | parser
    
    # Run the chain
    result = await chain.ainvoke(messages)
    
    return {
        "productCode": result["productCode"],
        "financingType": result["financingType"],
        "reasoning": result["reasoning"]
    }


# Define a new graph
workflow = StateGraph(State, config_schema=Configuration)

# Add the node to the graph
workflow.add_node("classify_with_llm", classify_with_llm)

# Set the entrypoint
workflow.add_edge("__start__", "classify_with_llm")

# Compile the workflow into an executable graph
graph = workflow.compile()
graph.name = "LLM Mortgage Classifier"
