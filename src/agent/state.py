"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass
from .constants import ProductType, FinancingType

@dataclass
class State:
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    for more information.
    """

    propertyAddress: str = "123 Main St, Anytown, USA"
    propertyType: str = "Single Family Home"
    purchasePrice: float = 500000
    asIsValue: float = 400000
    productCode: ProductType = ProductType.Value_Add
    financingType: FinancingType = FinancingType.Purchase
    pricingEngineParams: dict = {}