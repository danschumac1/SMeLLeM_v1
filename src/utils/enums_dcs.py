from dataclasses import dataclass
from typing import Dict #, List
from pydantic import BaseModel

class QAs(BaseModel):
    question: Dict[str, str]  # Multiple inputs as a dictionary
    answer: str | BaseModel  # Allow both strings and BaseModel

class AnswerBM(BaseModel):
    """
    Base model for the output of the LLM.
    """
    answer: str

@dataclass
class TSData:
    description: str
    description_short: str
    description_tiny: str
    characteristics: str
    generator: str
    metadata: str
    series: str
    uuid: str
    question: str
    options: Dict[str, str]
    answer_index: int

    @property
    def answer(self) -> str:
        return self.options[self.answer_index]
    
    @property
    def options_text(self) -> str:
        abcd = ['A', 'B', 'C', 'D']
        options = []
        for i, option in enumerate(self.options):
            options.append(f"{abcd[i]}: {option}")
        return "\n".join(options)