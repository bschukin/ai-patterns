---
title: "Few-Shot Prompt Template"
---

# Few-Shot Prompt Template

<br>
{{% hint info %}}

Giving the model a few (usually one to five) examples of a task that it needs to perform within the prompt itself is known as few-shot prompting. These examples provide the model with the context and clarify how to approach related activities. Even if the model is not specifically trained for that task, it can nonetheless learn the required structure and output type by observing these examples.

{{% /hint %}}   

### LangChain implementation

```python

from langchain_core.prompts import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate

# Define a template for each example
example_template = PromptTemplate(
    input_variables=["question", "answer"],
    template="Q: {question}\nA: {answer}\n"
)

# Define some examples for the few-shot prompt
examples = [
    {"question": "Who was the first President of the United States?", "answer": "George Washington."},
    {"question": "What year did World War II begin?", "answer": "1939."}
]

# Define the prompt with a few examples and placeholders for dynamic input
few_shot_template = FewShotPromptTemplate(
    examples=examples,  # Few-shot examples
    example_prompt=example_template,  # Format for the examples
    prefix="Here are some historical facts:\n\n",  # Prefix text before examples
    suffix="Q: {user_question}\nA:",  # Suffix text where the model generates the answer
    input_variables=["user_question"]  # The dynamic part (user's question)
)

template = few_shot_template.invoke({"user_question": "Who wrote the Declaration of Independence?"})
print(template)

```