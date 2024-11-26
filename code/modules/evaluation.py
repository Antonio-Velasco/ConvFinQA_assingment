from difflib import SequenceMatcher
import re


# Function to calculate sequence matcher ratio
def sequence_matcher(a, b):
    if a is not None and b is not None:
        return SequenceMatcher(None, a, b).ratio()
    else:
        return None


# Function to calculate Jaccard similarity
def jaccard_similarity(a, b):
    if a is not None and b is not None:
        intersection = len(set(a).intersection(set(b)))
        union = len(set(a).union(set(b)))
        return intersection / union
    else:
        return None


# Function to extract numbers from text
def extract_number(text):
    text = text.replace(',', '')
    numbers = re.findall(r'\d+\.\d+|\d+', text)
    numbers = [float(num) for num in numbers]
    return numbers[0] if numbers else None


# Function to compare numbers
def compare_numbers(a, b):
    label_num = None
    answer_num = None
    if a is not None and b is not None:
        label_num = extract_number(b)
        answer_num = extract_number(a)

    if label_num is not None and answer_num is not None:
        epsilon = 1e-10
        difference = abs(label_num - answer_num)
        percentage_error_normalized = (
            difference / (label_num + answer_num + epsilon))
        return 1 - percentage_error_normalized
    else:
        return None


# Function to self-evaluate
def self_evaluation(
                query,
                a,
                b,
                client,
                model="gpt-3.5-turbo",
                verbose=False
                ):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You are a helpful agent that will evaluate financial answers from a LLM.
                Given the USER_QUERY, compare CORRECT_ANSWER against LLM_ANSWER.
                Provide only a 1 if LLM_ANSWER is correct.
                Provide a number between 0 and 1 if LLM_ANSWER is partially correct.
                Provide a 0 if LLM_Answer is wrong.
                """ # noqa E501
            },
            {
                "role": "user",
                "content": f"""
                    USER_QUERY: {query}
                    CORRECT_ANSWER: {a}
                    LLM_ANSWER: {b}
                    """,
            },
            {
                "role": "assistant",
                "content": """
                    Write exclusively a number as your output. Eg: 1
                    """,
            }
        ],
        model=model,
    ).choices[0].message.content
    if verbose:
        return response, extract_number(response)
    return extract_number(response)
