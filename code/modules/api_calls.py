import re
from modules.evaluation import extract_number


def extract_bracket_content(response: str) -> str:
    # Function to extract content within brackets
    match = re.search(r'\[(.*?)\]', response)
    return match.group(1) if match else response


# Function to handle a single query
def single_query(
                doc,
                query,
                client,
                model="selected_model",
                verbose=False
                ):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """
                        You are a helpful assistant tasked with answering user queries about a page with financial data.
                        Reflect on the data provided and produce an answer.
                        Report the exact answer at the end in brackets '[]'.

                        Input:
                        - data_page: A page containing financial data.
                        - user_query: A question from the user about the financial data.

                        Output:
                        - Return an answer based on the data provided, with the exact answer at the end in brackets '[]'.

                        ###
                        Examples:
                        Example 1:
                        - data_page: "The company's revenue increased from $10 million in 2010 to $50 million in 2014."
                        - user_query: "What was the revenue increase from 2010 to 2014?"
                        - Output: "The company's revenue increased by $40 million from 2010 to 2014. [$40 million]"

                        Example 2:
                        - data_page: "The net profit margin for the year was 15%."
                        - user_query: "What was the net profit margin for the year?"
                        - Output: "The net profit margin for the year was 15%. [15%]"

                        Example 3:
                        - data_page: "The total assets of the company are valued at $200 million."
                        - user_query: "What is the value of the company's total assets?"
                        - Output: "The value of the company's total assets is $200 million. [$200 million]"
                        """ # noqa E501
            },
            {
                "role": "user",
                "content": f"""
                    USER_QUERY: {query['question']}
                    """,
            },
            {
                "role": "assistant",
                "content": f"""
                    CONTEXT_DOCUMENT: {doc}
                    """,
            }
        ],
        model=model,
    ).choices[0].message.content
    if verbose:
        return response, extract_bracket_content(response)
    return extract_bracket_content(response)


def classifier(
                doc,
                query,
                client,
                model="gpt-3.5-turbo",
                verbose=False
                ):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """
                        You are an intelligent and helpful classifier designed to sort context documents for a large language model (LLM).
                        Your task is to determine the relevance of a given document in relation to a user query.

                        Input:
                        - DOCUMENT: A piece of text that needs to be evaluated.
                        - USER_QUERY: A question or statement from the user that specifies the information they are seeking.

                        Output:
                        - Return 1 if the content of the document is relevant to the user_query.
                        - Return 0 if the content of the document is not relevant to the user_query.

                        Criteria for Relevance:
                        - The document directly addresses or answers the user_query.
                        - The document contains information that is contextually related to the user_query.
                        - The document provides useful insights or data that align with the intent of the user_query.
                        """ # noqa E501
            },
            {
                "role": "user",
                "content": f"""
                    USER_QUERY: {query}
                    DOCUMENT: {doc}
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


def context_augment(
                query,
                client,
                model="gpt-3.5-turbo",
                ):
    context_augmented = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """
                            You are a helpful assistant that, given a question, will produce a set of related terms to provide better context for the retrieval tool.
                            Make sure to give just points, being concise and only listing concepts.

                            Input:
                            - USER_QUERY: A question from the user that specifies the information they are seeking.

                            Output:
                            - Return a list of related terms that provide better context for the retrieval tool.

                            Examples:
                            Example 1:
                            - question: "What are the benefits of investing in stocks?"
                            - Output: ["capital gains", "dividends", "portfolio diversification", "market growth", "equity"]

                            Example 2:
                            - question: "How does a mortgage work?"
                            - Output: ["loan", "interest rate", "principal", "amortization", "down payment"]

                            Example 3:
                            - question: "What are the types of financial statements?"
                            - Output: ["balance sheet", "income statement", "cash flow statement", "statement"]
                            """ # noqa E501
                },
                {
                    "role": "user",
                    "content": f"""
                        USER_QUERY: {query}
                        """,
                }
            ],
            model=model,
        ).choices[0].message.content
    return context_augmented
