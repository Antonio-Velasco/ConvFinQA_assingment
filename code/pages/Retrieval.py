# Retrieval

import os
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents.base import Document

from modules.evaluation import (
    self_evaluation,
    sequence_matcher,
    jaccard_similarity,
    compare_numbers
)

from modules.api_calls import (
    single_query,
    context_augment,
    classifier
)

# Load environment variables
load_dotenv()

# Define the model list
models = {
    "Online Models": ["gpt-3.5-turbo", "gpt-4o"],
    "Local Models": ["llama-3.2-1b-instruct"]
}


def get_model_type(model_name: str) -> str:
    # Function to get model type
    for key, value in models.items():
        if model_name in value:
            return key
    return None


# Streamlit app
def main():
    st.title("Model Selection App")

    # Function to reset the "Run" button state
    def reset_run_state():
        st.session_state.clicked = False

    # Model selection
    model_name = st.selectbox(
                "Select a model",
                [model for sublist in models.values() for model in sublist],
                on_change=reset_run_state
                )
    model_type = get_model_type(model_name)

    # Display selected model type
    st.write(f"Selected Model: {model_name}")
    st.write(f"Model Type: {model_type}")

    # Handle local model URL input
    local_url = ""
    if model_type == "Local Models":
        local_url = st.text_input("Enter the local URL for the model")
        st.write(f"Local URL: {local_url}")

    # Sample size selection
    sample_size = st.number_input(
        "Enter sample size",
        min_value=10,
        max_value=100,
        value=10,
        step=10,
        on_change=reset_run_state
        )

    if 'index' not in st.session_state:
        st.session_state.index = 0

    if 'clicked' not in st.session_state:
        st.session_state.clicked = False

    def click_button():
        st.session_state.clicked = True

    # Load and preprocess data
    data_path = "data/train_extended.json"
    df = pd.read_json(data_path)
    df['qa'] = df['qa'].fillna(df['qa_0'])
    df = df.drop(['qa_0', 'qa_1'], axis=1)
    df = df.dropna(subset=['qa'])

    # Create Chroma index
    pages = []
    for i, entry in enumerate(df['join_text']):
        doc_with_metadata = Document(
            page_content=(entry),
            metadata={"page": i})
        pages.append(doc_with_metadata)

    st.button('Run', on_click=click_button)
    if st.session_state.clicked:
        client = OpenAI(
            api_key=os.getenv('SECRET_OPENAI_API_KEY')
        )
        if model_type == "Local Models":
            client = OpenAI(
                api_key="lm-studio",
                base_url=local_url
            )

        st.write("Client initialized successfully!")

        # create the open-source embedding function
        embedding_function = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2")
        # load it into Chroma
        db = Chroma.from_documents(pages, embedding_function)

        # Execute the single_query function for each row
        @st.cache_data
        def cache_computation(df, model_name, sample_size, _client, _db):
            df = df.sample(n=sample_size, random_state=42)
            doc_list = []
            for i, entry in enumerate(df['qa']):
                augmented_query = context_augment(entry['question'], _client)
                retrieved = _db.similarity_search(
                        entry['question'] + augmented_query, k=10)

                duplicates = []
                retrieved_unique = []
                for doc in retrieved:
                    if doc.metadata['page'] not in duplicates:
                        duplicates.append(doc.metadata['page'])
                        retrieved_unique.append(doc)
                doc_list.append(retrieved_unique)
            df['context'] = doc_list

            df['retrieved'] = df.apply(
                    lambda row: [i.metadata['page'] for i in row['context']],
                    axis=1)
            df['correct_retrieve'] = df.apply(
                    lambda row: row.name in row['retrieved'], axis=1)
            
            df[f'{model_name}_answer'] = df.apply(
                lambda row: single_query(
                    row['join_text'],
                    row['qa'],
                    client,
                    model=model_name,
                    verbose=True
                ),
                axis=1
            )

            # Calculate evaluation metrics
            df[f"{model_name}_numeric_perc"] = df.apply(
                lambda row: compare_numbers(
                    row[f'{model_name}_answer'][1],
                    row['qa']['answer']),
                axis=1)
            df[f"{model_name}_jaccard_sim"] = df.apply(
                lambda row: jaccard_similarity(
                    row[f'{model_name}_answer'][1],
                    row['qa']['answer']),
                axis=1)
            df[f"{model_name}_sequence"] = df.apply(
                lambda row: sequence_matcher(
                    row[f'{model_name}_answer'][1],
                    row['qa']['answer']),
                axis=1)
            df[f"{model_name}_self_ev"] = df.apply(
                lambda row: self_evaluation(
                    row['qa']['question'],
                    row[f'{model_name}_answer'][1],
                    row['qa']['answer'], client), axis=1)

            return df

        df_s = cache_computation(df, model_name, sample_size, client, db)

        # Visualization
        sorted_df = df_s.sort_values(by=f'{model_name}_numeric_perc')
        transposed_corpus = sorted_df.iloc[:, -4:].T

        # Calculate the percentage of True values
        st.markdown("### Augmented Retriever Recall")
        percentage_true = df_s['correct_retrieve'].mean() * 100
        st.write(f"Recall: {percentage_true:.2f}%")

        st.markdown(" ### Evaluation Metrics Heatmap:")
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(
            transposed_corpus,
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            annot=True,
            ax=ax,
            cbar_kws={'label': 'Similarity Score'}
            )
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Metrics')
        plt.xticks(rotation=90)
        st.pyplot(fig)

        # Describe for statistics
        st.markdown('### Metrics mean and statistics')
        st.write(df_s.describe())

        st.markdown('### Numeric percentage Barplot')
        # Create a bar plot
        fig, ax = plt.subplots(figsize=(10, 6))

        sorted_df[f'{model_name}_numeric_perc'].plot(kind='bar', ax=ax)

        # Add a horizontal line at y=0.95
        ax.axhline(y=0.95, color='r', linestyle='--', label='Threshold 0.95')

        # Add labels and title
        ax.set_xlabel('Index')
        ax.set_ylabel('Numeric percentage')
        ax.set_title('Bar Plot of Results with Threshold Line at 0.95')
        ax.legend()

        # Display the plot in Streamlit
        st.pyplot(fig)

        # Produce report button
        if st.button("Produce Report"):
            report_path = "data/report"
            os.makedirs(report_path, exist_ok=True)
            report_file = os.path.join(
                                report_path,
                                f"{model_name}_RAG_summary_report.txt")

            with open(report_file, "w", encoding="utf-8") as file:
                file.write("Summary Report\n")
                file.write("="*20 + "\n\n")
                file.write(f"Model: {model_name}\n")
                file.write(f"Total Entries: {len(df_s)}\n\n")

                file.write("Augmente Retrieval Revall:\n")
                file.write("-"*20 + "\n")
                file.write(str(percentage_true))
                file.write("\n\n")

                # Add df_s.describe() output
                file.write("DataFrame Description:\n")
                file.write("-"*20 + "\n")
                file.write(df_s.describe().to_string())
                file.write("\n\n")

                file.write("Evaluation Metrics (Averages):\n")
                file.write("-"*20 + "\n")
                file.write(f"""Average Self Evaluation:
                            {df_s[f'{model_name}_self_ev'].mean()}\n\n""")
                file.write(f"""Average Numeric Percentage
                            {df_s[f'{model_name}_numeric_perc'].mean()}\n""")
                file.write(f"""Average Jaccard Similarity:
                            {df_s[f'{model_name}_jaccard_sim'].mean()}\n""")
                file.write(f"""Average Sequence Matcher:
                            {df_s[f'{model_name}_sequence'].mean()}\n\n""")

                filtered_5_df = df_s[df_s[f'{model_name}_numeric_perc'] > 0.95]
                filtered_3_df = df_s[df_s[f'{model_name}_numeric_perc'] > 0.97]
                filtered_1_df = df_s[df_s[f'{model_name}_numeric_perc'] > 0.99]

                # Add filtered DataFrame counts
                file.write("Filtered DataFrame Counts:\n")
                file.write("-"*20 + "\n")
                file.write(f"""Correct answers with a 5% confidence range:
                            {len(filtered_5_df)}\n""")
                file.write(f"""Correct answers with a 3% confidence range:
                            {len(filtered_3_df)}\n""")
                file.write(f"""Correct answers with a 1% confidence range:
                            {len(filtered_1_df)}\n""")

            st.write(f"Report saved to {report_file}")


if __name__ == "__main__":
    main()
