# Query_Answer

import os
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from modules.evaluation import (
    self_evaluation,
    sequence_matcher,
    jaccard_similarity,
    compare_numbers
)

from modules.api_calls import single_query

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
        value=50,
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

    # Take a random sample of 100 entries
    df = df.sample(n=sample_size, random_state=42)

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

        # Execute the single_query function for each row
        @st.cache_data
        def cache_computation(df, model_name):
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

        df = cache_computation(df, model_name)

        # Visualization
        sorted_df = df.sort_values(by=f'{model_name}_numeric_perc')
        transposed_corpus = sorted_df.iloc[:, -4:].T

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

        st.session_state.index = st.slider(
            "Select an index",
            min_value=0,
            max_value=len(df)-1,
            value=st.session_state.index,
            step=1
            )

        # Display selected query, label, model answer, and reflection
        i = st.session_state.index
        st.write(f"Query: {df.iloc[i]['qa']['question']}")
        st.write(f"Label: {df.iloc[i]['qa']['answer']}")
        st.write(f"""{model_name} answer:
                            {df.iloc[i][f'{model_name}_answer'][1]}""")
        st.write(f"""{model_name} reflection:
                 {df.iloc[i][f'{model_name}_answer'][0]}""")

        # Describe for statistics
        st.markdown('### Metrics mean and statistics')
        st.write(df.describe())

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
                                f"{model_name}_summary_report.txt")

            with open(report_file, "w", encoding="utf-8") as file:
                file.write("Summary Report\n")
                file.write("="*20 + "\n\n")
                file.write(f"Model: {model_name}\n")
                file.write(f"Total Entries: {len(df)}\n\n")

                # Add df.describe() output
                file.write("DataFrame Description:\n")
                file.write("-"*20 + "\n")
                file.write(df.describe().to_string())
                file.write("\n\n")

                file.write("Evaluation Metrics (Averages):\n")
                file.write("-"*20 + "\n")
                file.write(f"""Average Self Evaluation:
                            {df[f'{model_name}_self_ev'].mean()}\n\n""")
                file.write(f"""Average Numeric Percentage
                            {df[f'{model_name}_numeric_perc'].mean()}\n""")
                file.write(f"""Average Jaccard Similarity:
                            {df[f'{model_name}_jaccard_sim'].mean()}\n""")
                file.write(f"""Average Sequence Matcher:
                            {df[f'{model_name}_sequence'].mean()}\n\n""")

                filtered_5_df = df[df[f'{model_name}_numeric_perc'] > 0.95]
                filtered_3_df = df[df[f'{model_name}_numeric_perc'] > 0.97]
                filtered_1_df = df[df[f'{model_name}_numeric_perc'] > 0.99]

                # Add filtered DataFrame counts
                file.write("Filtered DataFrame Counts:\n")
                file.write("-"*20 + "\n")
                file.write(f"""Correct answers with a 5% confidence range:
                            {len(filtered_5_df)}\n""")
                file.write(f"""Correct answers with a 3% confidence range:
                            {len(filtered_3_df)}\n""")
                file.write(f"""Correct answers with a 1% confidence range:
                            {len(filtered_1_df)}\n""")

                file.write("\nSample Entry:\n")
                file.write("-"*20 + "\n")
                file.write(f"Query: {df.iloc[i]['qa']['question']}\n")
                file.write(f"Label: {df.iloc[i]['qa']['answer']}\n")
                file.write(f"""{model_name} answer: 
                            {df.iloc[i][f'{model_name}_answer'][1]}\n""")
                file.write(f"""{model_name} reflection:
                            {df.iloc[i][f'{model_name}_answer'][0]}\n\n""")

            st.write(f"Report saved to {report_file}")


if __name__ == "__main__":
    main()
