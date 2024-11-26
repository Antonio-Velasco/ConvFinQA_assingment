# Technical Assignment ConvFinQA

## Overview
ConvFinQA is a task focused on answering questions derived from financial documents. The challenge is to accurately extract relevant information from a large collection of documents and generate correct answers. This document details the steps to implement and evaluate a model for this task, starting with known context and adding a retrieval layer.

## Resources
- Repository: https://github.com/Antonio-Velasco/ConvFinQA_assingment

## Content

### Code
- **home**: Landing page for a Streamlit app.

#### Modules
- **api_calls**: Contains the different calls to LLMs and the prompts.
- **evaluation**: Contains the functions used for metric evaluation.

#### Notebooks
- **EDA**: Exploration and insights on the dataset.
- **query**: Development of the straightforward query approach.
- **retrieval**: Development and experimentation on the augmented retrieval.

#### Pages
- **query_answer**: Query approach with UI. The user may select different models, sample size, then explore the results and output, and produce a report.
- **Retrieval**: Comprehensive build of this exercise. Allows the user to select model and sample size (corpus is the whole dataset) and produces visualizations and a report.

### Data
- **images**: Figures for the README.
- **report**: Different result reports from various models and scenarios.
- **train_extended**: Processed dataset.
- **train**: Original dataset.

## How to use

### Installation
- Git clone in your local machine.
- Create a '.env' file with: SECRET_OPENAI_API_KEY=[YOUR_KEY_HERE]
    - Alternatively, 'LM Studio' is compatible with OpenAI's python library, you can run local models with it.
- Create a virtual environment and install
the libraries in 'requirements.txt'
- In the terminal run the command 
    ``` 
    streamlit run code/Home.py
    ```
- Navigate on the interface, should be self explanatory.

## Data Preparation
- **Load Data**: Import the dataset.
- **Clean Data**: Handle missing values and preprocess text.
    - Joinin 'pre_text', 'table' and 'post_text' for simplicity.
    - Will take one question per document. Further questions beyond the first will be ignored.
    - Tables are also written in markdown format for easier handling by the LLMs

## DOCUMENT QUERY APPROACH

As a first step, let’s build a quick solution that can answer the questions about the relevant context. On a random sample of 100 entries from the whole dataset.

**Objective**: Develop and evaluate a question-answering model using the ConvFinQA dataset.

### Approach

A simple self-reflecting model with a few shot examples and post-processing.

### Evaluation Metrics

#### **LLM Model Self-Evaluation**
LLM model self-evaluation involves the model assessing its own performance. This can be done by comparing the model's predictions with the ground truth. It leverages the LLM flexibility of understanding non-exact yet true answers.

#### **Numeric Percentage Closeness**

Numeric percentage closeness measures how similar two numbers are. It is calculated using the following steps:

1. **Calculate the Difference**: Compute the absolute difference between the label and the answer.
   ```python
   difference = abs(label_num - answer_num)
   ```

2. **Calculate the Percentage Error**: Normalize the difference by dividing it by the sum of the label and answer, plus a small epsilon to avoid division by zero.
   ```python
   percentage_error_normalized = (difference / (label_num + answer_num + epsilon))
   ```

This metric helps in evaluating the accuracy of numerical predictions by showing how close the predicted value is to the actual value.

#### **SequenceMatcher** 

SequenceMatcher is a class in Python's difflib module that compares pairs of sequences of any type, as long as the elements are hashable. It finds the longest contiguous matching subsequence that contains no "junk" elements (e.g., blank lines or whitespace). This process is applied recursively to the left and right pieces of the matching subsequence.

#### **Jaccard Similarity Coefficient**

The Jaccard similarity coefficient (Jaccard index) measures the similarity and diversity of sample sets. It is defined as the size of the intersection divided by the size of the union of the sample sets:

$$ \text{Jaccard Similarity} = \frac{|A \cap B|}{|A \cup B|} $$

The Jaccard similarity ranges from 0 to 1. A value closer to 1 indicates higher similarity, while 0 means no common elements.

Caveats: Last two metrics are relevant when comparing text output, in this case most of the answers are numeric. Might be useful anyway.

# Result

## Summary Report

**Model**: gpt-4o


**Total Entries**: 100        

### DataFrame Description:
--------------------
| Statistic       | input_tokens | gpt-4o_numeric_perc | gpt-4o_jaccard_sim | gpt-4o_sequence |
|-----------------|--------------|---------------------|--------------------|-----------------|
| **count**       | 100.000000   | 95.000000           | 100.000000         | 100.000000      |
| **mean**        | 872.350000   | 0.723166            | 0.392243           | 0.441637        |
| **std**         | 288.228878   | 0.379744            | 0.326300           | 0.350856        |
| **min**         | 206.000000   | 9.999779e-13        | 0.000000           | 0.000000        |
| **25%**         | 706.500000   | 0.462010            | 0.070813           | 0.036787        |
| **50%**         | 846.000000   | 0.994865            | 0.333333           | 0.436508        |
| **75%**         | 1040.500000  | 0.999757            | 0.666667           | 0.732955        |
| **max**         | 1771.000000  | 1.000000            | 1.000000           | 1.000000        |

### Evaluation Metrics (Averages):
--------------------
- **Average Numeric Percentage**: 0.7231665185195969
- **Average Jaccard Similarity**: 0.39224333777072523
- **Average Sequence Matcher**: 0.44163700036095643

### Filtered DataFrame Counts:
--------------------
- **Correct answers with a 5% confidence range**: 55
- **Correct answers with a 3% confidence range**: 55
- **Correct answers with a 1% confidence range**: 52

![](/data/images/image-4.png)


Most queries and answers involve specific numeric values or percentages, making Numeric Percentage Closeness and self-evaluation the most appropriate metrics. Other matching metrics are less effective in this context.

While the context is accurate, answers often require reflection and calculations, leading to slight variations (e.g., decimals or approximations) and format differences. There is also a risk of incorrect evaluation due to the verbosity of the models.

Nevertheless, 55 out of 100 answers fall within a 3% range of numeric closeness.

Self-evaluation skipped with this model to save resources.

## Summary Report

**Model**: gpt-3.5-turbo

**Total Entries**: 100      

### DataFrame Description:
--------------------
| Statistic       | input_tokens | gpt-3.5-turbo_numeric_perc | gpt-3.5-turbo_jaccard_sim | gpt-3.5-turbo_sequence | gpt-3.5-turbo_self_ev |
|-----------------|--------------|----------------------------|---------------------------|------------------------|-----------------------|
| **count**       | 100.000000   | 95.000000                  | 100.000000                | 100.000000             | 100.000000            |
| **mean**        | 872.350000   | 0.779399                   | 0.469426                  | 0.518352               | 0.624000              |
| **std**         | 288.228878   | 0.350055                   | 0.332635                  | 0.349388               | 0.426844              |
| **min**         | 206.000000   | 5.714286e-08               | 0.000000                  | 0.000000               | 0.000000              |
| **25%**         | 706.500000   | 0.696552                   | 0.164474                  | 0.216667               | 0.000000              |
| **50%**         | 846.000000   | 0.996564                   | 0.500000                  | 0.593407               | 0.800000              |
| **75%**         | 1040.500000  | 0.999646                   | 0.714286                  | 0.800000               | 1.000000              |
| **max**         | 1771.000000  | 1.000000                   | 1.000000                  | 1.000000               | 1.000000              |

### Evaluation Metrics (Averages):
--------------------
- **Average Self Evaluation**: 0.624
- **Average Numeric Percentage**: 0.7793988762024824
- **Average Jaccard Similarity**: 0.46942560507059883
- **Average Sequence Matcher**: 0.5183515851275569

### Filtered DataFrame Counts:
--------------------
- **Correct answers with a 5% confidence range**: 61
- **Correct answers with a 3% confidence range**: 60
- **Correct answers with a 1% confidence range**: 53

![](/data/images/image-3.png)

Performing slightly better, gpt-3.5-turbo with an Average Numeric percentage of 0.78 and 60 entries out of 100 on a 3% range of confidence.
Self evaluation at 0.624.

## Summary Report

**Model**: llama-3.2-1b-instruct

**Total Entries**: 100

### DataFrame Description:
--------------------
| Statistic       | input_tokens | llama-3.2-1b-instruct_numeric_perc | llama-3.2-1b-instruct_jaccard_sim | llama-3.2-1b-instruct_sequence | llama-3.2-1b-instruct_self_ev |
|-----------------|--------------|------------------------------------|-----------------------------------|--------------------------------|-------------------------------|
| **count**       | 100.000000   | 86.000000                          | 100.000000                        | 100.000000                      | 96.000000                      |
| **mean**        | 872.350000   | 0.288455                           | 0.090649                          | 0.082120                        | 521298.763034375               |
| **std**         | 288.228878   | 0.297906                           | 0.134936                          | 0.156007                        | 5103056.000000000              |
| **min**         | 206.000000   | 8.130163e-13                       | 0.000000                          | 0.000000                        | 0.000000                       |
| **25%**         | 706.500000   | 0.020350                           | 0.000000                          | 0.000000                        | 0.000000                       |
| **50%**         | 846.000000   | 0.197846                           | 0.068966                          | 0.006822                        | 0.000000                       |
| **75%**         | 1040.500000  | 0.470239                           | 0.106961                          | 0.117647                        | 2.772500                       |
| **max**         | 1771.000000  | 1.000000                           | 1.000000                          | 1.000000                        | 50000000.000000000             |

### Evaluation Metrics (Averages):
--------------------
- **Average Self Evaluation**: 521298.763034375
- **Average Numeric Percentage**: 0.28845470284270996
- **Average Jaccard Similarity**: 0.09064913537988988
- **Average Sequence Matcher**: 0.08212000343833789

### Filtered DataFrame Counts:
--------------------
- **Correct answers with a 5% confidence range**: 3
- **Correct answers with a 3% confidence range**: 3
- **Correct answers with a 1% confidence range**: 0

![](/data/images/image-2.png)


Llama-3b runnin in local falls deeply behind in quality, struggling to understand the prompt.
Hiraliously givin himself a stratospheric score on self evaluation while failing to understand that values should stay between 0 and 1

## Augmented Retrieval 

Now that we have a working solution to extract given the correct context, let's try to bring up a tool that brings said context to our Query Model.

**Objective**: Develop and evaluate a Retrieval tool

### Approach

Build an embeddings index in chroma with the corpus. 
Augment the queries so that they matching against the index is of better quality.

### Overal final Architechture

![](/data/images/image-5.png)

### Metric

- **Recall**: Measure the proportion of relevant information successfully retrieved by the model. Recall is calculated as the number of relevant documents retrieved divided by the total number of relevant documents in the dataset. This metric helps assess the effectiveness of the retrieval technique in finding all pertinent information needed to answer the questions accurately.

After producing the augmented query we retrieve the 10 nearest pages. To avoid token limitations and noise, a classifier is built in place, this is the model classifying as either 1 or 0 this documents. Those deemed relevant will be added as context. The final model is feed all of them for extraction.

We can check how many of these final documents are the desired context.

From a sample of 500 entries it got a recall of 0.46 while developing.

**Caveat**: Increasing the sample to the over 3000 entries will make the task harder and thus decrease the recall drastically.

Let's try the whole thing now.

## Results

## Summary Report

**Model**: gpt-3.5-turbo

**Total Entries**: 100 and 3037 indexed

### Augmented Retrieval Recall:
--------------------
16.0

### DataFrame Description:
--------------------
| Statistic       | input_tokens | gpt-3.5-turbo_numeric_perc | gpt-3.5-turbo_jaccard_sim | gpt-3.5-turbo_sequence | gpt-3.5-turbo_self_ev |
|-----------------|--------------|----------------------------|---------------------------|------------------------|-----------------------|
| **count**       | 100.000000   | 96.000000                  | 100.000000                | 100.000000             | 100.000000            |
| **mean**        | 872.350000   | 0.639278                   | 0.320274                  | 0.357746               | 0.483400              |
| **std**         | 288.228878   | 0.400283                   | 0.291966                  | 0.321968               | 0.452379              |
| **min**         | 206.000000   | 5.714286e-08               | 0.000000                  | 0.000000               | 0.000000              |
| **25%**         | 706.500000   | 0.196243                   | 0.076442                  | 0.048036               | 0.000000              |
| **50%**         | 846.000000   | 0.817104                   | 0.222222                  | 0.296703               | 0.500000              |
| **75%**         | 1040.500000  | 0.999122                   | 0.517857                  | 0.666667               | 1.000000              |
| **max**         | 1771.000000  | 1.000000                   | 1.000000                  | 1.000000               | 1.000000              |

### Evaluation Metrics (Averages):
--------------------
- **Average Self Evaluation**: 0.48340000000000005
- **Average Numeric Percentage**: 0.6392783865136628
- **Average Jaccard Similarity**: 0.3202741526084767
- **Average Sequence Matcher**: 0.3577462461422044

### Filtered DataFrame Counts:
--------------------
- **Correct answers with a 5% confidence range**: 41
- **Correct answers with a 3% confidence range**: 41
- **Correct answers with a 1% confidence range**: 38

![](/data/images/image-6.png)


Curiously, while the retrieval recall is only 16%, the model achieves an average numeric percentage of 63% and 41 out of 100 answers fall within a 3% confidence range. This performance is almost as good as when the correct context is provided.

This suggests that relevant data might be distributed across different documents, indicating that the retrieval mechanism is functioning as intended. Therefore, focusing solely on increasing recall might not be beneficial and could unnecessarily compromise the overall quality of the results.

Instead, it might be more effective to balance recall with precision to ensure that the retrieved documents are both relevant and high-quality, thereby maintaining the integrity of the model's performance.

# Future Development

#### Prompt Refinement
- **Further develop the prompt** and refine it while testing for quality.

#### Evaluation and Reporting
- **Set the calls to verbose** and compare the reflection with self-evaluation. Many correct answers might not be reported due to format discrepancies.

#### Hyperparameter Optimization
- **Optimize hyperparameters** to improve model performance.

#### Model Testing
- **Test more models** for different tasks. A small local model might be effective as a classifier.

#### Code Improvement
- **Refactor code** for better readability and maintainability.
- **Enhance documentation** to ensure clarity and comprehensiveness.
- **Implement logging** for better tracking and debugging.
- **Develop unit and integration tests** to ensure code reliability.

#### Model Fine-Tuning
- **Fine-tune models** with the available data to improve accuracy and relevance.

#### Custom Embedding Model
- **Train a simple custom embedding model** specialized in financial terms to enhance understanding and performance in financial contexts.