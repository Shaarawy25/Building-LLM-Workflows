# Buidling LLM Workflows

## Overview
This repository contains the implementation of various LLM workflows as part of **Buidling LLM Workflows**. The goal of this project is to design modular systems that break down complex tasks into smaller components and assemble them into effective workflows. The workflows include:

1. **Basic LLM Workflow**: A pipeline and DAG-based approach to repurpose blog posts into summaries, social media posts, and email newsletters.
2. **Self-Correction with Reflexion**: Enhances the basic workflow with iterative quality evaluation and improvement using Reflexion.
3. **Advanced Agent-Driven Workflow**: Uses an LLM agent to dynamically decide task execution and complete the workflow.

---

## Table of Contents
1. [Setup Instructions](#setup-instructions)
2. [Implementation Documentation](#implementation-documentation)
3. [Example Outputs](#example-outputs)
4. [Analysis of Workflow Approaches](#analysis-of-workflow-approaches)
5. [Challenges Encountered](#challenges-encountered)

---

## Setup Instructions

### Prerequisites
- Python 3.10+
- Access to LLM APIs (OpenAI, Groq, etc.)
- Familiarity with tool usage in LLMs

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/[your-repo-url](https://github.com/Shaarawy25/Building-LLM-Workflows).git
   cd your-repo-directory
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your `.env` file with your API keys and configurations:
   ```env
   MODEL_SERVER=OPENAI
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_BASE_URL=https://api.openai.com/v1
   OPENAI_MODEL=gpt-4o
   ```

4. Place the `sample_blog_post.json` file in the root directory of the project.

### Running the Code
Run the script to execute the workflows:
```bash
python llm_workflow.py
```

---

## Implementation Documentation

### Basic LLM Workflow
- **Pipeline Workflow**: Executes tasks sequentially:
  1. Extract key points from the blog post.
  2. Generate a concise summary.
  3. Create social media posts for Twitter, LinkedIn, and Facebook.
  4. Generate an email newsletter.

- **DAG Workflow**: Allows tasks to receive input from multiple upstream tasks. For example:
  - The email newsletter task uses both the original blog post and the summary.

### Self-Correction with Reflexion
- Implements quality evaluation and iterative improvement for each task.
- Tasks are re-executed up to a maximum of 3 attempts if the quality score is below 0.8.

### Advanced Agent-Driven Workflow
- An LLM agent dynamically decides which tools to use and in what order.
- The agent completes the workflow by calling the `finish` tool with the final results.

---

## Example Outputs

### Input Blog Post
```json
{
    "title": "The Future of Artificial Intelligence",
    "content": "Artificial Intelligence (AI) is transforming industries worldwide..."
}
```

### Pipeline Workflow Output
```json
{
    "key_points": [
        "AI is automating tasks across industries.",
        "Key developments include machine learning and NLP.",
        "Ethical concerns like bias are being raised."
    ],
    "summary": "AI is revolutionizing industries but raises ethical concerns.",
    "social_posts": {
        "twitter": "AI is transforming industries but raises ethical concerns. #AI #Future",
        "linkedin": "AI is reshaping industries while posing ethical challenges...",
        "facebook": "Artificial Intelligence is changing the world..."
    },
    "email": {
        "subject": "The Future of AI: Transforming Industries",
        "body": "Dear Reader, AI is transforming industries worldwide..."
    }
}
```

### Reflexion-Enhanced Workflow Output
Reflexion ensures higher-quality outputs by iteratively improving content based on feedback.

### Agent-Driven Workflow Output
The agent dynamically selects tools and generates similar outputs as above but with more flexibility in task ordering.

---

## Analysis of Workflow Approaches

### Strengths of Each Workflow Approach
1. **Pipeline Workflow**:
   - Simple and easy to implement.
   - Suitable for linear workflows where tasks depend on each other sequentially.

2. **DAG Workflow**:
   - More flexible than the pipeline workflow.
   - Allows tasks to receive input from multiple upstream tasks, making it suitable for complex workflows.

3. **Agent-Driven Workflow**:
   - Highly flexible and dynamic.
   - The agent can adaptively decide the best sequence of tasks based on the input and intermediate results.

### Weaknesses of Each Workflow Approach
1. **Pipeline Workflow**:
   - Limited flexibility; tasks must be executed in a fixed order.
   - Errors in early tasks propagate downstream.

2. **DAG Workflow**:
   - Requires careful management of task dependencies.
   - Can become complex for workflows with many interdependent tasks.

3. **Agent-Driven Workflow**:
   - Computationally expensive due to the need for dynamic decision-making.
   - May produce inconsistent results if the agent fails to make optimal decisions.

---

## Challenges Encountered

1. **API Key Management**:
   - Ensuring secure storage and access to API keys was challenging. This was addressed by using a `.env` file and loading environment variables.

2. **Error Handling**:
   - Handling errors from the LLM API required robust exception handling. This was implemented using try-except blocks in the `call_llm` function.

3. **Iterative Improvement**:
   - Implementing Reflexion required designing evaluation criteria and feedback mechanisms. This was achieved by creating detailed evaluation prompts and parsing responses.

4. **Dynamic Task Execution**:
   - The agent-driven workflow required implementing a state management system to track task progress. This was resolved by maintaining a `workflow_state` dictionary.

---

## Conclusion
This project demonstrates the effectiveness of different LLM workflow approaches in repurposing content. While each approach has its strengths and weaknesses, the agent-driven workflow offers the most flexibility and adaptability for complex tasks. Reflexion provides a valuable mechanism for iterative improvement, enhancing the quality of generated content.

For further improvements, consider optimizing the agent's decision-making process and refining the evaluation criteria for Reflexion.

---

Let me know if you need any additional details or adjustments!
