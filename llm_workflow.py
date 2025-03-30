import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the model server type
model_server = os.getenv('MODEL_SERVER', 'GROQ').upper()  # Default to GROQ if not set
if model_server == "GROQ":
    API_KEY = os.getenv('GROQ_API_KEY')
    BASE_URL = os.getenv('GROQ_BASE_URL')
    LLM_MODEL = os.getenv('GROQ_MODEL')
elif model_server == "NGU":
    API_KEY = os.getenv('NGU_API_KEY')
    BASE_URL = os.getenv('NGU_BASE_URL')
    LLM_MODEL = os.getenv('NGU_MODEL')
else:
    raise ValueError(f"Unsupported MODEL_SERVER: {model_server}")

# Initialize the OpenAI client with custom base URL
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Define a function to make LLM API calls
def call_llm(messages, tools=None, tool_choice=None):
    """
    Make a call to the LLM API with the specified messages and tools.
    Args:
        messages: List of message objects
        tools: List of tool definitions (optional)
        tool_choice: Tool choice configuration (optional)
    Returns:
        The API response
    """
    kwargs = {
        "model": LLM_MODEL,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools
    if tool_choice:
        kwargs["tool_choice"] = tool_choice
    try:
        response = client.chat.completions.create(**kwargs)
        return response
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return None

# Task 1.1: Set Up Sample Data
def get_sample_blog_post():
    """
    Read the sample blog post from a JSON file.
    """
    try:
        with open('sample-blog-post.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print("Error: sample_blog_post.json file not found.")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in sample_blog_post.json.")
        return None

# Task 1.2: Define Tool Schemas
extract_key_points_schema = {
    "type": "function",
    "function": {
        "name": "extract_key_points",
        "description": "Extract key points from a blog post",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The title of the blog post"},
                "content": {"type": "string", "description": "The content of the blog post"},
                "key_points": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of key points extracted from the blog post"
                }
            },
            "required": ["key_points"]
        }
    }
}

generate_summary_schema = {
    "type": "function",
    "function": {
        "name": "generate_summary",
        "description": "Generate a concise summary from the key points",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Concise summary of the blog post"}
            },
            "required": ["summary"]
        }
    }
}

create_social_media_posts_schema = {
    "type": "function",
    "function": {
        "name": "create_social_media_posts",
        "description": "Create social media posts for different platforms",
        "parameters": {
            "type": "object",
            "properties": {
                "twitter": {"type": "string", "description": "Post optimized for Twitter/X (max 280 characters)"},
                "linkedin": {"type": "string", "description": "Post optimized for LinkedIn (professional tone)"},
                "facebook": {"type": "string", "description": "Post optimized for Facebook"}
            },
            "required": ["twitter", "linkedin", "facebook"]
        }
    }
}

create_email_newsletter_schema = {
    "type": "function",
    "function": {
        "name": "create_email_newsletter",
        "description": "Create an email newsletter from the blog post and summary",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {"type": "string", "description": "Email subject line"},
                "body": {"type": "string", "description": "Email body content in plain text"}
            },
            "required": ["subject", "body"]
        }
    }
}

# Task 1.3: Implement Task Functions
def task_extract_key_points(blog_post):
    messages = [
        {"role": "system", "content": "You are an expert at analyzing content and extracting key points from articles."},
        {"role": "user", "content": f"Extract the key points from this blog post:\n\nTitle: {blog_post['title']}\n\nContent: {blog_post['content']}"}
    ]
    response = call_llm(messages=messages, tools=[extract_key_points_schema], tool_choice={"type": "function", "function": {"name": "extract_key_points"}})
    if response and response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        result = json.loads(tool_call.function.arguments)
        return result.get("key_points", [])
    return []

def task_generate_summary(key_points, max_length=150):
    messages = [
        {"role": "system", "content": "You are an expert at summarizing content concisely while preserving key information."},
        {"role": "user", "content": f"Generate a summary based on these key points, max {max_length} words:\n\n" + "\n".join([f"- {point}" for point in key_points])}
    ]
    response = call_llm(messages=messages, tools=[generate_summary_schema], tool_choice={"type": "function", "function": {"name": "generate_summary"}})
    if response and response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        result = json.loads(tool_call.function.arguments)
        return result.get("summary", "")
    return ""

def task_create_social_media_posts(key_points, blog_title):
    messages = [
        {"role": "system", "content": "You are a social media expert who creates engaging posts optimized for different platforms."},
        {"role": "user", "content": f"Create social media posts for Twitter, LinkedIn, and Facebook based on this blog title: '{blog_title}' and these key points:\n\n" + "\n".join([f"- {point}" for point in key_points])}
    ]
    response = call_llm(messages=messages, tools=[create_social_media_posts_schema], tool_choice={"type": "function", "function": {"name": "create_social_media_posts"}})
    if response and response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        return json.loads(tool_call.function.arguments)
    return {"twitter": "", "linkedin": "", "facebook": ""}

def task_create_email_newsletter(blog_post, summary, key_points):
    messages = [
        {"role": "system", "content": "You are an email marketing specialist who creates engaging newsletters."},
        {"role": "user", "content": f"Create an email newsletter based on this blog post:\n\nTitle: {blog_post['title']}\n\nSummary: {summary}\n\nKey Points:\n" + "\n".join([f"- {point}" for point in key_points])}
    ]
    response = call_llm(messages=messages, tools=[create_email_newsletter_schema], tool_choice={"type": "function", "function": {"name": "create_email_newsletter"}})
    if response and response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        return json.loads(tool_call.function.arguments)
    return {"subject": "", "body": ""}

# Task 1.3: Implement the Pipeline Workflow
def run_pipeline_workflow(blog_post):
    key_points = task_extract_key_points(blog_post)
    summary = task_generate_summary(key_points)
    social_posts = task_create_social_media_posts(key_points, blog_post['title'])
    email = task_create_email_newsletter(blog_post, summary, key_points)
    return {
        "key_points": key_points,
        "summary": summary,
        "social_posts": social_posts,
        "email": email
    }

# Task 1.4: Implement the DAG Workflow
def run_dag_workflow(blog_post):
    key_points = task_extract_key_points(blog_post)
    summary = task_generate_summary(key_points)
    social_posts = task_create_social_media_posts(key_points, blog_post['title'])
    email = task_create_email_newsletter(blog_post, summary, key_points)
    return {
        "key_points": key_points,
        "summary": summary,
        "social_posts": social_posts,
        "email": email
    }

# Task 1.5: Add Chain-of-Thought Reasoning
def extract_key_points_with_cot(blog_post):
    cot_prompt = f"""
    I need to extract the key points from this blog post. Let me think through this step by step.
    Blog Title: {blog_post['title']}
    Blog Content:
    {blog_post['content']}
    First, I'll identify the main topic and purpose of this blog post.
    Next, I'll extract the main arguments or claims made in the post.
    Then, I'll identify any supporting evidence, statistics, or examples.
    Finally, I'll distill these into clear, concise key points.
    Let me work through this methodically:
    """
    messages = [
        {"role": "system", "content": "You are an expert at analyzing content and extracting key points from articles. Think step by step."},
        {"role": "user", "content": cot_prompt}
    ]
    cot_response = call_llm(messages)
    if cot_response:
        messages.append({"role": "assistant", "content": cot_response.choices[0].message.content})
        messages.append({"role": "user", "content": "Based on your analysis, extract the key points in a structured format."})
        final_response = call_llm(messages=messages, tools=[extract_key_points_schema], tool_choice={"type": "function", "function": {"name": "extract_key_points"}})
        if final_response and final_response.choices[0].message.tool_calls:
            tool_call = final_response.choices[0].message.tool_calls[0]
            result = json.loads(tool_call.function.arguments)
            return result.get("key_points", [])
    return task_extract_key_points(blog_post)

# Part 2: Implementing Self-Correction with Reflexion
def evaluate_content(content, content_type):
    evaluation_criteria = {
        "summary": [
            "Conciseness: Is the summary brief and to the point?",
            "Comprehensiveness: Does it cover all the important points?",
            "Clarity: Is the summary clear and easy to understand?",
            "Accuracy: Does it accurately represent the original content?"
        ],
        "social_media_post": [
            "Platform appropriateness: Is the post optimized for the specific platform?",
            "Engagement potential: Is it likely to engage the target audience?",
            "Clarity: Is the message clear and concise?",
            "Call to action: Does it include an effective call to action if appropriate?"
        ],
        "email": [
            "Subject line effectiveness: Is the subject line compelling?",
            "Content quality: Is the email content informative and valuable?",
            "Structure: Is the email well-structured and easy to read?",
            "Call to action: Does it include a clear call to action if appropriate?"
        ],
        "key_points": [
            "Relevance: Are the key points relevant to the main topic?",
            "Completeness: Do they cover all the important aspects?",
            "Clarity: Are the key points clearly articulated?",
            "Conciseness: Are they concise and to the point?"
        ]
    }
    criteria = evaluation_criteria.get(content_type, evaluation_criteria["summary"])
    formatted_content = ""
    if isinstance(content, list):
        formatted_content = "\n".join([f"- {item}" for item in content])
    elif isinstance(content, dict):
        formatted_content = "\n".join([f"{key}: {value}" for key, value in content.items()])
    else:
        formatted_content = content
    evaluation_prompt = f"""
    Please evaluate this {content_type} based on the following criteria:
    {content_type.upper()}:
    {formatted_content}
    EVALUATION CRITERIA:
    {criteria}
    For each criterion, provide a score from 0 to 10 and brief feedback.
    Then provide an overall quality score from 0 to 1 and suggestions for improvement.
    """
    messages = [
        {"role": "system", "content": "You are an expert content evaluator who provides objective and constructive feedback."},
        {"role": "user", "content": evaluation_prompt}
    ]
    response = call_llm(messages)
    if response:
        response_text = response.choices[0].message.content
        quality_score = 0.5
        feedback = response_text
        try:
            import re
            score_match = re.search(r'overall quality score[:\s]+([0-9.]+)', response_text.lower())
            if score_match:
                quality_score = float(score_match.group(1))
                if quality_score > 1:
                    quality_score /= 10
        except:
            pass
        return {"quality_score": quality_score, "feedback": feedback}
    return {"quality_score": 0.5, "feedback": "Unable to evaluate content due to API error."}

def improve_content(content, feedback, content_type):
    formatted_content = ""
    if isinstance(content, list):
        formatted_content = "\n".join([f"- {item}" for item in content])
    elif isinstance(content, dict):
        formatted_content = "\n".join([f"{key}: {value}" for key, value in content.items()])
    else:
        formatted_content = content
    improvement_prompt = f"""
    Please improve this {content_type} based on the following feedback:
    CURRENT {content_type.upper()}:
    {formatted_content}
    FEEDBACK:
    {feedback}
    Please provide an improved version that addresses the feedback.
    """
    messages = [
        {"role": "system", "content": "You are an expert content creator who excels at improving content based on feedback."},
        {"role": "user", "content": improvement_prompt}
    ]
    response = call_llm(messages)
    if response:
        return response.choices[0].message.content
    return content

def generate_with_reflexion(generator_func, max_attempts=3):
    def wrapped_generator(*args, **kwargs):
        content_type = kwargs.pop("content_type", "content")
        content = generator_func(*args, **kwargs)
        for attempt in range(max_attempts):
            evaluation = evaluate_content(content, content_type)
            if evaluation["quality_score"] >= 0.8:
                return content
            improved_content = improve_content(content, evaluation["feedback"], content_type)
            content = improved_content
        return content
    return wrapped_generator

def run_workflow_with_reflexion(blog_post):
    extract_key_points_with_reflexion = generate_with_reflexion(task_extract_key_points)
    generate_summary_with_reflexion = generate_with_reflexion(task_generate_summary)
    create_social_media_posts_with_reflexion = generate_with_reflexion(task_create_social_media_posts)
    create_email_newsletter_with_reflexion = generate_with_reflexion(task_create_email_newsletter)
    key_points = extract_key_points_with_reflexion(blog_post, content_type="key_points")
    summary = generate_summary_with_reflexion(key_points, content_type="summary")
    social_posts = create_social_media_posts_with_reflexion(key_points, blog_post['title'], content_type="social_media_post")
    email = create_email_newsletter_with_reflexion(blog_post, summary, key_points, content_type="email")
    return {
        "key_points": key_points,
        "summary": summary,
        "social_posts": social_posts,
        "email": email
    }

# Part 3: Advanced Agent-Driven Workflow
def define_agent_tools():
    finish_tool_schema = {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Complete the workflow and return the final results",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "The final summary"},
                    "social_posts": {"type": "object", "description": "The social media posts for each platform"},
                    "email": {"type": "object", "description": "The email newsletter"}
                },
                "required": ["summary", "social_posts", "email"]
            }
        }
    }
    return [
        extract_key_points_schema,
        generate_summary_schema,
        create_social_media_posts_schema,
        create_email_newsletter_schema,
        finish_tool_schema
    ]

def execute_agent_tool(tool_name, arguments, workflow_state):
    if tool_name == "extract_key_points":
        blog_post = workflow_state.get("blog_post", {})
        key_points = arguments.get("key_points", task_extract_key_points(blog_post))
        workflow_state["key_points"] = key_points
        return {"key_points": key_points}, workflow_state
    elif tool_name == "generate_summary":
        key_points = workflow_state.get("key_points", [])
        if not key_points:
            return {"error": "Key points not available. Extract key points first."}, workflow_state
        summary = arguments.get("summary", task_generate_summary(key_points))
        workflow_state["summary"] = summary
        return {"summary": summary}, workflow_state
    elif tool_name == "create_social_media_posts":
        key_points = workflow_state.get("key_points", [])
        blog_post = workflow_state.get("blog_post", {})
        if not key_points:
            return {"error": "Key points not available. Extract key points first."}, workflow_state
        social_posts = arguments
        if "twitter" not in social_posts or "linkedin" not in social_posts or "facebook" not in social_posts:
            social_posts = task_create_social_media_posts(key_points, blog_post.get("title", ""))
        workflow_state["social_posts"] = social_posts
        return social_posts, workflow_state
    elif tool_name == "create_email_newsletter":
        blog_post = workflow_state.get("blog_post", {})
        summary = workflow_state.get("summary", "")
        key_points = workflow_state.get("key_points", [])
        if not summary:
            return {"error": "Summary not available. Generate summary first."}, workflow_state
        if not key_points:
            return {"error": "Key points not available. Extract key points first."}, workflow_state
        email = arguments
        if "subject" not in email or "body" not in email:
            email = task_create_email_newsletter(blog_post, summary, key_points)
        workflow_state["email"] = email
        return email, workflow_state
    elif tool_name == "finish":
        return arguments, workflow_state
    else:
        return {"error": f"Unknown tool: {tool_name}"}, workflow_state

def run_agent_workflow(blog_post):
    system_message = """
    You are a Content Repurposing Agent. Your job is to take a blog post and repurpose it into different formats:
    1. Extract key points from the blog post
    2. Generate a concise summary
    3. Create social media posts for different platforms
    4. Create an email newsletter
    You have access to tools that can help you with each of these tasks. Think carefully about which tools to use and in what order.
    When you're done, use the 'finish' tool to complete the workflow.
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Please repurpose this blog post:\n\nTitle: {blog_post['title']}\n\nContent: {blog_post['content']}"}
    ]
    tools = define_agent_tools()
    workflow_state = {"blog_post": blog_post}
    max_iterations = 10
    for i in range(max_iterations):
        response = call_llm(messages, tools)
        if not response:
            return {"error": "LLM API call failed."}
        messages.append(response.choices[0].message)
        if not response.choices[0].message.tool_calls:
            break
        for tool_call in response.choices[0].message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            if tool_name == "finish":
                return arguments
            tool_result, workflow_state = execute_agent_tool(tool_name, arguments, workflow_state)
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(tool_result) if isinstance(tool_result, dict) else tool_result
            })
    return {"error": "The agent couldn't complete the workflow within the maximum number of iterations."}
# Main Function and Testing Section
def main():
    print("Starting Content Repurposing Workflow...\n")
    
    # Load the sample blog post from the JSON file
    blog_post = get_sample_blog_post()
    if not blog_post:
        print("Failed to load the sample blog post. Exiting...")
        return

    print("Sample Blog Post Loaded Successfully!")
    print(f"Title: {blog_post['title']}")
    print(f"Content: {blog_post['content'][:200]}...")  # Print first 200 characters of content
    print("\nRunning Workflows...\n")

    # Run the pipeline workflow
    print("Pipeline Workflow Results:")
    pipeline_results = run_pipeline_workflow(blog_post)
    print(json.dumps(pipeline_results, indent=4))
    print("\n")
    
    # Run the DAG workflow
    print("DAG Workflow Results:")
    dag_results = run_dag_workflow(blog_post)
    print(json.dumps(dag_results, indent=4))
    print("\n")
    
    # Run the Reflexion-enhanced workflow
    print("Reflexion-Enhanced Workflow Results:")
    reflexion_results = run_workflow_with_reflexion(blog_post)
    print(json.dumps(reflexion_results, indent=4))
    print("\n")

    # Run the agent-driven workflow
    print("Agent-Driven Workflow Results:")
    agent_results = run_agent_workflow(blog_post)
    print(json.dumps(agent_results, indent=4))
    print("\n")

    print("All workflows completed successfully!")
    
# Entry Point
if __name__ == "__main__":
    main()