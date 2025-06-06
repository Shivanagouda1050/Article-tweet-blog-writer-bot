from dataclasses import dataclass
import os
from dotenv import load_dotenv
from groq import Client
import json
import re

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

client = Client(api_key=api_key)

# Define your tool decorator
def tool(func):
    func.is_tool = True
    return func

# Tools registry and binding
class ModelWithTools:
    def __init__(self, model_client):
        self.model = model_client
        self.tools = {}

    def bind_tools(self, tools_list):
        for t in tools_list:
            if getattr(t, "is_tool", False):
                self.tools[t.__name__] = t
        return self

    def extract_json(self, text):
        # Try to extract JSON substring from text (handles extra text)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group(0)
        return text

    def invoke(self, user_input):
        prompt = f"""
        You are a function-calling assistant.  
        Based on user input, choose one of these functions: {list(self.tools.keys())}  
        Return a JSON object with 'name' and 'args' fields.

        User input: {user_input}

        Example JSON:
        {{
          "name": "write_article",
          "args": {{"subject": "AI"}}
        }}
        """

        response = self.model.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150,
        )
        resp_text = response.choices[0].message.content.strip()
        json_text = self.extract_json(resp_text)
        try:
            tool_call = json.loads(json_text)
            name = tool_call.get("name")
            args = tool_call.get("args") or {}
            if not isinstance(args, dict):
                args = {}
        except json.JSONDecodeError:
            return {"error": "Model output not JSON", "raw_output": resp_text}

        if name not in self.tools:
            return {"error": f"Unknown tool {name}", "raw_output": resp_text}

        # Call the tool function with args
        try:
            result = self.tools[name](**args)
        except Exception as e:
            return {"error": f"Tool execution error: {e}"}

        self.last_tool_call = {"name": name, "args": args}
        return {"result": result, "tool_call": self.last_tool_call}

# Define your tools
@tool
def write_article(subject: str):
    prompt = f"Write a detailed article of 200-300 tokens about {subject}."
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()

@tool
def write_comic(subject: str):
    prompt = f"Write a comic of 150-200 tokens about {subject}."
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()
@tool
def write_tweet(subject: str):
    prompt = f"Write a tweet of 80-100 tokens about {subject}."
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()
# Usage
model_with_tools = ModelWithTools(client).bind_tools([write_article, write_comic, write_tweet])
