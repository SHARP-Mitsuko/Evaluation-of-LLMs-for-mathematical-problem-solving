import json
import openai
import re
from dotenv import load_dotenv

# ✅ Load API
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.Client(api_key=API_KEY)

def load_jsonl(file_path):
    """ Read JSONL data line by line """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# Load training data
train_data = load_jsonl("train.jsonl")

def clean_number(num_str):
    """ Clean numeric strings by removing commas, spaces, and percentage signs, then convert to integer or float """
    if not num_str:
        return None
    try:
        num_str = re.sub(r"[,%\s]", "", num_str)  # Remove commas, spaces, and percentage signs
        num = float(num_str)  # Convert to float
        return str(int(num)) if num.is_integer() else str(num)  # Format output
    except ValueError:
        return num_str  # Keep the original value if conversion fails

def extract_numeric_answer(text):
    """ Extract the numeric value from the answer """
    numbers = re.findall(r"-?\d+\.?\d*", text)  # Match integers and decimals
    return clean_number(numbers[-1]) if numbers else None  # Use the last number found

def solve_math_problem(question):
    """ Ask GPT-4o to return only the final numeric answer (without units, commas, percentage signs, or unnecessary zeros) """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a statistics master student. Provide only the numeric answer with no explanation, no units, no commas, no percent signs, and no additional text."},
                {"role": "user", "content": question}
            ]
        )
        gpt_answer = response.choices[0].message.content.strip()

        # 🔹 Extract only the numeric part of GPT-4o's response
        numbers = re.findall(r"-?\d+\.?\d*", gpt_answer)  # Extract numbers
        return clean_number(numbers[-1]) if numbers else gpt_answer  # Format output

    except Exception as e:
        return f"❌ API failure: {str(e)}"

def check_answer(gpt_answer, correct_answer):
    """ Compare GPT-generated answer with the correct answer (removing units, commas, percentage signs, and unnecessary zeros) """
    correct_number = extract_numeric_answer(correct_answer)  # Extract numeric part of correct answer
    return gpt_answer == correct_number  # Compare only the numeric values

# Evaluate GPT-4o
correct_count = 0
total = 5000

for i in range(total):
    sample_question = train_data[i]["question"]
    correct_answer = train_data[i]["answer"]
    gpt_answer = solve_math_problem(sample_question)  # Get GPT-4o's answer

    if check_answer(gpt_answer, correct_answer):
        correct_count += 1
    else:
        print(f"❌ Question {i+1} incorrect")
        print(f"   ✅ Correct answer: {extract_numeric_answer(correct_answer)}")
        print(f"   ❌ GPT generated: {gpt_answer}\n")

# Print accuracy
accuracy = (correct_count / total) * 100
print(f"✅ GPT-4o math problem accuracy: {accuracy:.2f}%")
