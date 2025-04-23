import json
import openai #改模型
import re

API_KEY = "#api key"
client = openai.Client(api_key=API_KEY)

def load_jsonl(file_path):
    """ 逐行读取 JSONL 数据 """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# 加载训练数据
train_data = load_jsonl("grade_school_math/data/train.jsonl")

def clean_number(num_str):
    """ 清理数值字符串，去掉逗号、空格、百分号，并转换为整数或浮点数 """
    if not num_str:
        return None
    try:
        num_str = re.sub(r"[,%\s]", "", num_str)  # 去掉逗号、空格、百分号
        num = float(num_str)  # 转换为浮点数
        return str(int(num)) if num.is_integer() else str(num)  # 格式化
    except ValueError:
        return num_str  # 保留无法转换的值

def extract_numeric_answer(text):
    """ 提取答案中的数值 """
    numbers = re.findall(r"-?\d+\.?\d*", text)  # 匹配整数和小数
    return clean_number(numbers[-1]) if numbers else None  # 取最后一个数值

def solve_math_problem(question):
    """ 让 GPT-4o 仅返回数学题的最终数值答案（去掉单位、逗号、百分号和多余的0） """
    try:
        response = client.chat.completions.create(
            model="gpt-4o", #改模型
            messages=[
                {"role": "system", "content": "You are a math expert. Provide only the numeric answer with no explanation, no units, no commas, no percent signs, and no additional text."},
                {"role": "user", "content": question}
            ]
        )
        gpt_answer = response.choices[0].message.content.strip()

        # 🔹 只保留 GPT-4o 生成答案的数值部分，去掉单位、逗号、百分号和多余的0
        numbers = re.findall(r"-?\d+\.?\d*", gpt_answer)  # 提取数字
        return clean_number(numbers[-1]) if numbers else gpt_answer  # 处理格式

    except Exception as e:
        return f"❌ API 失败: {str(e)}"

def check_answer(gpt_answer, correct_answer):
    """ 直接比对 GPT 生成的答案和标准答案（去掉单位、逗号、百分号和多余的0） """
    correct_number = extract_numeric_answer(correct_answer)  # 提取正确答案数值
    return gpt_answer == correct_number  # 只比较数值部分

# 评估 GPT-4o 在前 500 道数学题的表现
correct_count = 0
total = 500  # 评估前 500 道题

for i in range(total):
    sample_question = train_data[i]["question"]
    correct_answer = train_data[i]["answer"]
    gpt_answer = solve_math_problem(sample_question)  # GPT-4o 生成答案

    if check_answer(gpt_answer, correct_answer):
        correct_count += 1
    else:
        print(f"❌ 第 {i+1} 题错误")
        print(f"   ✅ 正确答案: {extract_numeric_answer(correct_answer)}")
        print(f"   ❌ GPT 生成: {gpt_answer}\n")

# 输出正确率
accuracy = (correct_count / total) * 100
print(f"✅ GPT-4o 的数学解题正确率: {accuracy:.2f}%")