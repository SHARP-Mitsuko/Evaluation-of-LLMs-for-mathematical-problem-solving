import json
import openai
import re
import time
import matplotlib.pyplot as plt

# 配置 DeepSeek API
API_KEY = "sk-da0025c4e3f84de082271474bd734f96"  # 替换为你的密钥
client = openai.OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com/v1"
)

# 数据加载
def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

train_data = load_jsonl(r"C:\Users\wcf10\Desktop\math_test\gsm8k\train.jsonl")  # 数据集路径

# 答案处理
def clean_number(num_str):
    num_str = re.sub(r"[,%\s]", "", str(num_str))
    try:
        num = float(num_str)
        return str(int(num)) if num.is_integer() else f"{num:.2f}".rstrip('0').rstrip('.')
    except:
        return num_str

# 模型调用（含重试逻辑）
def solve_math_problem(question, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "严格按以下规则回答：1.只返回纯数字 2.小数保留两位 3.无单位或文本"},
                    {"role": "user", "content": question}
                ],
                temperature=0.1
            )
            answer = response.choices[0].message.content
            numbers = re.findall(r"-?\d+\.?\d*", answer)
            return clean_number(numbers[-1]) if numbers else "N/A"
        except Exception as e:
            print(f"第 {attempt+1} 次重试，错误：{str(e)}")
            time.sleep(2)
    return "❌ 超过最大重试次数"

# 主测试流程
total = 7473  # 测试题量（GSM8K完整测试设为7473）
correct_count = 0
error_log = []

for i in range(total):
    try:
        question = train_data[i]["question"]
        correct_answer = train_data[i]["answer"].split("#### ")[-1].strip()  # 适配GSM8K答案格式
        
        pred_answer = solve_math_problem(question)
        correct_clean = clean_number(correct_answer)
        
        if pred_answer == correct_clean:
            correct_count += 1
        else:
            error_log.append({
                "题号": i+1,
                "问题": question,
                "正确答案": correct_clean,
                "模型答案": pred_answer
            })
            print(f"❌ 第 {i+1} 题错误")
        
        time.sleep(0.5)  # 控制请求频率
        
    except Exception as e:
        print(f"数据处理异常：{str(e)}")

# 输出结果
accuracy = (correct_count / total) * 100
print(f"\n✅ 最终正确率：{accuracy:.2f}%")

# 保存错误日志
with open("error_log.json", "w", encoding="utf-8") as f:
    json.dump(error_log, f, ensure_ascii=False, indent=2)

# 可视化
plt.figure(figsize=(10,6))
plt.bar(["正确", "错误"], [correct_count, total-correct_count], color=["#4CAF50", "#F44336"])
plt.title(f"DeepSeek GSM8K 测试结果（{total}题）")
plt.ylabel("题数")
plt.savefig("gsm8k_result.png", dpi=300)
plt.show()
