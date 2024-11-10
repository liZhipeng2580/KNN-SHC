# -*- coding: utf-8 -*-
import re
import pandas as pd
data = []
# 读取日志文件
log_file_path = 'iris_2024-10-29_10-08-25.log'
with open(log_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 提取需要的信息
        if "NMI" in line:
            dataset = line.split(": ")[1].strip()
        elif "生成了" in line:
            pseudo_labels = re.findall(r'\d+', line)[0]
        elif "分配簇的ACC为" in line:
            acc = float(line.split(": ")[1].strip())
        elif "迭代" in line and "损失函数" in line:
            loss = float(re.findall(r'[\d.]+', line)[-1])
            iteration = re.findall(r'迭代 (\d+)', line)[0]

            # 保存数据到列表
            data.append({
                "Dataset": dataset,
                "Pseudo Labels": pseudo_labels,
                "Iteration": iteration,
                "Accuracy": acc,
                "Loss": loss
            })

    # 转换为 DataFrame
    df = pd.DataFrame(data)

    # 保存到 Excel 文件
    excel_file = 'results.xlsx'
    df.to_excel(excel_file, index=False)

    print(f"数据已保存到 {excel_file}")

# 保存到 Excel
excel_file_path = 'output.xlsx'
df.to_excel(excel_file_path, index=False)
