from itertools import product

# 定义分组
groups = {
    "group_0": {
        0: "person",
        1: "bicycle",
        2: "truck",
        3: "traffic light"
    },
    "group_1": {
        0: "bird",
        1: "cat",
        2: "dog",
        3: "horse"
    },
    "group_2": {
        0: "backpack",
        1: "umbrella",
        2: "handbag",
        3: "suitcase"
    },
    "group_3": {
        0: "tv",
        1: "laptop",
        3: "cell phone"
    },
    "group_4": {
    0: "cup",
    1: "fork",
    2: "knife",
    3: "spoon",
    4: "bowl"
    }
}


def generate_combinations(group):
    items = list(group.values())
    combinations = []
    # 排列组合（数量从 0 到 max_count，每种物体的数量独立）
    for counts in product(range(3 + 1), repeat=len(items)):
        # 排除所有数量为 0 的组合
        if all(count == 0 for count in counts):
            continue
        
        combination = []
        for count, item in zip(counts, items):
            if count > 0:
                combination.append(f"{count} {item}")
            else:
                combination.append(f"no {item}")
        combinations.append(" ".join(combination))
    return combinations

# 生成所有组的字符串组合
for group_name, group in groups.items():
    print(f"Combinations for {group_name}:")
    combinations = generate_combinations(group)
    for combo in combinations:
        print(combo)
    print()

def concatenate_words(word_list):
    # 使用join将单词列表拼接成字符串
    return "".join(word_list)

# 示例输入
outputs = ['', '', '```python\n', '', '# ', 'Initialize ', 'the ', 'canvas\n', '', 'canvas ', '= ', 'Canvas()\n\n', '', '# ', 'Set ', 'a ', 'global ', 'description ', 'for ', 'the ', 'canvas\n', '', '', '', '', 'canvas.set_global_description(\n', '   ', ' ', '', '', "description='A ", 'scene ', 'featuring ', 'two ', '', 'people, ', 'three ', '', 'trucks, ', 'two ', '', 'bikes, ', 'and ', 'two ', 'traffic ', "lights.',\n", '   ', ' ', '', '', 'detailed_descriptions=[\n', '       ', ' ', '', "'The ", 'imag']

# 调用函数并打印结果
result = concatenate_words(outputs)
print(result)
