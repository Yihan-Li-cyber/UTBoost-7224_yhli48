import json
import sys
import os
import shutil
from pathlib import Path
from glob import glob


def convert_jsonl_to_json(input_filepath, output_filepath):
    """
    将JSONL文件转换为特定格式的JSON文件

    参数:
        input_filepath: 输入的JSONL文件路径
        output_filepath: 输出的JSON文件路径
    """
    result = {}

    with open(input_filepath, 'r', encoding='utf-8') as infile:
        for line in infile:
            print(line)

            data = json.loads(line.strip())
            instance_id = data.get("instance_id")

            # 确定使用哪个diff内容
            diff_content = data.get("model_patch")
            if not diff_content:
                diff_content = data.get("raw_model_patch")



            result[instance_id] = diff_content


    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        json.dump(result, outfile, indent=2, ensure_ascii=False)




def convert_existing_json(input_json_path, output_json_path):
    """
    将已有JSON文件转换为新的嵌套格式

    参数:
        input_json_path: 上一步生成的JSON文件路径
        output_json_path: 新的输出文件路径
    """
    # 读取原始格式数据
    with open(input_json_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    # 转换数据结构
    nested_data = {
        instance_id: {
            "aug_test": diff_content,
            "augFail2Pass": []
        }
        for instance_id, diff_content in original_data.items()
    }

    # 写入新格式
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(nested_data, f, indent=2, ensure_ascii=False)



if __name__ == '__main__':
    '''
    input_file = '/Users/maoqing/Desktop/starter_task/UTBoost-7224/UTGenerator/results_224/new_gen_testCase_t099_lm01/output_1_processed.jsonl'
    output_file = '/Users/maoqing/Desktop/starter_task/UTBoost-7224/UTGenerator/results_224/new_gen_testCase_t099_lm01/bench.json'
    convert_jsonl_to_json(input_file, output_file)
    '''

    convert_existing_json(
        input_json_path='/Users/maoqing/Desktop/starter_task/UTBoost-7224/UTGenerator/results_224/new_gen_testCase_t099_lm01/bench.json',
        output_json_path='/Users/maoqing/Desktop/starter_task/UTBoost-7224/UTGenerator/results_224/new_gen_testCase_t099_lm01/case.json'
    )