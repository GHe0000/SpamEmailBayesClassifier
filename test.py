'''
原始数据处理
'''

import os
import re
from itertools import islice
import jieba


# 读取停用词
def load_stopwords(stopwords_file):
    stopwords = []
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            line = line.replace(' ', '')
            if line == '' or line.isalpha():
                continue
            words = line.split(',')
            stopwords.extend(words)
    return stopwords


# 预处理单个函数
def preprocess(raw_text, stopwords):
    header_end = re.search(r'\n\s*\n', raw_text)
    body = raw_text[header_end.end():] if header_end else raw_text
    cleaned = re.sub(r'=\?gb2312\?B\?.*?\?=', ' ', body)
    cleaned = re.sub(r'<.*?>', ' ', cleaned)
    cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', cleaned)
    words = jieba.lcut(cleaned)
    filtered = [
        word.lower() for word in words
        if len(word) > 1
        and word not in stopwords
        and not re.match(r'^\d+$', word)
        and not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', word)
    ]
    return ' '.join(filtered)


if __name__ == "__main__":
    index_file = "./Data/trec06c/delay/index"
    N = 4000  # 读取前 N 条邮件

    stopwords_file = "./Data/stopwords.txt"
    stopwords = load_stopwords(stopwords_file)

    output_dir = "./Data/PD2"

    # 创建输出目录
    for label in ['spam', 'ham']:
         if not os.path.exists(os.path.join(output_dir, label.lower())):
             os.makedirs(os.path.join(output_dir, label.lower()))
    
    spam_count = 0
    ham_count = 0
    with open(index_file, 'r', encoding='utf-8') as f:

        for line in islice(f, N):
            line = line.strip()
            if not line:
                continue

            # 解析标签和路径
            try:
                label, path = re.split(r'\s+', line, 1)
                path = os.path.abspath(os.path.join(os.path.dirname(index_file), path))
                path = re.sub(r'data', 'utf8', path)
                if label in ['SPAM', 'HAM']:
                    continue
                label = label.lower()
                if label not in ['spam', 'ham']:
                    continue

                if label =='spam':
                    spam_count += 1
                    if(spam_count == 1028):
                        print(path)
                else:
                    ham_count += 1
            except ValueError:
                continue

            # 读取邮件内容
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                try:
                    with open(path, 'r', encoding='gbk') as f:
                        content = f.read()
                except:
                    print(f"Encoding error: {path}")
                    continue
            except Exception as e:
                print(f"Read failed: {path}: {str(e)}")
                continue

            # 预处理内容
            processed = preprocess(content, stopwords)

            # 生成存储路径
            filename = f"{spam_count if label =='spam' else ham_count}"
            output_path = os.path.join(output_dir, label.lower(), f"{filename}.txt")
            # print(f"{output_path}")

            # 写入处理结果
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(processed)
                # print(f"{path} -> {output_path}")
            except Exception as e:
                print(f"Write failed: {output_path}: {str(e)}")
    print(f"Processed {spam_count} spam and {ham_count} ham emails.")
