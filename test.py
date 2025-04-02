'''
临时测试用代码
随便写的，没什么用
'''


import os
import re
import jieba

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

stopwords = load_stopwords('./Data/stopwords.txt')

with open("./Data/trec06c/utf8/000/000",'r',encoding='utf-8') as f:
    raw_text = f.read()
    cleaned_text = preprocess(raw_text, stopwords)
    print(cleaned_text)
