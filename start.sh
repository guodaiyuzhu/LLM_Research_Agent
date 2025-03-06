#!/bin/bash

source /root/anaconda3/bin/activate python310

#获取当前文件所在目录的绝对路径
currentDir = $(dirname "$(readLink -f "$0")")

# 拼接字符串 main.py 到路径末尾
fullPath = "$currentDir/main.py"

# 输出最终路径
echo "Full path to main.py: $fullPath"

nohup python3 $fullPath >/dev/null 2>&1 &

echo "run success!"