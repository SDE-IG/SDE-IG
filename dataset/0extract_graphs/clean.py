import os
import json
import argparse
from pathlib import Path

# ================= 配置区 =================
DATASETS = {
    "APTMalware": "/moredata_b/users/zfz/APTMalware_Graphs",
    "MOTIF": "/moredata_b/users/zfz/MOTIF_Graphs"
}
# ==========================================

def clean_dataset(dataset_name, base_dir):
    print(f"\n>>> 🧹 开始清理数据集: {dataset_name} ({base_dir})")
    
    if not os.path.exists(base_dir):
        print(f"[-] 目录不存在，跳过: {base_dir}")
        return

    json_files = list(Path(base_dir).rglob("*.json"))
    total_files = len(json_files)
    
    if total_files == 0:
        print("[-] 没有找到任何 JSON 文件。")
        return

    print(f"[*] 发现 {total_files:,} 个文件，开始扫描无效空壳...\n")

    deleted_count = 0
    error_count = 0

    for i, file_path in enumerate(json_files, 1):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            functions = data.get("functions", [])
            
            # 1. 检查函数数量
            if len(functions) == 0:
                os.remove(file_path)
                deleted_count += 1
                print(f"  [🗑️ 删除] {file_path.name} (原因: 函数个数为 0)")
                continue
            
            # 2. 检查基本块 (Nodes) 总数
            # 有时候 Ghidra 识别出了函数名，但函数体内全是未知数据，导致基本块为 0
            total_bbs = sum(len(func.get("nodes", [])) for func in functions)
            if total_bbs == 0:
                os.remove(file_path)
                deleted_count += 1
                print(f"  [🗑️ 删除] {file_path.name} (原因: 函数数为 {len(functions)}，但基本块总数为 0)")

        except json.JSONDecodeError:
            # 如果 JSON 损坏（比如跑一半被强制杀死的残骸），也一并清理
            os.remove(file_path)
            deleted_count += 1
            print(f"  [⚠️ 删除] {file_path.name} (原因: JSON 文件格式损坏)")
        except Exception as e:
            error_count += 1

        # 进度显示
        if i % 500 == 0 or i == total_files:
            print(f"  [进度] 已扫描 {i:,}/{total_files:,} 个文件 \r", end="")

    print(f"\n\n{'='*50}")
    print(f"✨ 【{dataset_name}】清洗报告")
    print(f"{'='*50}")
    print(f"🔍 初始样本总数 : {total_files:,} 个")
    print(f"🗑️ 删除空壳/损坏 : {deleted_count:,} 个")
    print(f"✅ 剩余健康样本 : {total_files - deleted_count:,} 个")
    if error_count > 0:
        print(f"❌ 权限/读取异常 : {error_count:,} 个")
    print(f"{'='*50}\n")

def main():
    parser = argparse.ArgumentParser(description="恶意软件空壳 JSON 清理工具")
    parser.add_argument("--dataset", choices=["APTMalware", "MOTIF", "ALL"], default="ALL")
    args = parser.parse_args()

    if args.dataset in ["APTMalware", "ALL"]:
        clean_dataset("APTMalware", DATASETS["APTMalware"])
        
    if args.dataset in ["MOTIF", "ALL"]:
        clean_dataset("MOTIF", DATASETS["MOTIF"])

if __name__ == "__main__":
    main()