import os
import json
import argparse
from pathlib import Path

# ================= 通用配置 =================
APT_OUTPUT_BASE_DIR = "/home/zhufengzhou/APTMalware_FCG_Graphs"
DIKE_OUTPUT_DIR = "/home/zhufengzhou/Dike_FCG_Graphs"
ZFZ_OUTPUT_BASE_DIR = "/home/zhufengzhou/ZfzMalware_FCG_Graphs"
# ============================================

def clean_directory(base_dir, dataset_name):
    """
    遍历指定目录，清理节点数为0或边数为0的JSON文件。
    兼容多级目录结构 (如 ZfzMalware 的 家族/测试集 结构)
    """
    print(f"\n>>> 开始清理 {dataset_name} 数据集: {base_dir}")
    if not os.path.exists(base_dir):
        print(f"[-] 找不到目录: {base_dir}")
        return

    scanned_count = 0
    deleted_count = 0
    error_count = 0

    # 使用 os.walk 自动递归遍历所有子目录
    for root, _, files in os.walk(base_dir):
        for file_name in files:
            if not file_name.endswith(".json"):
                continue
                
            file_path = os.path.join(root, file_name)
            scanned_count += 1
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # NetworkX 导出的 node-link 格式包含 "nodes" 和 "links" 列表
                num_nodes = len(data.get("nodes", []))
                num_edges = len(data.get("links", []))
                
                if num_nodes == 0 or num_edges == 0:
                    # 关闭文件后执行删除操作
                    os.remove(file_path)
                    deleted_count += 1
                    # 打印被删除的文件名及原因
                    short_path = Path(file_path).relative_to(base_dir)
                    print(f"[-] 删除: {short_path} (节点: {num_nodes}, 边: {num_edges})")
                    
            except json.JSONDecodeError:
                # 如果遇到 JSON 损坏的文件，也一并清理掉
                os.remove(file_path)
                deleted_count += 1
                error_count += 1
                short_path = Path(file_path).relative_to(base_dir)
                print(f"[!] 删除损坏文件 (JSON 解析失败): {short_path}")
            except Exception as e:
                error_count += 1
                print(f"[x] 读取异常 {file_name}: {str(e)}")

    print("-" * 50)
    print(f"[*] {dataset_name} 清理完毕！")
    print(f"总计扫描: {scanned_count} 个文件")
    print(f"成功清理: {deleted_count} 个无效图")
    if error_count > 0:
        print(f"异常跳过: {error_count} 个文件")
    print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="清理无效的 FCG (节点为0或边为0的图)")
    parser.add_argument("--dataset", type=str, choices=["APTMalware", "Dike", "ZfzMalware", "ALL"], default="ALL")
    args = parser.parse_args()

    if args.dataset in ["APTMalware", "ALL"]:
        clean_directory(APT_OUTPUT_BASE_DIR, "APTMalware")
        
    if args.dataset in ["Dike", "ALL"]:
        clean_directory(DIKE_OUTPUT_DIR, "Dike")
        
    if args.dataset in ["ZfzMalware", "ALL"]:
        clean_directory(ZFZ_OUTPUT_BASE_DIR, "ZfzMalware")

if __name__ == "__main__":
    main()