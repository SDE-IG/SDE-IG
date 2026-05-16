import os
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import networkx as nx

# ================= 配置区 =================
DATASETS = {
    "APTMalware": "/moredata_b/users/zfz/APTMalware_Graphs",
    "MOTIF": "/moredata_b/users/zfz/MOTIF_Graphs"
}
MAX_WORKERS = 16
# ==========================================


def analyze_single_json(json_path):
    """
    解析单个 JSON，构建全局大图，返回详细统计指标
    返回: (成功状态, 函数数, 基本块数, 连通子图数, 文件名, 错误信息)
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        functions = data.get("functions", [])
        call_graph = data.get("call_graph", [])

        func_count = len(functions)
        bb_count = 0

        G = nx.Graph()  # 构建无向图用于计算连通子图
        first_nodes = {}  # 记录每个函数的入口节点，用于连接 Call Graph

        # 1. 遍历所有函数，添加局部节点和边
        for func in functions:
            f_name = func.get("function", "unknown")
            nodes = func.get("nodes", [])
            edges = func.get("edges", [])

            bb_count += len(nodes)

            if not nodes:
                # 应对极端情况：如果函数为空，用一个虚拟节点代表它
                dummy_node = f"{f_name}@@EMPTY"
                G.add_node(dummy_node)
                first_nodes[f_name] = dummy_node
                continue

            # 添加节点
            for n in nodes:
                node_id = f"{f_name}@@{n['id']}"
                G.add_node(node_id)
                # 记录该函数的第一个节点，作为跨函数调用的“锚点”
                if f_name not in first_nodes:
                    first_nodes[f_name] = node_id

            # 添加基本块之间的局部边 (CFG/DFG)
            for e in edges:
                src_id = f"{f_name}@@{e['source']}"
                tgt_id = f"{f_name}@@{e['target']}"
                G.add_edge(src_id, tgt_id)

        # 2. 遍历调用图，跨函数搭桥
        for call in call_graph:
            src_func = call.get("source")
            tgt_func = call.get("target")

            # 将调用者的入口节点与被调用者的入口节点相连
            if src_func in first_nodes and tgt_func in first_nodes:
                G.add_edge(first_nodes[src_func], first_nodes[tgt_func])

        # 3. 计算连通子图数量 (Connected Components)
        cc_count = nx.number_connected_components(G) if len(G.nodes) > 0 else 0

        return True, func_count, bb_count, cc_count, Path(json_path).name, None

    except Exception as e:
        return False, 0, 0, 0, Path(json_path).name, str(e)


def scan_dataset(dataset_name, base_dir):
    print(f"\n>>> 开始深度扫描数据集: {dataset_name} ({base_dir})")

    if not os.path.exists(base_dir):
        print(f"[-] 目录不存在，已跳过: {base_dir}")
        return

    json_files = list(Path(base_dir).rglob("*.json"))
    total_files = len(json_files)

    if total_files == 0:
        print("[-] 没有找到任何 JSON 文件。")
        return

    print(f"[*] 发现 {total_files} 个文件。正在使用 {MAX_WORKERS} 进程分析连通性...\n")

    # 统计用的极值字典
    stats = {
        "max_funcs": {"val": -1, "file": ""},
        "min_funcs": {"val": float("inf"), "file": ""},
        "max_bbs": {"val": -1, "file": ""},
        "min_bbs": {"val": float("inf"), "file": ""},
        "max_cc": {"val": -1, "file": ""},
        "min_cc": {"val": float("inf"), "file": ""},
    }

    success_files = 0
    failed_files = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {executor.submit(analyze_single_json, str(path)): path for path in json_files}
        for i, future in enumerate(as_completed(future_to_file), 1):
            success, f_cnt, bb_cnt, cc_cnt, filename, err = future.result()

            if success:
                success_files += 1
                # 更新函数极值
                if f_cnt > stats["max_funcs"]["val"]:
                    stats["max_funcs"] = {"val": f_cnt, "file": filename}
                if f_cnt < stats["min_funcs"]["val"]:
                    stats["min_funcs"] = {"val": f_cnt, "file": filename}
                # 更新基本块极值
                if bb_cnt > stats["max_bbs"]["val"]:
                    stats["max_bbs"] = {"val": bb_cnt, "file": filename}
                if bb_cnt < stats["min_bbs"]["val"]:
                    stats["min_bbs"] = {"val": bb_cnt, "file": filename}
                # 更新连通子图极值
                if cc_cnt > stats["max_cc"]["val"]:
                    stats["max_cc"] = {"val": cc_cnt, "file": filename}
                if cc_cnt < stats["min_cc"]["val"]:
                    stats["min_cc"] = {"val": cc_cnt, "file": filename}
            else:
                failed_files += 1

            if i % 100 == 0 or i == total_files:
                print(f"    [进度] 已分析 {i}/{total_files} 个样本 \r", end="")

    # 防止所有文件都失败导致 min 值为 inf
    for k in stats:
        if stats[k]["val"] == float("inf"):
            stats[k]["val"] = 0

    print(f"\n\n{'=' * 60}")
    print(f"📊 【{dataset_name}】数据集逐样本深度统计报告")
    print(f"{'=' * 60}")
    print(f"✅ 有效样本总数: {success_files:,} (失败: {failed_files:,})")
    print("-" * 60)
    print(f"🚀 函数规模 (Functions):")
    print(f"  - 最多: {stats['max_funcs']['val']:,} 个 (来自: {stats['max_funcs']['file']})")
    print(f"  - 最少: {stats['min_funcs']['val']:,} 个 (来自: {stats['min_funcs']['file']})")
    print("-" * 60)
    print(f"🧱 基本块规模 (Basic Blocks / Nodes):")
    print(f"  - 最多: {stats['max_bbs']['val']:,} 个 (来自: {stats['max_bbs']['file']})")
    print(f"  - 最少: {stats['min_bbs']['val']:,} 个 (来自: {stats['min_bbs']['file']})")
    print("-" * 60)
    print(f"🕸️ 连通子图数量 (Connected Components):")
    print(f"  - 最碎片化 (最多子图): {stats['max_cc']['val']:,} 个 (来自: {stats['max_cc']['file']})")
    print(f"  - 最完整 (最少子图): {stats['min_cc']['val']:,} 个 (来自: {stats['min_cc']['file']})")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="恶意软件 JSON 全局图结构多维分析")
    parser.add_argument("--dataset", choices=["APTMalware", "MOTIF", "ALL"], default="ALL")
    args = parser.parse_args()

    if args.dataset in ["APTMalware", "ALL"]:
        scan_dataset("APTMalware", DATASETS["APTMalware"])

    if args.dataset in ["MOTIF", "ALL"]:
        scan_dataset("MOTIF", DATASETS["MOTIF"])


if __name__ == "__main__":
    main()