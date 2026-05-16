import os
import re
import json
import argparse
import networkx as nx
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ================= 通用配置 =================
# APTMalware 相关目录
APT_INPUT_BASE_DIR = "/home/zhufengzhou/APTMalware_Graphs"
APT_OUTPUT_BASE_DIR = "/home/zhufengzhou/APTMalware_FCG_Graphs"

# Dike 相关目录
DIKE_INPUT_DIR = "/home/zhufengzhou/Dike_Graphs"
DIKE_OUTPUT_DIR = "/home/zhufengzhou/Dike_FCG_Graphs"

# ZfzMalware 相关目录 (新增)
ZFZ_INPUT_BASE_DIR = "/home/zhufengzhou/ZfzMalware_Graphs"
ZFZ_OUTPUT_BASE_DIR = "/home/zhufengzhou/ZfzMalware_FCG_Graphs"

MAX_WORKERS = 8
# ============================================

def process_instructions_for_clap(func_data):
    """
    将 Ghidra 导出的带有地址锚点的汇编数据，转换为 CLAP 论文标准格式：
    1. Tokenization: 去除逗号，中括号两边加空格打散。
    2. Address Rebasing: 使用 1 开始的连续行号替换真实地址。
    3. Jump Target: 将跳转目标替换为 INSTR<N> 格式。
    """
    # 提取所有基本块并排序
    sorted_blocks = sorted(func_data.get("nodes", []), key=lambda x: x.get("id", 0))
    flat_instructions = []
    for block in sorted_blocks:
        flat_instructions.extend(block.get("instructions", []))

    # 步骤 1: 建立全局相对行号映射表 (Address Rebasing)
    addr_to_index = {}
    for idx, inst_dict in enumerate(flat_instructions, 1):  # 从 1 开始编号
        addr = inst_dict.get("addr")
        if addr:
            addr_to_index[addr] = idx

    # 步骤 2: 格式化指令与替换跳转目标 (Tokenization & Jump Target)
    clap_asm_lines = []
    for idx, inst_dict in enumerate(flat_instructions, 1):
        raw_asm = inst_dict.get("asm", "")
        target_addr = inst_dict.get("target_addr")

        # 预处理：去逗号，并在中括号周围加上空格以完美配合 WordPiece
        # 例如: "mov [rbp+var_18], rax" -> "mov [ rbp+var_18 ] rax"
        cleaned_asm = raw_asm.replace(",", " ")
        cleaned_asm = cleaned_asm.replace("[", " [ ").replace("]", " ] ")
        # 替换多余的空格
        cleaned_asm = re.sub(r'\s+', ' ', cleaned_asm).strip()

        # 处理跳转目标替换
        # 如果存在目标跳转地址，并且该地址在当前函数内部
        if target_addr and target_addr != "null" and target_addr in addr_to_index:
            target_idx = addr_to_index[target_addr]
            
            # 分离出操作码 (如 jz, jmp, call)
            parts = cleaned_asm.split()
            if len(parts) > 0:
                mnemonic = parts[0]
                # 强制替换目标为 INSTR<N>
                cleaned_asm = f"{mnemonic} INSTR{target_idx}"

        # 组装带行号的最终 CLAP 格式文本 (例如 "12: jmp INSTR16")
        final_line = f"{idx}: {cleaned_asm}"
        clap_asm_lines.append(final_line)

    return "\n".join(clap_asm_lines)


def build_fcg_and_save(input_json_path, output_json_path):
    """
    读取原始 Ghidra 导出的 JSON，构建 FCG，并保存为 NetworkX Node-Link JSON 格式
    """
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        fcg = nx.DiGraph()

        # 1. 遍历 functions 构建节点
        for func_data in data.get("functions", []):
            func_name = func_data.get("function")
            if not func_name:
                continue
            
            # 调用 CLAP 格式化核心函数
            full_asm_text = process_instructions_for_clap(func_data)
            
            # 将函数名作为节点 ID，处理好的 CLAP 汇编代码作为节点属性
            fcg.add_node(func_name, assembly=full_asm_text)

        # 2. 遍历 call_graph 构建调用边
        for edge in data.get("call_graph", []):
            src = edge.get("source")
            tgt = edge.get("target")
            
            # 过滤掉不存在的节点（例如未解析的外部导入函数）
            if fcg.has_node(src) and fcg.has_node(tgt):
                fcg.add_edge(src, tgt)

        # 3. 保存为 NetworkX 标准的 Node-Link JSON 格式
        node_link_data = nx.node_link_data(fcg)
        with open(output_json_path, 'w', encoding='utf-8') as f_out:
            json.dump(node_link_data, f_out, ensure_ascii=False, indent=2)
            
        return True, f"成功 | 节点: {fcg.number_of_nodes()}, 边: {fcg.number_of_edges()}"
        
    except json.JSONDecodeError:
        return False, "失败 | JSON 解析错误 (文件可能不完整)"
    except Exception as e:
        return False, f"失败 | 异常: {str(e)}"

def process_single_task(input_path, output_path, group_name="Dike"):
    """ 多进程 Worker 函数 """
    file_name = Path(input_path).name
    success, msg = build_fcg_and_save(input_path, output_path)
    status_str = f"[{group_name}] {file_name[:20]}... -> {msg}"
    return success, status_str

def process_dataset(dataset_name):
    tasks = []
    skipped_count = 0
    
    if dataset_name == "APTMalware":
        print(f"\n>>> 开始处理 APTMalware 转换为 FCG (CLAP 格式)")
        if not os.path.exists(APT_INPUT_BASE_DIR):
            print(f"[-] 找不到输入目录: {APT_INPUT_BASE_DIR}")
            return
            
        for apt_group in os.listdir(APT_INPUT_BASE_DIR):
            group_in_dir = os.path.join(APT_INPUT_BASE_DIR, apt_group)
            if not os.path.isdir(group_in_dir):
                continue
                
            group_out_dir = os.path.join(APT_OUTPUT_BASE_DIR, apt_group)
            os.makedirs(group_out_dir, exist_ok=True)
            
            for file_name in os.listdir(group_in_dir):
                if not file_name.endswith(".json"):
                    continue
                    
                input_path = os.path.join(group_in_dir, file_name)
                output_path = os.path.join(group_out_dir, file_name.replace(".json", "_fcg.json"))
                
                if os.path.exists(output_path):
                    skipped_count += 1
                    continue
                    
                tasks.append((input_path, output_path, apt_group))
                
    elif dataset_name == "Dike":
        print(f"\n>>> 开始处理 Dike 转换为 FCG (CLAP 格式)")
        if not os.path.exists(DIKE_INPUT_DIR):
            print(f"[-] 找不到输入目录: {DIKE_INPUT_DIR}")
            return
            
        os.makedirs(DIKE_OUTPUT_DIR, exist_ok=True)
        for file_name in os.listdir(DIKE_INPUT_DIR):
            if not file_name.endswith(".json"):
                continue
                
            input_path = os.path.join(DIKE_INPUT_DIR, file_name)
            output_path = os.path.join(DIKE_OUTPUT_DIR, file_name.replace(".json", "_fcg.json"))
            
            if os.path.exists(output_path):
                skipped_count += 1
                continue
                
            tasks.append((input_path, output_path, "Dike"))

    # ================= 新增 ZfzMalware 处理逻辑 =================
    elif dataset_name == "ZfzMalware":
        print(f"\n>>> 开始处理 ZfzMalware 转换为 FCG (CLAP 格式)")
        if not os.path.exists(ZFZ_INPUT_BASE_DIR):
            print(f"[-] 找不到输入目录: {ZFZ_INPUT_BASE_DIR}")
            return

        for family in os.listdir(ZFZ_INPUT_BASE_DIR):
            family_in_path = os.path.join(ZFZ_INPUT_BASE_DIR, family)
            if not os.path.isdir(family_in_path):
                continue

            # 修改点：将原来的 ["2021_Train", "Recent_Test"] 替换为 5 个年份
            for subset in ["2021", "2022", "2023", "2024", "2025"]:
                subset_in_path = os.path.join(family_in_path, subset)
                if not os.path.isdir(subset_in_path):
                    continue

                # 创建对应的输出子目录结构
                subset_out_path = os.path.join(ZFZ_OUTPUT_BASE_DIR, family, subset)
                os.makedirs(subset_out_path, exist_ok=True)

                for file_name in os.listdir(subset_in_path):
                    if not file_name.endswith(".json"):
                        continue

                    input_path = os.path.join(subset_in_path, file_name)
                    output_path = os.path.join(subset_out_path, file_name.replace(".json", "_fcg.json"))

                    if os.path.exists(output_path):
                        skipped_count += 1
                        continue

                    # 组合群组名称方便终端显示进度
                    group_name = f"{family}_{subset}"
                    tasks.append((input_path, output_path, group_name))
    # =========================================================

    total_tasks = len(tasks)
    print(f"[*] 扫描完毕！已跳过 {skipped_count} 个存在的 FCG 图。剩余 {total_tasks} 个待处理...")

    if total_tasks == 0:
        return

    success_cnt, fail_cnt = 0, 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(process_single_task, in_path, out_path, group): Path(in_path).name 
            for in_path, out_path, group in tasks
        }
        
        for future in as_completed(future_to_task):
            success, msg = future.result()
            if success:
                success_cnt += 1
                print(f"[+] {msg}")
            else:
                fail_cnt += 1
                print(f"[-] {msg}")

    print("\n" + "="*50)
    print(f"{dataset_name} FCG 转换流水线处理完毕！")
    print(f"新增提取: {success_cnt} 个 | 失败: {fail_cnt} 个")
    
    # 根据数据集动态显示正确的输出目录
    if dataset_name == "APTMalware":
        output_dir = APT_OUTPUT_BASE_DIR
    elif dataset_name == "Dike":
        output_dir = DIKE_OUTPUT_DIR
    elif dataset_name == "ZfzMalware":
        output_dir = ZFZ_OUTPUT_BASE_DIR
        
    print(f"FCG 图文件已归档至: {output_dir}")
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description="恶意软件 FCG 构建流水线 (CLAP 对齐版)")
    # 新增 ZfzMalware 选项，并将默认值设置为 ZfzMalware
    parser.add_argument("--dataset", type=str, choices=["APTMalware", "Dike", "ZfzMalware", "ALL"], default="ZfzMalware")
    args = parser.parse_args()

    if args.dataset in ["APTMalware", "ALL"]:
        process_dataset("APTMalware")
    if args.dataset in ["Dike", "ALL"]:
        process_dataset("Dike")
    if args.dataset in ["ZfzMalware", "ALL"]:
        process_dataset("ZfzMalware")

if __name__ == "__main__":
    main()