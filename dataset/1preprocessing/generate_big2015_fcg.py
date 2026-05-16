import os
import re
import json
import argparse
import networkx as nx
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ================= 通用配置 =================
BIG2015_BASE_DIR = "/home/zhufengzhou/malware-classification" 
BIG2015_OUTPUT_BASE_DIR = "/home/zhufengzhou/BIG2015_FCG_Graphs"
MAX_WORKERS = 8
# ============================================

def build_fcg_from_asm_and_save(input_asm_path, output_json_path):
    """ 处理 BIG 2015 的 .asm 文件，提取 FCG，剥离机器码，转换为 CLAP 格式 """
    fcg = nx.DiGraph()
    
    # 正则表达式：匹配函数头部和尾部
    func_start_re = re.compile(r'^[.a-zA-Z0-9_]+:[0-9A-Fa-f]+\s+([^\s]+)\s+proc\s+(near|far)')
    func_end_re = re.compile(r'^[.a-zA-Z0-9_]+:[0-9A-Fa-f]+\s+([^\s]+)\s+endp')

    current_func = None
    func_instructions = [] 
    label_to_line = {}     
    line_idx = 1
    edges_to_add = []      

    try:
        # BIG 2015 文件包含杂乱字节，必须使用 latin-1 忽略错误
        with open(input_asm_path, 'r', encoding='latin-1', errors='ignore') as f:
            for line in f:
                # 1. 匹配函数开始
                start_match = func_start_re.match(line)
                if start_match:
                    current_func = start_match.group(1)
                    fcg.add_node(current_func)
                    # 初始化函数内部状态
                    func_instructions = []
                    label_to_line = {}
                    line_idx = 1
                    continue
                
                # 2. 匹配函数结束，执行格式化清洗
                end_match = func_end_re.match(line)
                if end_match:
                    if current_func:
                        clap_lines = []
                        for i, (text, target_func) in enumerate(func_instructions, 1):
                            # Tokenization: 去逗号，中括号加空格
                            cleaned_asm = text.replace(",", " ").replace("[", " [ ").replace("]", " ] ")
                            cleaned_asm = re.sub(r'\s+', ' ', cleaned_asm).strip()
                            
                            # 跳转目标替换为 INSTR<N>
                            parts = cleaned_asm.split()
                            if len(parts) > 1 and parts[0].startswith('j'):
                                for part_idx, part in enumerate(parts):
                                    if part in label_to_line:
                                        parts[part_idx] = f"INSTR{label_to_line[part]}"
                                cleaned_asm = " ".join(parts)
                                
                            clap_lines.append(f"{i}: {cleaned_asm}")
                            
                            # 收集真实的函数调用边
                            if target_func:
                                edges_to_add.append((current_func, target_func))
                                
                        # 将整理好的文本存入该节点
                        fcg.nodes[current_func]['assembly'] = "\n".join(clap_lines)
                        current_func = None
                    continue

                # 3. 函数内部指令提取与机器码智能剥离
                if current_func:
                    code_part = line.split(';')[0].strip()
                    if not code_part: continue
                        
                    tokens = code_part.split()
                    if len(tokens) == 0: continue
                    
                    # 剥离段地址头 (如 .text:0040105E)
                    if re.match(r'^[.a-zA-Z0-9_]+:[0-9A-Fa-f]+$', tokens[0]):
                        tokens = tokens[1:] 
                        
                    # 剥离十六进制机器码，暴露真实的汇编助记符
                    while len(tokens) > 0 and re.match(r'^[0-9A-Fa-f]{2}$', tokens[0]):
                        # 保护恰好是两位的合法伪指令
                        if tokens[0].lower() in ['db', 'dw', 'dd']:
                            break
                        tokens = tokens[1:]
                        
                    if len(tokens) == 0: continue
                        
                    # 记录跳转标号位置
                    if tokens[0].endswith(':'):
                        label_name = tokens[0][:-1]
                        label_to_line[label_name] = line_idx
                        tokens = tokens[1:] 
                        
                    if len(tokens) == 0: continue
                        
                    # 过滤无用数据伪指令
                    if tokens[0] in ['assume', 'align', 'db', 'dw', 'dd']:
                        continue
                        
                    raw_asm = " ".join(tokens)
                    target_func = None
                    
                    # 精准提取 call 边
                    if tokens[0] == 'call' and len(tokens) > 1:
                        t_func = tokens[-1].replace('ds:', '').replace('__imp_', '')
                        # 核心修正：过滤寄存器，过滤 loc_ 开头的局部标号
                        if t_func.lower() not in ['eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'ebp', 'esp']:
                            if not t_func.startswith('loc_') and not t_func.startswith('offset '):
                                target_func = t_func
                            
                    func_instructions.append((raw_asm, target_func))
                    line_idx += 1

        # 4. 统一添加 FCG 边
        # 注意：像 GetProcAddress 这种外部 API 会在这里被创建为 assembly="" 的节点
        for u, v in edges_to_add:
            if not fcg.has_node(v):
                fcg.add_node(v, assembly="") 
            fcg.add_edge(u, v)

        # 5. 导出 JSON
        node_link_data = nx.node_link_data(fcg)
        with open(output_json_path, 'w', encoding='utf-8') as f_out:
            json.dump(node_link_data, f_out, ensure_ascii=False, indent=2)
            
        return True, f"成功 | 节点: {fcg.number_of_nodes()}, 边: {fcg.number_of_edges()}"
        
    except FileNotFoundError:
        return False, "失败 | 找不到文件"
    except Exception as e:
        return False, f"失败 | 异常: {str(e)}"

def process_single_task(input_path, output_path):
    """ 多进程 Worker """
    file_name = Path(input_path).name
    success, msg = build_fcg_from_asm_and_save(input_path, output_path)
    return success, f"[BIG2015] {file_name[:20]}... -> {msg}"

def main():
    print(f"\n>>> 开始处理 BIG 2015 转换为纯净 FCG")
    if not os.path.exists(BIG2015_BASE_DIR):
        print(f"[-] 找不到输入目录: {BIG2015_BASE_DIR}")
        return
        
    tasks = []
    skipped_count = 0
    
    # 自动遍历 train 和 test
    for split_folder in ["train", "test"]:
        split_in_dir = os.path.join(BIG2015_BASE_DIR, split_folder)
        if not os.path.isdir(split_in_dir):
            continue
            
        split_out_dir = os.path.join(BIG2015_OUTPUT_BASE_DIR, split_folder)
        os.makedirs(split_out_dir, exist_ok=True)
        
        for file_name in os.listdir(split_in_dir):
            if not file_name.endswith(".asm"): 
                continue
                
            input_path = os.path.join(split_in_dir, file_name)
            output_path = os.path.join(split_out_dir, file_name.replace(".asm", "_fcg.json"))
            
            if os.path.exists(output_path):
                skipped_count += 1
                continue
                
            tasks.append((input_path, output_path))

    total_tasks = len(tasks)
    print(f"[*] 扫描完毕！已跳过 {skipped_count} 个存在的 FCG 图。剩余 {total_tasks} 个待处理...")

    if total_tasks == 0:
        return

    success_cnt, fail_cnt = 0, 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(process_single_task, in_path, out_path): Path(in_path).name 
            for in_path, out_path in tasks
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
    print(f"BIG 2015 FCG 转换流水线处理完毕！")
    print(f"新增提取: {success_cnt} 个 | 失败: {fail_cnt} 个")
    print(f"FCG 图文件已归档至: {BIG2015_OUTPUT_BASE_DIR}")
    print("="*50)

if __name__ == "__main__":
    main()