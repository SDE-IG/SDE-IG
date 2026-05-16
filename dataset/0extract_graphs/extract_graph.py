import os
import argparse
import subprocess
import shutil
import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# 导入自定义子模块 (新增了 run_unzip_zfz)
from unzip_malware import run_unzip, run_unzip_zfz
from fix_nested_zips import run_fix_nested

# ================= 通用配置 =================
GHIDRA_HEADLESS = "/home/zhufengzhou/ghidra_11.0.1_PUBLIC/support/analyzeHeadless"
GHIDRA_PROJECT_DIR = "/tmp/ghidra_projects"
SCRIPT_PATH = "/home/zhufengzhou/ghidra_scripts"
POST_SCRIPT = "ExportBinaryGraph.java"

USER_HOME = "/home/zhufengzhou"

# 错误日志专门存放的目录
LOG_OUTPUT_DIR = "/home/zhufengzhou/extract_logs"

# APTMalware 相关目录
UNZIPPED_BASE_DIR = "/home/zhufengzhou/APTMalware_Unzipped"
APT_OUTPUT_BASE_DIR = "/home/zhufengzhou/APTMalware_Graphs"

# MOTIF 相关目录
MOTIF_SOURCE_DIR = "/home/zhufengzhou/MOTIF/MOTIF/MOTIF_defanged"
MOTIF_OUTPUT_DIR = "/home/zhufengzhou/MOTIF_Graphs"

# Dike 相关目录
DIKE_SOURCE_DIR = "/home/zhufengzhou/DikeDataset/files/malware"
DIKE_OUTPUT_DIR = "/home/zhufengzhou/Dike_Graphs"

# ZfzMalware 相关目录
ZFZ_SOURCE_DIR = "/home/zhufengzhou/Malware_Dataset"
ZFZ_UNZIPPED_BASE_DIR = "/home/zhufengzhou/ZfzMalware_Unzipped"
ZFZ_OUTPUT_BASE_DIR = "/home/zhufengzhou/ZfzMalware_Graphs"

# ================= 核心防御参数 =================
MAX_WORKERS = 4           # 建议4，防止 decompile 裂变吃光内存
MAX_FILE_SIZE_MB = 15.0    # 跳过大于 5MB 的超大文件 (可根据需要修改为 10.0 等)
# ============================================

def process_sample(sample_path, group_name, dest_dir):
    """ 
    调用 Ghidra 处理单个二进制样本 
    """
    sample_file = Path(sample_path)
    binary_name = sample_file.name
    project_name = f"Proj_{group_name.replace(' ', '_')}_{binary_name[-12:]}"
    
    cmd = [
        GHIDRA_HEADLESS, GHIDRA_PROJECT_DIR, project_name,
        "-import", str(sample_path), "-overwrite",
        "-scriptPath", SCRIPT_PATH, "-postScript", POST_SCRIPT, "-deleteProject"
    ]
    
    expected_json_path = os.path.join(USER_HOME, f"{binary_name}_asm_graph.json")
    final_json_path = os.path.join(dest_dir, f"{binary_name}.json")
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True, timeout=300)
        
        if os.path.exists(expected_json_path):
            shutil.move(expected_json_path, final_json_path)
            return True, f"[{group_name}] {binary_name[:20]}... 提取成功", None
        else:
            log_tail = "\n".join(result.stdout.strip().split("\n")[-15:])
            short_msg = f"[{group_name}] {binary_name[:20]}... 失败: 未生成 JSON"
            detailed_msg = f"文件: {sample_path}\n原因: 运行结束但未生成 JSON\n--- Ghidra 日志尾部 ---\n{log_tail}\n"
            return False, short_msg, detailed_msg
            
    # 捕获超时并抢救 JSON
    except subprocess.TimeoutExpired as e:
        if os.path.exists(expected_json_path):
            shutil.move(expected_json_path, final_json_path)
            return True, f"[{group_name}] {binary_name[:20]}... 超时强杀，但抢救到 JSON!", None
            
        # === 核心修复点：安全解码字节流 ===
        out_data = e.stdout
        if isinstance(out_data, bytes):
            out_data = out_data.decode('utf-8', errors='ignore')
            
        log_tail = "\n".join(out_data.strip().split("\n")[-15:]) if out_data else "无输出日志"
        short_msg = f"[{group_name}] {binary_name[:20]}... 失败: 分析超时(>5分钟)"
        detailed_msg = f"文件: {sample_path}\n原因: Ghidra 分析彻底死锁超时\n--- 截断日志 ---\n{log_tail}\n"
        return False, short_msg, detailed_msg

    except subprocess.CalledProcessError as e:
        if os.path.exists(expected_json_path):
            shutil.move(expected_json_path, final_json_path)
            return True, f"[{group_name}] {binary_name[:20]}... 崩溃退出，但抢救到 JSON!", None
            
        # === 核心修复点：安全解码字节流 ===
        out_data = e.stdout
        if isinstance(out_data, bytes):
            out_data = out_data.decode('utf-8', errors='ignore')
            
        log_tail = "\n".join(out_data.strip().split("\n")[-15:]) if out_data else "无输出日志"
        short_msg = f"[{group_name}] {binary_name[:20]}... 失败: Ghidra 崩溃 (退出码 {e.returncode})"
        detailed_msg = f"文件: {sample_path}\n原因: Ghidra 分析崩溃\n--- Ghidra 日志尾部 ---\n{log_tail}\n"
        return False, short_msg, detailed_msg
        
    except Exception as e:
        short_msg = f"[{group_name}] {binary_name[:20]}... 失败: 脚本异常"
        detailed_msg = f"文件: {sample_path}\n原因: 外部脚本发生异常: {str(e)}\n"
        return False, short_msg, detailed_msg

def is_file_too_large(file_path):
    """ 检查文件是否超过限制大小 """
    try:
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        return size_mb > MAX_FILE_SIZE_MB, size_mb
    except OSError:
        return False, 0

def process_apt_dataset(dataset_path):
    samples_dir = os.path.join(dataset_path, "samples")
    if not run_unzip(samples_dir, UNZIPPED_BASE_DIR):
        return
    run_fix_nested(UNZIPPED_BASE_DIR)

    print(f"\n>>> 阶段 3: 开始提取 APTMalware 图特征")
    tasks = []
    skipped_count = 0
    large_skipped_count = 0
    
    for apt_group in os.listdir(UNZIPPED_BASE_DIR):
        group_path = os.path.join(UNZIPPED_BASE_DIR, apt_group)
        if not os.path.isdir(group_path):
            continue
            
        dest_group_dir = os.path.join(APT_OUTPUT_BASE_DIR, apt_group)
        os.makedirs(dest_group_dir, exist_ok=True)
        
        for file_name in os.listdir(group_path):
            sample_path = os.path.join(group_path, file_name)
            if os.path.isfile(sample_path):
                expected_json = os.path.join(dest_group_dir, f"{file_name}.json")
                if os.path.exists(expected_json):
                    skipped_count += 1
                    continue
                
                # 新增：超大文件过滤
                too_large, size_mb = is_file_too_large(sample_path)
                if too_large:
                    print(f"[!] 跳过超大文件 ({size_mb:.1f} MB): {file_name}")
                    large_skipped_count += 1
                    continue
                    
                tasks.append((sample_path, apt_group, dest_group_dir))

    print(f"[*] 发现 {large_skipped_count} 个超大文件已自动跳过。")
    execute_tasks(tasks, skipped_count, APT_OUTPUT_BASE_DIR, "APTMalware")

def process_motif_dataset(source_dir):
    print(f"\n>>> 开始提取 MOTIF 数据集图特征")
    os.makedirs(MOTIF_OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(source_dir):
        print(f"[-] 找不到 MOTIF 源目录: {source_dir}")
        return

    tasks = []
    skipped_count = 0
    large_skipped_count = 0

    for file_name in os.listdir(source_dir):
        sample_path = os.path.join(source_dir, file_name)
        if os.path.isfile(sample_path):
            expected_json = os.path.join(MOTIF_OUTPUT_DIR, f"{file_name}.json")
            if os.path.exists(expected_json):
                skipped_count += 1
                continue
                
            too_large, size_mb = is_file_too_large(sample_path)
            if too_large:
                print(f"[!] 跳过超大文件 ({size_mb:.1f} MB): {file_name}")
                large_skipped_count += 1
                continue
                
            tasks.append((sample_path, "MOTIF", MOTIF_OUTPUT_DIR))

    print(f"[*] 发现 {large_skipped_count} 个超大文件已自动跳过。")
    execute_tasks(tasks, skipped_count, MOTIF_OUTPUT_DIR, "MOTIF")

def process_dike_dataset(source_dir):
    print(f"\n>>> 开始提取 Dike 数据集图特征")
    os.makedirs(DIKE_OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(source_dir):
        print(f"[-] 找不到 Dike 源目录: {source_dir}")
        return

    tasks = []
    skipped_count = 0
    large_skipped_count = 0

    for file_name in os.listdir(source_dir):
        sample_path = os.path.join(source_dir, file_name)
        if os.path.isfile(sample_path) and file_name.lower().endswith('.exe'):
            expected_json = os.path.join(DIKE_OUTPUT_DIR, f"{file_name}.json")
            if os.path.exists(expected_json):
                skipped_count += 1
                continue
                
            too_large, size_mb = is_file_too_large(sample_path)
            if too_large:
                print(f"[!] 跳过超大文件 ({size_mb:.1f} MB): {file_name}")
                large_skipped_count += 1
                continue
                
            tasks.append((sample_path, "Dike", DIKE_OUTPUT_DIR))

    print(f"[*] 发现 {large_skipped_count} 个超大文件已自动跳过。")
    execute_tasks(tasks, skipped_count, DIKE_OUTPUT_DIR, "Dike")

def process_zfz_dataset(source_dir):
    print(f"\n>>> 开始处理 ZfzMalware 数据集")
    if not run_unzip_zfz(source_dir, ZFZ_UNZIPPED_BASE_DIR):
        return

    print(f"\n>>> 阶段 3: 开始提取 ZfzMalware 图特征")
    tasks = []
    skipped_count = 0
    large_skipped_count = 0
    
    if not os.path.exists(ZFZ_UNZIPPED_BASE_DIR):
        print(f"[-] 找不到解压后的目录: {ZFZ_UNZIPPED_BASE_DIR}")
        return

    for family in os.listdir(ZFZ_UNZIPPED_BASE_DIR):
        family_path = os.path.join(ZFZ_UNZIPPED_BASE_DIR, family)
        if not os.path.isdir(family_path):
            continue
            
        for subset in os.listdir(family_path):
            subset_path = os.path.join(family_path, subset)
            if not os.path.isdir(subset_path):
                continue
                
            dest_subset_dir = os.path.join(ZFZ_OUTPUT_BASE_DIR, family, subset)
            os.makedirs(dest_subset_dir, exist_ok=True)
            
            for file_name in os.listdir(subset_path):
                sample_path = os.path.join(subset_path, file_name)
                if os.path.isfile(sample_path):
                    expected_json = os.path.join(dest_subset_dir, f"{file_name}.json")
                    if os.path.exists(expected_json):
                        skipped_count += 1
                        continue
                    
                    # 新增：超大文件过滤
                    too_large, size_mb = is_file_too_large(sample_path)
                    if too_large:
                        print(f"[!] 跳过超大文件 ({size_mb:.1f} MB): {family}_{subset} / {file_name}")
                        large_skipped_count += 1
                        continue
                    
                    group_name = f"{family}_{subset}" 
                    tasks.append((sample_path, group_name, dest_subset_dir))

    print(f"[*] 发现 {large_skipped_count} 个超大文件已自动跳过。")
    execute_tasks(tasks, skipped_count, ZFZ_OUTPUT_BASE_DIR, "ZfzMalware")

def execute_tasks(tasks, skipped_count, output_dir, dataset_name):
    total_tasks = len(tasks)
    print(f"[*] {dataset_name} 第三阶段扫描完毕！已提取跳过 {skipped_count} 个。剩余 {total_tasks} 个待跑图...")

    if total_tasks == 0:
        print("[*] 所有样本特征均已提取完毕，无需运行！")
        return

    os.makedirs(LOG_OUTPUT_DIR, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(LOG_OUTPUT_DIR, f"error_log_{dataset_name}_{current_time}.txt")

    success_cnt, fail_cnt = 0, 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(process_sample, path, group, dest): Path(path).name 
            for path, group, dest in tasks
        }
        
        for future in as_completed(future_to_task):
            success, short_msg, detailed_msg = future.result()
            if success:
                success_cnt += 1
                print(f"[+] {short_msg}")
            else:
                fail_cnt += 1
                print(f"[-] {short_msg} (已记录至日志)")
                
                with open(log_file_path, "a", encoding="utf-8") as log_file:
                    log_file.write(f"{'='*60}\n")
                    log_file.write(f"时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    log_file.write(f"{detailed_msg}\n")

    print("\n" + "="*50)
    print(f"{dataset_name} 全流水线处理完毕！")
    print(f"本次新增提取: {success_cnt} 个 | 失败: {fail_cnt} 个")
    if fail_cnt > 0:
        print(f"[*] 详细报错日志已生成至: {log_file_path}")
    print(f"高质量汇编图归档至: {output_dir}")
    print("="*50)

def main():
    os.makedirs(GHIDRA_PROJECT_DIR, exist_ok=True)
    parser = argparse.ArgumentParser(description="恶意软件图特征提取全自动流水线")
    parser.add_argument("--dataset", type=str, choices=["APTMalware", "MOTIF", "Dike", "ZfzMalware"], default="APTMalware")
    args = parser.parse_args()

    if args.dataset == "APTMalware":
        apt_base_path = "/home/zfz/APTMalware"
        process_apt_dataset(apt_base_path)
    elif args.dataset == "MOTIF":
        process_motif_dataset(MOTIF_SOURCE_DIR)
    elif args.dataset == "Dike":
        process_dike_dataset(DIKE_SOURCE_DIR)
    elif args.dataset == "ZfzMalware":
        process_zfz_dataset(ZFZ_SOURCE_DIR)

if __name__ == "__main__":
    main()