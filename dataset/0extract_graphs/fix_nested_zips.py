import os
import zipfile
import shutil

def run_fix_nested(unzipped_dir):
    """
    供主脚本调用的内层套娃清理函数 (完美断点续传)
    """
    if not os.path.exists(unzipped_dir):
        print(f"[-] 找不到目录: {unzipped_dir}")
        return

    fixed_count = 0
    error_count = 0
    skip_count = 0
    already_binary_count = 0
    PASSWORDS = [None, b"infected", b"virus"]

    print(f"\n>>> 阶段 2: 扫描并清理套娃压缩包 (剥离非二进制载荷)...")
    
    for root, dirs, files in os.walk(unzipped_dir):
        for filename in files:
            file_path = os.path.join(root, filename)

            # === 阶段 2 断点续传 ===
            # 只有当文件仍是 ZIP 结构时才处理；如果已经是 PE/ELF，直接跳过
            if not zipfile.is_zipfile(file_path):
                already_binary_count += 1
                continue
            # =======================

            try:
                with zipfile.ZipFile(file_path, 'r') as zf:
                    inner_files = zf.namelist()
                    if len(inner_files) == 0:
                        continue
                    
                    payload_file = None
                    for name in inner_files:
                        if name.lower().endswith(('.exe', '.dll', '.bin', '.elf', '.sys', '.ocx')):
                            payload_file = name
                            break
                    
                    # 里面全是 xml等非二进制文档，无需替换，留给Ghidra自然过滤
                    if not payload_file:
                        skip_count += 1
                        continue

                    temp_path = file_path + ".temp"
                    success = False
                    
                    for pwd in PASSWORDS:
                        try:
                            with zf.open(payload_file, pwd=pwd) as source, open(temp_path, "wb") as target:
                                shutil.copyfileobj(source, target)
                            success = True
                            break 
                        except RuntimeError as e:
                            if 'password required' in str(e).lower() or 'bad password' in str(e).lower():
                                continue
                            else:
                                raise e 

                    if success:
                        # 原地替换
                        os.replace(temp_path, file_path)
                        print(f"    [+] 成功掏出二进制实体 '{payload_file}' (原名: {filename})")
                        fixed_count += 1
                    else:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        error_count += 1
                        
            except Exception as e:
                error_count += 1

    print(f"[*] 第二阶段扫描完毕！发现 {already_binary_count} 个已是纯正机器码的样本（跳过）。")
    print("-" * 40)
    print(f"阶段 2 完成! 新增修复: {fixed_count} 个 | 纯文档保留: {skip_count} 个 | 顽固样本: {error_count} 个")
    print("-" * 40)