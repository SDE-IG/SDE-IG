import os
import csv
import json
import torch
import argparse
import networkx as nx
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Data

# 导入同级目录下的 CLAP 官方代码
from clap_modeling import AsmTokenizer, AsmEncoder

# ================= 通用配置 =================
DATASETS = {
    "APTMalware": {
        "input_dir": "/home/zhufengzhou/APTMalware_FCG_Graphs",
        "output_dir": "/home/zhufengzhou/APTMalware_PyG_Data"
    },
    "BIG2015": {
        "input_dir": "/home/zhufengzhou/BIG2015_FCG_Graphs",
        "output_dir": "/home/zhufengzhou/BIG2015_PyG_Data",
        "label_csv": "/home/zhufengzhou/malware-classification/trainLabels.csv"
    },
    "Dike": {
        "input_dir": "/home/zhufengzhou/Dike_FCG_Graphs",
        "output_dir": "/home/zhufengzhou/Dike_PyG_Data",
        "label_csv": "/home/zhufengzhou/DikeDataset/labels/malware.csv"
    },
    # 新增 ZfzMalware 配置
    "ZfzMalware": {
        "input_dir": "/home/zhufengzhou/ZfzMalware_FCG_Graphs",
        "output_dir": "/home/zhufengzhou/ZfzMalware_PyG_Data"
    }
}

# 指向你刚才放置配置和模型权重的文件夹
MODEL_WEIGHTS_DIR = "/home/zhufengzhou/malware_gene/clap_weights"

# 显卡配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64  # 如果爆显存可以调小至 8 或 4

# ============================================

def parse_asm_text_to_dict(asm_text):
    """
    将 `1: push rbp` 的长文本还原为 AsmTokenizer 要求的字典格式。
    返回示例: {"1": "push rbp", "2": "mov rbp rsp"}
    """
    func_dict = {}
    for line in asm_text.strip().split('\n'):
        if ': ' in line:
            # 仅按第一个 ': ' 切分，防止汇编内部自带冒号干扰
            key, val = line.split(': ', 1)
            func_dict[key.strip()] = val.strip()
    return func_dict


def load_embedding_model():
    print(f"[*] 正在从 {MODEL_WEIGHTS_DIR} 加载 CLAP 模型至 {DEVICE}...")

    # 使用 AsmTokenizer 和 AsmEncoder 加载本地权重
    tokenizer = AsmTokenizer.from_pretrained(MODEL_WEIGHTS_DIR)
    model = AsmEncoder.from_pretrained(MODEL_WEIGHTS_DIR).to(DEVICE)

    model.eval()  # 关闭 Dropout，保证推理一致性
    return tokenizer, model


@torch.no_grad()
def get_node_embeddings(texts, tokenizer, model):
    """
    批量调用 CLAP 获取节点的 Assembly Embedding
    """
    if not texts:
        return torch.empty((0, 768))  # CLAP RoFormer 默认 hidden_size 是 768

    # 1. 将文本列表转换为 AsmTokenizer 要求的 Dict 列表
    dict_functions = [parse_asm_text_to_dict(text) for text in texts]

    all_embeddings = []

    # 2. 批量推理
    for i in range(0, len(dict_functions), BATCH_SIZE):
        batch_funcs = dict_functions[i: i + BATCH_SIZE]

        # AsmTokenizer 已经重写了 __call__，它会自动处理 padding 和 truncating
        inputs = tokenizer(batch_funcs, return_tensors="pt")

        # 强制将 Tokenizer 输出的 tensor 转换为 torch.long 类型
        inputs = {k: v.to(DEVICE, dtype=torch.long, non_blocking=True) for k, v in inputs.items()}

        # CLAP 的 AsmEncoder forward() 函数直接返回了 normalization 后的 asm_embedding
        asm_embedding = model(**inputs)

        # 立即转移到 CPU 防止 GPU 显存溢出
        all_embeddings.append(asm_embedding.cpu())

    return torch.cat(all_embeddings, dim=0)


def load_big2015_labels(csv_path):
    """
    读取 BIG2015 的 label csv 文件，返回字典 { hash_id: label_int }
    """
    label_dict = {}
    if not os.path.exists(csv_path):
        print(f"[-] 警告：找不到标签文件 {csv_path}")
        return label_dict

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头 (Id, Class)
        for row in reader:
            if len(row) >= 2:
                sample_id = row[0].strip('"')
                label = int(row[1])
                label_dict[sample_id] = label
    
    print(f"[*] 成功加载了 {len(label_dict)} 个 BIG2015 训练标签。")
    return label_dict


def load_dike_labels(csv_path):
    """
    读取 Dike 软标签，将各类别中得分最高的作为硬标签 (0~8 对应 generic~downloader)
    """
    label_dict = {}
    if not os.path.exists(csv_path):
        print(f"[-] 警告：找不到标签文件 {csv_path}")
        return label_dict

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        # header 示例: type,hash,malice,generic,trojan,ransomware,worm,backdoor,spyware,rootkit,encrypter,downloader
        
        for row in reader:
            if len(row) >= 12:
                sample_hash = row[1].strip()
                try:
                    # 提取具体的恶意软件类别分数 (索引 3 到最后)
                    scores = [float(x) for x in row[3:]]
                    # 获取最高分所在的索引作为硬标签
                    hard_label = scores.index(max(scores))
                    label_dict[sample_hash] = hard_label
                except ValueError:
                    continue
    
    print(f"[*] 成功加载了 {len(label_dict)} 个 Dike 训练标签 (已转为硬标签)。")
    return label_dict


def process_single_json(json_path, output_path, tokenizer, model, label=None):
    """ 处理单图 JSON 并保存为 .pt 文件，支持传入 label """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 抑制 NetworkX 3.6 版本的警告，显式传入 edges="links"
        G = nx.node_link_graph(data, edges="links") 
        node_mapping = {node_str: i for i, node_str in enumerate(G.nodes())}

        # 提取当前图的所有节点汇编
        node_texts = []
        for node_str in G.nodes():
            text = G.nodes[node_str].get("assembly", "")
            node_texts.append(text)

        # 调用 CLAP 模型获取张量，尺寸为 [num_nodes, 768]
        x = get_node_embeddings(node_texts, tokenizer, model)

        edges = []
        for src, tgt in G.edges():
            if src in node_mapping and tgt in node_mapping:
                edges.append([node_mapping[src], node_mapping[tgt]])

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # 构建 PyG Data 对象，如果传入了 label 就加上 y 属性
        if label is not None:
            y = torch.tensor([label], dtype=torch.long)
            pyg_data = Data(x=x, edge_index=edge_index, y=y)
        else:
            pyg_data = Data(x=x, edge_index=edge_index)

        torch.save(pyg_data, output_path)
        return True

    except Exception as e:
        print(f"\n[-] 处理文件 {Path(json_path).name} 失败: {str(e)}")
        return False


def process_dataset(dataset_name, tokenizer, model):
    input_dir = DATASETS[dataset_name]["input_dir"]
    output_dir = DATASETS[dataset_name]["output_dir"]

    print(f"\n>>> 🚀 开始使用 CLAP 提取 {dataset_name} 特征并转化为 PyG 格式")

    if not os.path.exists(input_dir):
        print(f"[-] 找不到输入目录: {input_dir}")
        return

    # 预加载标签字典或初始化标签映射器
    label_dict = {}
    zfz_label_map = {}  # 用于自动记录 ZfzMalware 的家族名称到数字ID的映射

    if dataset_name == "BIG2015":
        label_dict = load_big2015_labels(DATASETS["BIG2015"]["label_csv"])
    elif dataset_name == "Dike":
        label_dict = load_dike_labels(DATASETS["Dike"]["label_csv"])

    json_files = list(Path(input_dir).rglob("*_fcg.json"))
    if not json_files:
        print("[-] 没有找到可用的 JSON 文件")
        return

    success_count = 0

    for json_path in tqdm(json_files, desc=f"Encoding {dataset_name}"):
        rel_path = json_path.relative_to(input_dir)
        output_path = Path(output_dir) / rel_path.with_suffix('.pt')

        # 保持 train/test 等原有目录结构
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            continue

        # ====== 确定 Label ======
        current_label = None
        sample_hash = json_path.name.replace("_fcg.json", "")

        if dataset_name == "BIG2015":
            if "train" in json_path.parts:
                current_label = label_dict.get(sample_hash, None)
                if current_label is None:
                    print(f"\n[-] 警告: BIG2015 训练集样本 {sample_hash} 在 CSV 中未找到对应标签。")
            elif "test" in json_path.parts:
                current_label = -1 
        elif dataset_name == "Dike":
            clean_hash = sample_hash.replace(".exe", "")
            current_label = label_dict.get(clean_hash, None)
            if current_label is None:
                print(f"\n[-] 警告: Dike 样本 {sample_hash} 在 CSV 中未找到对应标签。")
        
        # 新增的 ZfzMalware 标签逻辑
        elif dataset_name == "ZfzMalware":
            # 路径示例: /home/.../ZfzMalware_FCG_Graphs/<family_name>/<subset>/xxx_fcg.json
            # json_path.parts[-3] 就是家族名称
            family_name = json_path.parts[-3]
            
            if family_name not in zfz_label_map:
                zfz_label_map[family_name] = len(zfz_label_map)
                
            current_label = zfz_label_map[family_name]
        # ========================

        if process_single_json(str(json_path), str(output_path), tokenizer, model, label=current_label):
            success_count += 1

    print(f"[*] {dataset_name} 提取完毕！生成了 {success_count} 个 PyG Data 对象。")

    # 处理完成后，保存 ZfzMalware 的标签映射字典
    if dataset_name == "ZfzMalware" and zfz_label_map:
        map_path = os.path.join(output_dir, "family_label_map.json")
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(zfz_label_map, f, indent=4, ensure_ascii=False)
        print(f"[*] ZfzMalware 家族标签映射表已保存至: {map_path} (请妥善保管，训练时需用)")


def main():
    parser = argparse.ArgumentParser(description="PyG 离线图特征提取引擎 (基于 CLAP)")
    # 将 ZfzMalware 设为可选项和默认项
    parser.add_argument("--dataset", type=str, choices=["APTMalware", "BIG2015", "Dike", "ZfzMalware", "ALL"], default="ZfzMalware")
    args = parser.parse_args()

    tokenizer, model = load_embedding_model()

    datasets_to_process = []
    if args.dataset == "ALL":
        datasets_to_process = ["APTMalware", "BIG2015", "Dike", "ZfzMalware"]
    else:
        datasets_to_process = [args.dataset]

    for ds in datasets_to_process:
        process_dataset(ds, tokenizer, model)


if __name__ == "__main__":
    main()