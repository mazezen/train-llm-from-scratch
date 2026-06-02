import os
import subprocess
import sys


def download_pretrain_data():
    """下载并解压预训练数据集（来自 ModelScope）"""
    dataset_id = "ddzhu123/seq-monkey"
    file_name = "mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2"
    local_dir = "./autodl-tmp/dataset/pretrain_data"
    extracted_file = os.path.join(local_dir, file_name.replace(".tar.bz2", ""))

    # 如果已存在，跳过下载
    if os.path.exists(extracted_file):
        print(f"[OK] 预训练数据已存在: {extracted_file}")
        return

    # 检查 modelscope 是否安装
    try:
        subprocess.run(["modelscope", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("正在安装 modelscope...")
        subprocess.run([sys.executable, "-m", "pip", "install", "modelscope"], check=True)

    # 下载
    os.makedirs(local_dir, exist_ok=True)
    print(f"正在下载 {dataset_id}/{file_name} ...")
    ret = os.system(
        f"modelscope download --dataset {dataset_id} {file_name} --local_dir {local_dir}"
    )
    if ret != 0:
        print(f"[错误] 下载失败，返回码 {ret}")
        sys.exit(1)

    # 解压到目标目录
    tar_path = os.path.join(local_dir, file_name)
    print(f"正在解压 {tar_path} 到 {local_dir} ...")
    ret = os.system(f"tar -xvf {tar_path} -C {local_dir}")
    if ret != 0:
        print(f"[错误] 解压失败，返回码 {ret}")
        sys.exit(1)

    print(f"[OK] 预训练数据集已准备好: {extracted_file}")


def download_sft_data():
    """下载 BelleGroup 指令微调数据集（来自 HuggingFace）"""
    local_dir = "./data/BelleGroup"
    if os.path.exists(os.path.join(local_dir, "train_3.5M_CN.json")):
        print(f"[OK] 微调数据已存在: {local_dir}")
        return

    os.makedirs(local_dir, exist_ok=True)
    print("正在下载 BelleGroup/train_3.5M_CN ...")
    ret = os.system(
        f"hf download --repo-type dataset BelleGroup/train_3.5M_CN --local-dir {local_dir}"
    )
    if ret != 0:
        print(f"[警告] HuggingFace 下载失败（可能未安装 huggingface-hub），返回码 {ret}")
    else:
        print(f"[OK] 微调数据集已下载到 {local_dir}")


if __name__ == "__main__":
    download_pretrain_data()
    download_sft_data()
