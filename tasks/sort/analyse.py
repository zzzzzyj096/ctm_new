import numpy as np
import torch
import random
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from models.ctm_sort import ContinuousThoughtMachineSORT
from models.lstm import LSTMBaseline
from models.ff import FFBaseline
from models.simpleEIRNN_sort import NetSORT
from models.simpleRNN_sort import SimpleNetSORT
from data.custom_datasets import SortDataset
from tasks.sort.utils import decode_predictions
import argparse

def parse_args():
    """Parses command-line arguments."""
    # Note: Original had two ArgumentParser instances, using the second one.
    parser = argparse.ArgumentParser(description='Sorting Task Analyzer')
    parser.add_argument('--model_type', type=str, default='ctm', choices=['ctm', 'lstm', 'ff', 'eirnn', 'simplernn'],
                        help='Model type to analysis.')
    parser.add_argument('--device', type=int, nargs='+', default=[-1], help="GPU device index or -1 for CPU")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/imagenet/ctm_imagenet.pt',
                        help="Path to ATM checkpoint")
    parser.add_argument('--output_dir', type=str, default='tasks/image_classification/analysis/outputs/imagenet_viz',
                        help="Directory for visualization outputs")
    parser.add_argument('--data_indices', type=int, nargs='+', default=[],
                        help="Use specific indices in validation data for demos, otherwise random.")

    return parser.parse_args()


# 设置全局种子的函数
def set_global_seed(seed):
    """设置所有随机库的全局种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FixedSortDataset(SortDataset):
    """
    固定种子的排序数据集
    在初始化时预生成所有数据，确保每次加载相同
    """

    def __init__(self, num_samples, sequence_length, seed=42):
        """
        参数:
        num_samples: 样本数量
        sequence_length: 序列长度
        seed: 随机种子
        """
        # 设置种子
        set_global_seed(seed)

        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.data = []

        # 预生成所有数据
        for _ in range(num_samples):
            data = torch.zeros(sequence_length).normal_()
            ordering = torch.argsort(data)
            self.data.append((data, ordering))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        inputs, ordering = self.data[idx]
        return (inputs), (ordering)

class SortingAnalyzer:
    def __init__(self, model, model_type, model_name="Model", device='cpu', seed=42, dataset=None, model_args=None):
        """
        排序任务分析器 - 直接使用已加载模型和内置解码函数

        参数:
        model: 已加载的模型实例
        model_name (str): 模型名称（用于图表标题）
        device: 计算设备（如'cuda:0'或'cpu'）
        """
        self.seed = seed
        set_global_seed(seed)
        self.model = model
        self.model_type = model_type
        self.model.to(device)
        self.model_name = model_name
        self.model_args = model_args
        self.device = device
        self.dataset = dataset  # 保存数据集
        self.clear_data()
        self.model.eval()

    def clear_data(self):
        """重置所有存储的数据"""
        self.input_sequences = []  # 输入序列
        self.output_sequences = []  # 输出序列
        self.wait_times = []  # 每个位置的等待时间
        self.sample_names = []  # 样本标识符
        self.raw_predictions = []  # 存储原始预测数据供分析使用

    def run_inference(self, input_sequence):
        input_tensor = torch.tensor(input_sequence, dtype=torch.float32, device=self.device)

        if self.model_type in ['eirnn', 'simplernn']:
            # EIRNN期望输入形状为 [B, sequence_length]
            if input_tensor.dim() != 1:
                input_tensor = input_tensor.flatten()
            input_tensor = input_tensor.unsqueeze(0)  # [sequence_length] -> [1, sequence_length]

        with torch.no_grad():
            model_output = self.model(input_tensor)

            # 处理不同模型的返回格式
            if isinstance(model_output, tuple):
                predictions = model_output[0]  # 取第一个元素（predictions）
                certainties = model_output[1] if len(model_output) > 1 else None
            else:
                predictions = model_output
                certainties = None

        self.raw_predictions.append(predictions)

        # 使用原始的decode_predictions函数
        decoded_sequences, wait_times_all = decode_predictions(predictions, predictions.shape[1] - 1, return_wait_times=True)
        output_sequence = decoded_sequences[0][:len(input_sequence)]
        wait_times = wait_times_all[0]  # 取第一个样本的等待时间

        return output_sequence, wait_times

    def add_sample(self, input_sequence, sample_name=None):
        if sample_name is None:
            sample_name = f"Sample {len(self.input_sequences) + 1}"

            # 运行推理
        output_sequence, wait_times = self.run_inference(input_sequence)

        # 存储结果 - 处理GPU张量
        self.input_sequences.append(np.array(input_sequence))

        # 检查是否为PyTorch张量并移动到CPU
        if hasattr(output_sequence, 'cpu'):
            self.output_sequences.append(output_sequence.cpu().numpy())
        else:
            self.output_sequences.append(np.array(output_sequence))

        if hasattr(wait_times, 'cpu'):
            self.wait_times.append(wait_times.cpu().numpy())
        else:
            self.wait_times.append(np.array(wait_times))

        self.sample_names.append(sample_name)

    def add_multiple_samples(self, sample_indices=None, sample_names=None):
        """
        批量添加样本并运行推理

        参数:
        input_sequences (list): 输入序列列表
        sample_names (list): 样本标识符列表（可选）
        """
        if self.dataset is None:
            raise ValueError("Dataset not provided for sample loading")

        if sample_indices is None:
            sample_indices = range(len(self.dataset))

        if sample_names is None:
            sample_names = [f"Sample {i}" for i in sample_indices]

        for idx, name in zip(tqdm(sample_indices, desc="Processing samples"), sample_names):
            inputs, _ = self.dataset[idx]  # 获取输入序列
            self.add_sample(inputs.numpy(), name)

    def compute_data_deltas(self):
        """计算所有样本的数据变化量"""
        data_deltas_list = []

        for input_seq in self.input_sequences:
            deltas = np.zeros_like(input_seq)
            deltas[1:] = np.abs(np.diff(input_seq))
            data_deltas_list.append(deltas)

        return data_deltas_list

    def analyze_position_dependency(self, ax=None):
        """分析等待时间与序列位置的关系"""
        # 获取所有样本的等待时间
        all_wait_times = []
        for wait_times in self.wait_times:
            # 添加每个位置的等待时间
            for i, wt in enumerate(wait_times):
                if i < len(wait_times):  # 确保位置索引有效
                    all_wait_times.append((i, wt))

        # 如果没有数据，返回空结果
        if not all_wait_times:
            return np.array([]), np.array([])

        # 转换为数组：位置索引和对应的等待时间
        positions, wait_values = zip(*all_wait_times)
        positions = np.array(positions)
        wait_values = np.array(wait_values)

        # 计算每个位置的平均等待时间
        unique_positions = np.unique(positions)
        position_means = [np.mean(wait_values[positions == pos]) for pos in unique_positions]
        position_stderr = [np.std(wait_values[positions == pos]) / np.sqrt(np.sum(positions == pos))
                           for pos in unique_positions]

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        # 绘制结果
        ax.errorbar(unique_positions, position_means, yerr=position_stderr,
                    fmt='o-', color='#1f77b4', lw=2, label='Mean Wait Time')

        ax.set_title(f'{self.model_name}: Wait Time vs. Sequence Position', fontsize=14)
        ax.set_xlabel('Sequence Position', fontsize=12)
        ax.set_ylabel('Wait Time (steps)', fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend()

        return position_means, position_stderr

    def analyze_delta_dependency(self, ax=None):
        """分析等待时间与数据变化量的关系"""
        # 计算数据变化量
        data_deltas_list = self.compute_data_deltas()

        # 准备数据
        flat_wait = []
        flat_deltas = []

        for wait_times, deltas in zip(self.wait_times, data_deltas_list):
            # 仅在有数据的位置进行比较
            min_len = min(len(wait_times), len(deltas))
            if min_len > 0:
                # 跳过第一个位置（delta=0）
                start_index = 1
                # 确保位置索引有效
                valid_indices = range(start_index, min_len)
                flat_wait.extend(wait_times[i] for i in valid_indices)
                flat_deltas.extend(deltas[i] for i in valid_indices)

        if not flat_wait or not flat_deltas:
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center')
            return 0.0

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        # 计算相关系数
        correlation = np.corrcoef(flat_wait, flat_deltas)[0, 1]

        # 绘制回归关系
        sns.regplot(x=flat_deltas, y=flat_wait,
                    scatter_kws={'alpha': 0.3, 'color': '#2ca02c', 's': 15},
                    line_kws={'lw': 2, 'color': '#d62728'},
                    ax=ax)

        ax.set_title(f'{self.model_name}: Wait Time vs. Data Delta (ρ = {correlation:.2f})', fontsize=14)
        ax.set_xlabel('|Current Value - Previous Value|', fontsize=12)
        ax.set_ylabel('Wait Time (steps)', fontsize=12)
        ax.grid(alpha=0.3)

        return correlation

    def visualize_single_process(self, sample_idx=0, ax=None):
        """可视化单个排序过程"""
        # 获取样本数据
        input_seq = self.input_sequences[sample_idx]
        wait_times = self.wait_times[sample_idx]
        sample_name = self.sample_names[sample_idx]

        # 收集所有样本的等待时间进行统计分析
        all_wait_times = []
        for wt in self.wait_times:
            all_wait_times.extend(wt)

        if not all_wait_times:
            if ax is None:
                fig, ax = plt.subplots(figsize=(12, 7))
            ax.text(0.5, 0.5, 'No wait time data available', ha='center', va='center')
            return ax

        # 计算平均等待时间
        avg_wait = np.mean(all_wait_times)

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 7))

        # 创建彩虹颜色映射表示原始序列位置
        cmap = plt.get_cmap('rainbow')
        position_colors = cmap(np.linspace(0, 1, len(input_seq)))

        # 创建红绿偏差颜色映射
        bias_cmap = LinearSegmentedColormap.from_list('rg', ['green', 'white', 'red'])

        # 绘制原始序列值（带彩虹颜色）
        for i, val in enumerate(input_seq):
            # 仅在有等待时间数据的位置绘制柱状图
            if i < len(wait_times):
                bar_height = val
                ax.bar(i, bar_height, color=position_colors[i], alpha=0.7, width=0.8)

                # 计算当前等待时间与平均值的偏差
                deviation = wait_times[i] - avg_wait
                # 归一化到[-1,1]范围
                deviation_norm = max(-1.0, min(1.0, deviation / (2 * avg_wait)))
                # 转换为[0,1]用于颜色映射
                color_value = (deviation_norm + 1) / 2

                # 绘制等待时间偏差条
                ax.plot([i - 0.4, i + 0.4],
                        [bar_height + 0.1, bar_height + 0.1],
                        lw=8,
                        color=bias_cmap(color_value),
                        solid_capstyle='round')

                # 添加等待时间文本
                ax.text(i, bar_height + 0.15, f'{wait_times[i]:.1f}',
                        ha='center', fontsize=9, fontweight='bold')

                # 添加数据变化量文本（如果适用）
                if i > 0:
                    delta = abs(input_seq[i] - input_seq[i - 1])
                    ax.text(i - 0.5, bar_height - 0.15, f'Δ={delta:.2f}',
                            ha='center', fontsize=8, color='dimgrey')

        # 设置坐标轴和标签
        ax.set_title(f'{self.model_name}: Sorting Process ({sample_name})', fontsize=16)
        ax.set_xlabel('Sequence Position', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticks(range(len(input_seq)))

        # 添加图例
        ax.plot([], [], lw=8, color='red', label=f'Longer than avg ({avg_wait:.1f})')
        ax.plot([], [], lw=8, color='green', label=f'Shorter than avg ({avg_wait:.1f})')
        ax.legend(loc='upper right')

        return ax

    def generate_report(self, sample_idx=0, save_path=None,seed=None):
        """生成完整分析报告"""
        if seed is not None:
            set_global_seed(seed)
        fig = plt.figure(figsize=(16, 12), dpi=100)
        gs = fig.add_gridspec(2, 2)

        # 位置依赖分析
        ax1 = fig.add_subplot(gs[0, 0])
        self.analyze_position_dependency(ax1)

        # 数据变化依赖分析
        ax2 = fig.add_subplot(gs[0, 1])
        self.analyze_delta_dependency(ax2)

        # 单样本可视化
        ax3 = fig.add_subplot(gs[1, :])
        self.visualize_single_process(sample_idx, ax3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"报告已保存至: {save_path}")
        else:
            plt.show()

        return fig


if __name__ == '__main__':
    SEED = 42
    # --- 解析命令行参数 ---
    args = parse_args()

    # --- 设置设备 ---
    if args.device[0] != -1 and torch.cuda.is_available():
        device = f'cuda:{args.device[0]}'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    # --- 加载模型检查点 ---
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # 从检查点获取模型参数
    model_args = checkpoint.get('args', None)
    if model_args is None:
        raise ValueError("Checkpoint does not contain model arguments")

        # 使用检查点中的N_to_sort参数创建数据集
    N_to_sort = model_args.N_to_sort  # 从检查点获取序列长度
    validation_dataset = FixedSortDataset(num_samples=1000, sequence_length=N_to_sort, seed=SEED)
    prediction_reshaper = [-1]  # Problem specific

    # 调试用参数验证代码
    d_input_calculated = model_args.out_dims - 1
    heads = model_args.heads
    # 确保d_input能被heads整除
    if d_input_calculated % heads != 0:
        print(f"Warning: d_input ({d_input_calculated}) not divisible by heads ({heads})")
        # 可以选择调整参数或抛出错误

    # --- 初始化模型 ---
    model = None
    if args.model_type == 'ctm':
        print("Instantiating CTM model...")
        model = ContinuousThoughtMachineSORT(
            iterations=model_args.iterations,
            d_model=model_args.d_model,
            d_input=model_args.out_dims - 1,
            heads=model_args.heads,
            n_synch_out=model_args.n_synch_out,
            n_synch_action=model_args.n_synch_action,
            synapse_depth=model_args.synapse_depth,
            memory_length=model_args.memory_length,
            deep_nlms=model_args.deep_memory,
            memory_hidden_dims=model_args.memory_hidden_dims,
            do_layernorm_nlm=model_args.do_normalisation,
            backbone_type=model_args.backbone_type,
            positional_embedding_type=model_args.positional_embedding_type,
            out_dims=model_args.out_dims,
            prediction_reshaper=[-1],
            dropout=0,
            neuron_select_type=model_args.neuron_select_type,
            n_random_pairing_self=model_args.n_random_pairing_self,
        )
    elif args.model_type == 'eirnn':
        print("Instantiating eirnn model...")
        if args.model_type == 'eirnn':
            model = NetSORT(
                iterations=model_args.iterations,  # 25
                d_model=model_args.d_model,
                d_input=model_args.out_dims - 1,  # 6
                heads=model_args.heads,  # 3
                backbone_type='none',  # 关键：sort任务使用none
                num_layers=model_args.num_layers,
                positional_embedding_type=model_args.positional_embedding_type,
                out_dims=model_args.out_dims,  # 7
                prediction_reshaper=[-1],
                dropout=0,
            )
    elif args.model_type == 'simplernn':
        print("Instantiating simplernn model...")
        model = SimpleNetSORT(
            iterations=model_args.iterations,
            d_model=model_args.d_model,
            d_input=model_args.out_dims - 1,
            heads=model_args.heads,
            backbone_type='none',
            positional_embedding_type=model_args.positional_embedding_type,
            out_dims=model_args.out_dims,
            dropout=model_args.dropout,
            prediction_reshaper=prediction_reshaper,
            num_layers=model_args.num_layers,
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

        # 调试
        print(f"model_args.N_to_sort: {model_args.N_to_sort}")
        print(f"model_args.out_dims: {model_args.out_dims}")
        print(f"Calculated d_input: {model_args.out_dims - 1}")
        print(f"Input tensor shape: {input_tensor.shape}")

    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    #
    print(f"Model input layer weight shape: {model.lstm.input_layer.weight.shape}")
    model.to(device)
    model.eval()

    print(f"{args.model_type} model loaded successfully")

    # 创建分析器
    model_analyzer = SortingAnalyzer(
        model,
        model_name=args.model_type.upper(),
        device=device,
        dataset=validation_dataset,
        seed=SEED,
        model_type=args.model_type,
        model_args=model_args
    )

    # 批量添加样本并运行推理
    model_analyzer.add_multiple_samples(sample_indices=range(100))

    os.makedirs(args.output_dir, exist_ok=True)

    # 生成分析报告
    report_path = os.path.join(args.output_dir, "sorting_report.png")
    model_analyzer.generate_report(sample_idx=0, save_path=report_path)
    print(f"Analysis report saved to: {report_path}")

