import numpy as np
import torch
import random
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from data.custom_datasets import SortDataset
from utils.losses import sort_loss
from tasks.sort.utils import compute_ctc_accuracy, decode_predictions
from tasks.sort.utils import prepare_model
import argparse
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', type=str, default='eirnn',
                        choices=['ctm', 'lstm', 'simplernn', 'eirnn', 'ctm_simple'], help='Type of model to use.')
    # 从训练脚本中复制必要的参数
    parser.add_argument('--d_model', type=int, default=64, help='Dimension of the model.')
    parser.add_argument('--d_input', type=int, default=128, help='Dimension of the input.')
    parser.add_argument('--heads', type=int, default=3, help='Number of attention heads.')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM/RNN layers')
    parser.add_argument('--iterations', type=int, default=25, help='Number of internal ticks.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--positional_embedding_type', type=str, default='none',
                        help='Type of positional embedding.', choices=['none',
                                                                       'learnable-fourier',
                                                                       'multi-learnable-fourier',
                                                                       'custom-rotational'])
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--batch_size_test', type=int, default=32, help='Batch size for testing.')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate for the model.')
    parser.add_argument('--training_iterations', type=int, default=100001, help='Number of training iterations.')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='Number of warmup steps.')
    parser.add_argument('--use_scheduler', action=argparse.BooleanOptionalAction, default=True,
                        help='Use a learning rate scheduler.')
    parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['multistep', 'cosine'],
                        help='Type of learning rate scheduler.')
    parser.add_argument('--milestones', type=int, default=[8000, 15000, 20000], nargs='+',
                        help='Learning rate scheduler milestones.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate scheduler gamma for multistep.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay factor.')
    parser.add_argument('--weight_decay_exclusion_list', type=str, nargs='+', default=[],
                        help='List to exclude from weight decay. Typically good: bn, ln, bias, start')
    parser.add_argument('--gradient_clipping', type=float, default=-1,
                        help='Gradient quantile clipping value (-1 to disable).')
    parser.add_argument('--use_amp', action=argparse.BooleanOptionalAction, default=False, help='AMP autocast.')
    parser.add_argument('--do_compile', action=argparse.BooleanOptionalAction, default=False,
                        help='Try to compile the synapses, backbone, and nlms.')
    parser.add_argument('--N_to_sort', type=int, default=6, help='N numbers to sort.')
    parser.add_argument('--save_every', type=int, default=1000, help='Save checkpoints every this many iterations.')
    parser.add_argument('--seed', type=int, default=412, help='Random seed.')
    parser.add_argument('--reload', action=argparse.BooleanOptionalAction, default=False, help='Reload from disk?')
    parser.add_argument('--reload_model_only', action=argparse.BooleanOptionalAction, default=False,
                        help='Reload only the model from disk?')
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--track_every', type=int, default=1000, help='Track metrics every this many iterations.')
    parser.add_argument('--n_test_batches', type=int, default=2,
                        help='How many minibatches to approx metrics. Set to -1 for full eval')
    parser.add_argument('--output_dir', type=str, default='./comparison_results',
                        help='Directory to save comparison results')
    parser.add_argument('--device', type=int, nargs='+', default=[-1],
                        help='List of GPU(s) to use. Set to -1 to use CPU.')

    args = parser.parse_args()
    return args

# 设置全局种子的函数
def set_global_seed(seed):
    """设置所有随机库的全局种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compare_args(saved_args, current_args):
    """比较两个参数对象是否相同"""
    # 比较关键参数
    key_params = [
        'd_model', 'd_input', 'n_synch_out', 'n_synch_action',
        'heads', 'synapse_depth', 'memory_length', 'deep_memory'
    ]
    for key in key_params:
        if getattr(saved_args, key, None) != getattr(current_args, key, None):
            return False
    return True

#class FixedSortDataset(SortDataset):
    #"""
    #固定种子的排序数据集
    #在初始化时预生成所有数据，确保每次加载相同
    #"""

    #def __init__(self, num_samples, sequence_length, seed=42):
        #"""
        #参数:
        #num_samples: 样本数量
        #sequence_length: 序列长度
        #seed: 随机种子
        #"""
        # 设置种子
        #set_global_seed(seed)

        #self.num_samples = num_samples
        #self.sequence_length = sequence_length
        #self.data = []

        # 预生成所有数据
        #for _ in range(num_samples):
            #data = torch.zeros(sequence_length).normal_()
            #ordering = torch.argsort(data)
            #self.data.append((data, ordering))

    #def __len__(self):
        #return self.num_samples

    #def __getitem__(self, idx):
        #inputs, ordering = self.data[idx]
        #return (inputs), (ordering)

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
        #调试
        print(f"Raw predictions shape: {predictions.shape}")
        print(f"Input sequence length: {len(input_sequence)}")
        # 调试添加对原始预测路径的分析
        # 详细分析原始预测
        probs = F.softmax(predictions, dim=-1)
        best_path = torch.argmax(probs[0], dim=-1)
        print(f"Best path (raw): {best_path}")
        print(f"Unique values in best path: {torch.unique(best_path)}")
        # 分析解码前后
        #decoded_sequences, wait_times_all = decode_predictions(predictions, predictions.shape[1] - 1,return_wait_times=True)
        #print(f"Full decoded sequence: {decoded_sequences[0]}")

        # 使用原始的decode_predictions函数
        decoded_sequences, wait_times_all = decode_predictions(predictions, predictions.shape[1] - 1, return_wait_times=True)
        print(f"Decoded sequence length: {len(decoded_sequences[0])}")
        print(f"Full decoded sequence: {decoded_sequences[0]}")
        #output_sequence = decoded_sequences[0][:len(input_sequence)]   #这句应该要修改为类似于训练时截短版
        target_length = len(input_sequence)
        output_sequence = decoded_sequences[0][:target_length]  # 截断到目标长度
        #print(f"Truncated to input length: {decoded_sequences[0][:len(input_sequence)]}")

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

        ax.set_title(f'{self.model_name}: Wait Time vs. Data Delta ', fontsize=14)
        ax.set_xlabel('|Current Value - Previous Value|', fontsize=12)
        ax.set_ylabel('Wait Time (steps)', fontsize=12)
        ax.grid(alpha=0.3)

        return correlation

    def visualize_single_process(self, sample_idx=0, ax=None):
        """可视化单个排序过程"""
        # 获取样本数据
        input_seq = self.input_sequences[sample_idx]
        output_indices = self.output_sequences[sample_idx]
        wait_times = self.wait_times[sample_idx]
        sample_name = self.sample_names[sample_idx]

        # 将索引转换为实际的排序值
        if len(output_indices) > 0:
            output_seq = input_seq[output_indices]
        else:
            output_seq = []

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

        # 绘制排序后的序列值
        for i in range(len(output_indices)):
            if i < len(wait_times):
                # 使用排序后的实际值作为柱状图高度
                bar_height = output_seq[i]

                # 根据原始位置选择颜色（保持颜色与原始位置的对应关系）
                original_position = output_indices[i]
                bar_color = position_colors[original_position]

                ax.bar(i, bar_height, color=bar_color, alpha=0.7, width=0.8)

                # 计算当前等待时间与平均值的偏差
                deviation = wait_times[i] - avg_wait
                deviation_norm = max(-1.0, min(1.0, deviation / (2 * avg_wait)))
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

                # 添加原始位置标签
                #ax.text(i, -0.2, f'原始位置:{original_position}',
                        #ha='center', fontsize=8, color='blue', alpha=0.7)

                # 设置坐标轴和标签
        ax.set_title(f'{self.model_name}: Sorted Output Sequence ({sample_name})', fontsize=16)
        ax.set_xlabel('Sorted Position (排序后位置)', fontsize=12)
        ax.set_ylabel('Value (数值)', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticks(range(len(output_indices)))

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


def main():
    SEED = 42
    # --- 解析命令行参数 ---
    args = parse_args()

    # --- 设置设备 ---
    if args.device[0] != -1 and torch.cuda.is_available():
        device = f'cuda:{args.device[0]}'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    validation_data = SortDataset(args.N_to_sort)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=1)

    # --- 初始化模型 ---
    model = None
    checkpoint_path = os.path.join(args.log_dir, 'checkpoint.pt')

    # --- 加载模型检查点 ---
    print(f"Loading checkpoint: {checkpoint_path}")
    # 加载检查点
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )

        # 使用检查点中保存的参数构建模型
        if 'args' in checkpoint:
            saved_args = checkpoint['args']

            if hasattr(saved_args, 'out_dims'):
                saved_args.d_input = saved_args.out_dims - 1

            # print("Saved args:", saved_args)
            # print("Current args:", args)

            if model is None or not compare_args(saved_args, args):
                print(f"Building model with saved parameters")
                # 注意：prepare_model 需要 prediction_reshaper 作为第一个参数
                prediction_reshaper = [-1]  # 排序任务的 reshaper
                model = prepare_model(prediction_reshaper, saved_args, device)
        else:
            if model is None:
                prediction_reshaper = [-1]
                model = prepare_model(prediction_reshaper, args, device)

                # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print(f"Successfully loaded modelei from {checkpoint_path}")

    except FileNotFoundError:
        print(f"Checkpoint file not found: {checkpoint_path}")
        return
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        return
    finally:
        # 清理内存（如果checkpoint已加载）
        if 'checkpoint' in locals():
            del checkpoint
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    model.eval()

    print(f"{args.model_type} model loaded successfully")

    # 创建分析器
    model_analyzer = SortingAnalyzer(
        model,
        model_name=args.model_type.upper(),
        device=device,
        dataset=validation_data,
        seed=SEED,
        model_type=args.model_type,
        model_args=saved_args
    )

    # 批量添加样本并运行推理
    model_analyzer.add_multiple_samples(sample_indices=range(100))

    os.makedirs(args.output_dir, exist_ok=True)

    # 生成分析报告
    report_path = os.path.join(args.output_dir, "sorting_report.png")
    model_analyzer.generate_report(sample_idx=0, save_path=report_path)
    print(f"Analysis report saved to: {report_path}")

    print("Analysis completed!")

if __name__ == '__main__':
    main()