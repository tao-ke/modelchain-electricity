"""
电力价格预测系统 - 基于 PyTorch Forecasting TFT (Temporal Fusion Transformer) 架构

核心特点：
1. 原生支持多步直接预测 (无需单步递归)
2. 自动适配 GPU/CPU 环境
3. 采用分位数回归 (QuantileLoss) 准确预测极端电价的波动范围 (P10, P50, P90)
4. 将过去的 7 天 (672步) 编码，直接输出未来 1 天 (96步) 的预测结果
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# PyTorch 生态
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

# PyTorch Forecasting
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from pytorch_forecasting.data import GroupNormalizer

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei',"SimHei"]
plt.rcParams['axes.unicode_minus'] = False

# =========================================================
# 1. 数据加载与清理模块
# =========================================================

def _parse_single_station_file(path):
    try:
        df = pd.read_excel(path)
    except Exception as e:
        print(f"⚠ 读取文件失败: {path} -> {e}")
        return None

    cols = list(df.columns)
    lower_cols = [c.lower() for c in cols]
    
    if 'time' in lower_cols or 'timestamp' in lower_cols:
        time_col = cols[lower_cols.index('time')] if 'time' in lower_cols else cols[lower_cols.index('timestamp')]
        price_candidates = [c for c in cols if str(c).lower() in ('price', '电价', 'price(元)')]
        price_col = price_candidates[0] if price_candidates else cols[-1]

        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=[time_col])
        out = df[[time_col, price_col]].copy()
        out.columns = ['timestamp', 'price']
        out['timestamp'] = pd.to_datetime(out['timestamp'])
        out = out.set_index('timestamp').sort_index()
        return out

    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

    time_cols = df.columns[1:97]
    if len(time_cols) == 0:
        return None

    df_long = df.melt(id_vars=[date_col], value_vars=time_cols, var_name="timestep", value_name="price")
    df_long["timestep"] = df_long["timestep"].astype(str).str.strip()
    df_long["timestamp"] = pd.to_datetime(df_long[date_col].dt.strftime("%Y/%m/%d") + " " + df_long["timestep"], errors="coerce")

    out = df_long[["timestamp", "price"]].dropna(subset=["timestamp"]).copy()
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out = out.dropna(subset=["price"]).sort_values("timestamp").set_index("timestamp")
    return out

def load_station_electricity_data(station_name, base_dir='电价数据', years=(2025, 2026)):
    base = Path(base_dir)
    if not base.exists():
        return None

    found_files = []
    station_lower = station_name.lower()
    for yr in years:
        search_dir = base / str(yr)
        if not search_dir.exists():
            continue
        for f in search_dir.rglob('*.xls*'):
            if station_lower in f.stem.lower() or station_lower in f.name.lower():
                found_files.append(f)

    if not found_files:
        for f in base.rglob('*.xls*'):
            if station_lower in f.stem.lower() or station_lower in f.name.lower():
                found_files.append(f)

    if not found_files:
        print(f"未找到匹配文件: {station_name}")
        return None

    parts = []
    for f in found_files:
        part = _parse_single_station_file(f)
        if part is not None and not part.empty:
            parts.append(part)

    if not parts:
        return None

    combined = pd.concat(parts)
    combined = combined[~combined.index.duplicated(keep='first')].sort_index()

    # == [TFT 核心修复] ==: TFT 严格要求时间序列不允许有任何断层
    # 将时间频率重采样为每15分钟
    combined = combined.resample('15T').mean()
    combined['price'] = pd.to_numeric(combined['price'], errors='coerce')
    
    # 处理极端异常值防止引起 Inf
    combined['price'] = combined['price'].replace([np.inf, -np.inf], np.nan)
    # 双向填充与插值，确保绝对没有任何 NaN 
    combined['price'] = combined['price'].interpolate(method='linear').ffill().bfill()
    
    combined = combined.reset_index()
    
    return combined


# =========================================================
# 2. TFT 高级预测器核心模块
# =========================================================

class TFTElectricityPredictor:
    def __init__(self, station_name):
        self.station_name = station_name
        self.model = None
        self.trainer = None
        
        # 硬件自适应：检测可用硬件
        self.accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.devices = 1
        
        # 视窗配置 (15分钟频次)
        self.max_encoder_length = 672  # 回看历史: 7 天 * 96
        self.max_prediction_length = 96 # 预测未来: 1 天 * 96
        
        self.model_dir = f'./model_sets_tft/models_{self.station_name}'
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        self.last_daily_update = None
        self.last_weekly_retrain = None
        self.update_window_days = 60
        self.retrain_window_days = 180
        
    def generate_tft_features(self, df):
        """为 TFT 构建标准的特征数据与 time_idx"""
        print("构建 TFT 时间序列特征...")
        data = df.copy()
        
        # 1. 明确的序列ID (TFT 要求明确的 group_id)
        data['station'] = self.station_name
        
        # 2. 连续整数时间索引 
        data['time_idx'] = np.arange(len(data))
        
        # 3. 日历特征
        data['hour'] = data['timestamp'].dt.hour.astype(str)
        data['minute'] = data['timestamp'].dt.minute.astype(str)
        data['day_of_week'] = data['timestamp'].dt.dayofweek.astype(str)
        # 将分类变量转为 category 格式
        data[['hour', 'minute', 'day_of_week', 'station']] = data[['hour', 'minute', 'day_of_week', 'station']].astype('category')
        
        # 4. 连续时间的周期化映射
        data['time_of_day_num'] = data['timestamp'].dt.hour + data['timestamp'].dt.minute / 60.0
        data['hour_sin'] = np.sin(2 * np.pi * data['time_of_day_num'] / 24.0)
        data['hour_cos'] = np.cos(2 * np.pi * data['time_of_day_num'] / 24.0)

        #data['hour_int'] = data['timestamp'].dt.hour
        #data['minute_int'] = data['timestamp'].dt.minute

        # period_of_day (0..95)
        #data['period_of_day'] = data['hour_int'] * 4 + (data['minute_int'] // 15)

        # 基于过去的滞后 (全部用 shift 保证无数据泄露)
        #data['price_lag_1'] = data['price'].shift(1)
        #data['price_lag_2'] = data['price'].shift(2)
        #data['price_lag_96'] = data['price'].shift(96)

        # 差分/交互/滚动统计（基于 past 值）
        #data['price_diff_96'] = data['price_lag_1'] - data['price_lag_96']
        #data['price_roll_mean_4'] = data['price_lag_1'].rolling(window=4).mean()
        #data['price_roll_std_12'] = data['price_lag_1'].rolling(window=12).std()
        #rolling_mean_24 = data['price_lag_1'].rolling(window=24).mean()
        #rolling_std_24 = data['price_lag_1'].rolling(window=24).std().replace(0, np.nan)
        #data['bollinger_ratio'] = ((data['price_lag_1'] - rolling_mean_24) / (2 * rolling_std_24)).fillna(0)
        #data['price_lag_1_x_price_lag_96'] = data['price_lag_1'] * data['price_lag_96']

        # hour_target_encoded: per-hour expanding mean of past price (shifted by 1 to avoid泄露)
        #data['hour_target_encoded'] = (
        #        data.groupby('hour_int')['price']
        #                .transform(lambda x: x.shift(1).expanding().mean())
        #        )
        #data['hour_target_encoded'] = data['hour_target_encoded'].fillna(data['price'].mean())

        # 5. 异常值
        #data['price'] = data['price'].clip(lower=-10000, upper=10000)
        
        return data
        
    def create_dataset(self, data, is_train=True):
        """生成 PyTorch Forecasting 的 TimeSeriesDataSet"""
        
        # 当作预测集时，可以放宽 min_encoder_length 允许序列短端预测
        min_enc_length = 96 if not is_train else 96
        
        return TimeSeriesDataSet(
            data,
            time_idx="time_idx",
            target="price",
            group_ids=["station"],
            min_encoder_length=min_enc_length,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=self.max_prediction_length,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=["station"],
            time_varying_known_categoricals=["hour", "minute", "day_of_week"],
            #time_varying_known_reals=["time_idx", "hour_sin", "hour_cos", "period_of_day"],
            time_varying_known_reals=["time_idx", "hour_sin", "hour_cos"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=[
                        "price"
                #        "price_lag_1", "price_lag_2", "price_lag_96",
                #        "price_diff_96", "price_roll_mean_4", "price_roll_std_12",
                #        "bollinger_ratio", "price_lag_1_x_price_lag_96", "hour_target_encoded"
                ],
            # 将价格归一化
            target_normalizer=GroupNormalizer(groups=["station"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
    def _predict_quantiles_no_logger(self, dataloader):
        return self.model.predict(
                dataloader,
                mode="quantiles",
                trainer_kwargs={
                "logger": False,
                "enable_checkpointing": False,

                "enable_model_summary": False,
                "default_root_dir": str(Path.cwd() / "pl_root_predict"),
                },
        )
        
    def train_model(self, data, max_epochs=10, batch_size=32, learning_rate=0.01):
        """训练 TFT 模型"""
        print(f"\n[硬件报告] 正在使用 {self.accelerator.upper()} 进行运算。")
        
        # 1. 划分训练和验证集 (保留最后两周作为验证)
        validation_cutoff = data["time_idx"].max() - (self.max_prediction_length * 7)
        training_cutoff = data["time_idx"].max() - self.max_prediction_length
        training_data = data[data["time_idx"] <= training_cutoff]
        validation_data = data[data["time_idx"] > validation_cutoff]
        
        # 2. 生成 Dataset 和 Dataloader
        print("初始化 Dataset 与 DataLoader...")
        train_dataset = self.create_dataset(training_data, is_train=True)
        val_dataset = TimeSeriesDataSet.from_dataset(
                train_dataset,
                data,
                predict=True,
                stop_randomization=True
                )
        
        # CPU 建议减少 num_workers 
        num_workers = 0 if self.accelerator == 'cpu' else 4
        train_dataloader = train_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
        val_dataloader = val_dataset.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=num_workers)
        
        # 3. 定义 TFT 模型
        print("初始化 Temporal Fusion Transformer 网络架构...")
        hidden_size = 32 if self.accelerator == 'cpu' else 64
        
        self.model = TemporalFusionTransformer.from_dataset(
            train_dataset,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=16,
            # 使用 QuantileLoss 直接预测 [最下限, 均值, 最上限]
            loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
            log_interval=10, 
            reduce_on_plateau_patience=4,
        )
        
        # 4. 设置 PyTorch Lightning Trainer
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, verbose=True, mode="min")
        # 每轮保存最好的检查点
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.model_dir,
            filename='tft-best-checkpoint',
            save_top_k=1,
            monitor="val_loss",
            mode="min"
        )
        log_root = Path.cwd() / "tb_logs"
        log_root.mkdir(parents=True, exist_ok=True)

        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=self.accelerator,
            devices=self.devices,
            enable_model_summary=True,
            callbacks=[early_stop_callback, checkpoint_callback],
            #logger=TensorBoardLogger("lightning_logs", name=f"tft_logs_{self.station_name}")
            logger=False, default_root_dir=str(Path.cwd() / "pl_root")
        )
        
        # 5. 执行训练
        print(" 开始深入训练 TFT 网络...")
        self.trainer.fit(self.model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        # --- 验证指标：在 validation dataloader 上计算 MAE/RMSE/MAPE + pinball(0.1,0.5,0.9) ---
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        # val_dataloader 之前已构建为 val_dataloader
        val_preds = self._predict_quantiles_no_logger(val_dataloader)  # shape (N, pred_len, 3)
        # 收集 y_true（和 predict 顺序一致）
        y_trues = []
        for xb, yb in val_dataloader:
                y_trues.append(yb.numpy())
        y_trues = np.concatenate(y_trues, axis=0)[:,:,0]  # shape (N, pred_len)

        # 展平进行整体度量
        pred_q10 = val_preds[:,:,0].flatten()
        pred_q50 = val_preds[:,:,1].flatten()
        pred_q90 = val_preds[:,:,2].flatten()
        y_all = y_trues.flatten()

        mae = mean_absolute_error(y_all, pred_q50)
        rmse = (mean_squared_error(y_all, pred_q50))**0.5
        mape = (np.abs((y_all - pred_q50) / (np.where(y_all==0, 1e-8, y_all)))).mean() * 100

        def pinball_loss(y, y_pred, q):
                d = y - y_pred
                return np.mean(np.maximum(q * d, (q - 1) * d))

        pin10 = pinball_loss(y_all, pred_q10, 0.1)
        pin50 = pinball_loss(y_all, pred_q50, 0.5)
        pin90 = pinball_loss(y_all, pred_q90, 0.9)

        print("=== Validation Metrics ===")
        print(f"MAE (median): {mae:.6f}")
        print(f"RMSE (median): {rmse:.6f}")
        print(f"MAPE (median): {mape:.3f}%")
        print(f"Pinball@0.1: {pin10:.6f}, @0.5: {pin50:.6f}, @0.9: {pin90:.6f}")
        # --- 结束验证指标 ---
        
        # 恢复验证集上取得最好损失的权重
        best_model_path = self.trainer.checkpoint_callback.best_model_path
        if best_model_path:
            print(f"✓ 训练完成，从最佳 Checkpoint 恢复权重: {best_model_path}")
            self.model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
            
    def load_latest_model(self):
        """尝试从磁盘加载最新训练好的模型"""
        # 查找最新的 ckpt 文件
        ckpt_files = list(Path(self.model_dir).glob("*.ckpt"))
        if not ckpt_files:
            return False
            
        print(f"找到历史模型: {ckpt_files[0].name}，开始加载...")
        self.model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt_files[0]))
        return True
        
    def forecast_future(self, data, forecast_days=1):
        """利用 TFT 原生的多步推断机制直接盲预测未来区块"""
        if self.model is None:
            raise ValueError("模型未初始化，请先加载或训练模型！")
            
        print(f"\n TFT 开始多步长推断：直接预测未来 {forecast_days} 天...")
        
        all_q10, all_q50, all_q90 = [], [], []
        # 构造给模型用的输入序列：取原数据的最后 max_encoder_length 长度
        encoder_data = data.iloc[-self.max_encoder_length:].copy()
        
        # 根据需要，伪造时间块让模型去推断未来的协变量 (比如未来的小时数)
        # 例如想要预测 1天(96序列)，我们需要预先给它未来 96 个格子的 [已知特征]（时间是不受电价影响的）
        last_timestamp = encoder_data['timestamp'].iloc[-1]
        last_time_idx = encoder_data['time_idx'].iloc[-1]
        
        future_steps = forecast_days * 96
        future_dates = [last_timestamp + pd.Timedelta(minutes=15 * i) for i in range(1, future_steps + 1)]
        
        # 构建未来解码块 (Decoder context)
        decoder_df = pd.DataFrame({'timestamp': future_dates})
        decoder_df['station'] = self.station_name
        decoder_df['time_idx'] = [last_time_idx + i for i in range(1, future_steps + 1)]
        
        # 填充已知特征
        decoder_df['hour'] = decoder_df['timestamp'].dt.hour.astype(str)
        decoder_df['minute'] = decoder_df['timestamp'].dt.minute.astype(str)
        decoder_df['day_of_week'] = decoder_df['timestamp'].dt.dayofweek.astype(str)
        decoder_df[['hour', 'minute', 'day_of_week', 'station']] = decoder_df[['hour', 'minute', 'day_of_week', 'station']].astype('category')
        
        decoder_df['time_of_day_num'] = decoder_df['timestamp'].dt.hour + decoder_df['timestamp'].dt.minute / 60.0
        decoder_df['hour_sin'] = np.sin(2 * np.pi * decoder_df['time_of_day_num'] / 24.0)
        decoder_df['hour_cos'] = np.cos(2 * np.pi * decoder_df['time_of_day_num'] / 24.0)
        
        # 对于未来的"不知道的价格"，用原序列的最后一个值或NaN占位，TFT内部会自动屏蔽
        decoder_df['price'] = encoder_data['price'].iloc[-1] 
        
        # 注意：TFT的预测长度受到 max_prediction_length 的限制 (96步)。
        # 如果要求预测更远的未来，只能利用 TFT 做自回归分块追加(Chunk by chunk)。
        
        all_preds_p50 = []
        current_data = encoder_data.copy()
        
        # 切片处理预测未来（按配置的 max_prediction_length = 96块 切成多块执行）
        for block_start in range(0, future_steps, self.max_prediction_length):
            block_end = min(block_start + self.max_prediction_length, future_steps)
            block_decoder = decoder_df.iloc[block_start:block_end].copy()
            
            # 将 Encoder(历史) 和 Decoder(未来已知日历) 拼接交给验证系统
            inference_df = pd.concat([current_data, block_decoder], ignore_index=True)
            
            # 建立仅供单次推理的数据集验证机制
            inference_dataset = self.create_dataset(inference_df, is_train=False)
            dataloader = inference_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
            
            # 关闭梯度，获取输出 (mode='quantiles'直接获得 0.1, 0.5, 0.9)
            predictions = self._predict_quantiles_no_logger(dataloader)[0] 
            pred_q10 = predictions[:, 0]
            pred_q50 = predictions[:, 1]
            pred_q90 = predictions[:, 2]
            # prediction的维度将是 [prediction_length, quantiles_num] 

            
            predicted_p50 = predictions[:, 1].numpy() # Index 1 是中位数 (50%)
            all_preds_p50.extend(predicted_p50)
            all_q10.extend(pred_q10)
            all_q50.extend(pred_q50)
            all_q90.extend(pred_q90)
            
            # 将预测结果填回当前数据区，并将窗口前推，供下一块使用
            block_decoder['price'] = predicted_p50
            current_data = pd.concat([current_data, block_decoder], ignore_index=True)
            current_data = current_data.iloc[-self.max_encoder_length:] # 保持观测窗的长度

        # 如果返回长超了，裁剪至实际天数
        all_preds_p50 = all_preds_p50[:future_steps]
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,5))
        plt.plot(future_dates, all_q50[:future_steps], label='P50', color='orange')
        plt.fill_between(future_dates, all_q10[:future_steps], all_q90[:future_steps], color='orange', alpha=0.2, label='P10-P90')
        plt.title(f"{self.station_name} TFT Forecast Quantiles (P10/P50/P90)")
        plt.xlabel("time")
        plt.ylabel("price")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        return future_dates, all_preds_p50


# =========================================================
# 3. 生产逻辑封装
# =========================================================

def train_station_tft(station_name):
    print(f"\n[训练阶段] 开始基于 TFT 深度学习架构针对电站: {station_name} 进行训练...")
    
    predictor = TFTElectricityPredictor(station_name)
    
    data = load_station_electricity_data(station_name)
    if data is None or data.empty:
        raise ValueError(f"未获取到 {station_name} 的数据，无法训练。")
    
    # 核心特征工程：转化为TFT连续序列
    data_with_features = predictor.generate_tft_features(data)
    
    # 设置 epochs：如果是 GPU 可以拉大，CPU 保持小一点即可见效
    epochs_num = 20 if predictor.accelerator == 'gpu' else 5
    
    # 训练模型
    predictor.train_model(data_with_features, max_epochs=epochs_num)
    print("\n[训练结束] 模型检查点已自动保存至本地。")
    return predictor, data_with_features

def run_production_inference_tft(station_name='百合站', forecast_days=1):
    """每日自动运行的 TFT 深度学习流水线"""
    print(f"=== TFT 电力价格预测系统 -  ({station_name}) ===")
    
    predictor = TFTElectricityPredictor(station_name)
    
    data = load_station_electricity_data(station_name)
    if data is None or data.empty:
        print("无数据，退出。")
        return
        
    data_with_features = predictor.generate_tft_features(data)
    
    # 1. 尝试加载训练好的 TFT 模型
    is_loaded = predictor.load_latest_model()
    
    if not is_loaded:
        print(f"⚠ 未发现 {station_name} 的深度学习预训练模型文件，执行【全量训练】...")
        predictor, _ = train_station_tft(station_name)
    else:
        print("✓ 加载本地最新历史检查点(.ckpt)成功。")
        
        # [如果需要可以在次实现微调(在线学习)逻辑：修改小学习率然后 trainer.fit(model)]
        
    # 2. 对未来进行推断
    print(f"\n开始对未来 {forecast_days} 天进行多步长预测...")
    future_dates, future_preds = predictor.forecast_future(data_with_features, forecast_days=forecast_days)
    
    # 3. 绘制“未来曲线结果”
    plt.figure(figsize=(14, 6))
    
    # 过去最后1天(96个点) 的波动
    history_to_show = data_with_features.iloc[-96:] 
    plt.plot(history_to_show['timestamp'], history_to_show['price'], label='观测视窗内已知的真实特征序列', color='navy')
    plt.plot(future_dates, future_preds, label='TFT原生多步未来预测曲线 (P50 波动)', color='darkorange', linewidth=2.5)
    
    plt.title(f"{station_name} - 基于 TFT 注意力的未来 {forecast_days} 天序列生成")
    plt.xlabel("时间")
    plt.ylabel("预测电价")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(alpha=0.6)
    plt.tight_layout()
    plt.show()

def main():
    stations_to_run = ['百合站']  # 可以按需扩充
    
    for station in stations_to_run:
        print("=" * 60)
        # 推理并预测未来 1 天 
        run_production_inference_tft(station_name=station, forecast_days=1)
        print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ 运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 提示：运行本代码需要安装如下深度学习包:")
        print("pip install torch pytorch-lightning pytorch-forecasting")
        
    input("\n按Enter键退出...")
    
