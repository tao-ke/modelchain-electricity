import concurrent.futures
import itertools
import os
import time
import tempfile
import glob
import gradio as gr
import numpy as np
import pandas as pd
import  matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict

# ===== 设置中文字体 =====
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===== 设置随机种子，确保结果可重复 =====


# ==================== 价格动态演化模型 ====================
@dataclass
class PriceEvolutionModel:
    """价格动态演化模型参数"""
    base_price: float = 11.0           # 基准价格（当前价格）
    target_price: float = 5.0          # 长期均衡价格目标
    demand_growth_rate: float = 0.02   # 需求年增长率
    capacity_growth_rate: float = 0.15 # 储能容量年增长率
    learning_rate: float = 0.05        # 学习效应系数（成本下降速度）
    competition_factor: float = 0.1    # 竞争加剧系数
    policy_factor: float = 0.0         # 政策调整因子（负值表示补贴退坡）
    
    def calculate_future_price(self, years: int, current_capacity: float, 
                               demand_limit: float) -> float:
        """
        计算未来价格
        
        价格演化公式考虑：
        1. 供需关系：供大于求时价格下降压力大
        2. 学习曲线：技术进步降低成本
        3. 竞争效应：参与者增多导致报价下降
        4. 政策因素：市场规则调整
        """
        # 预测未来容量和需求
        future_capacity = current_capacity * ((1 + self.capacity_growth_rate) ** years)
        future_demand = demand_limit * ((1 + self.demand_growth_rate) ** years)
        
        # 供需比影响（供大于求 → 价格下降）
        supply_demand_ratio = future_capacity / future_demand
        supply_pressure = max(0, (supply_demand_ratio - 1) * 0.3)  # 供过于求的压力
        
        # 学习曲线效应（成本下降）
        learning_effect = (1 - self.learning_rate) ** years
        
        # 竞争加剧效应
        competition_effect = max(0.5, 1 - self.competition_factor * years * 0.1)
        
        # 政策因素
        policy_effect = 1 + self.policy_factor * years
        
        # 综合计算未来价格
        price = self.base_price * learning_effect * competition_effect * policy_effect
        
        # 供需压力导致的价格下降
        price = price * (1 - supply_pressure)
        
        # 确保价格在合理范围内
        price = max(3.5, min(15.0, price))
        
        return price
    
    def generate_price_scenarios(self, years_list: List[int], 
                                  current_capacity: float,
                                  demand_limit: float) -> Dict[str, List[float]]:
        """生成多情景价格预测"""
        scenarios = {
            '乐观情景（政策利好+需求快速增长）': [],
            '基准情景（市场自然演化）': [],
            '悲观情景（竞争加剧+补贴退坡）': []
        }
        
        # 保存原始参数
        original_params = {
            'demand_growth_rate': self.demand_growth_rate,
            'capacity_growth_rate': self.capacity_growth_rate,
            'learning_rate': self.learning_rate,
            'competition_factor': self.competition_factor,
            'policy_factor': self.policy_factor
        }
        
        # 乐观情景：容量增长慢，价格跌幅小
        self.capacity_growth_rate = 0.10
        self.policy_factor = 0.05
        for year in years_list:
            scenarios['乐观情景（政策利好+需求快速增长）'].append(
                self.calculate_future_price(year, current_capacity, demand_limit)
            )
        
        # 基准情景：市场自然演化
        self.demand_growth_rate = original_params['demand_growth_rate']
        self.capacity_growth_rate = original_params['capacity_growth_rate']
        self.policy_factor = original_params['policy_factor']
        for year in years_list:
            scenarios['基准情景（市场自然演化）'].append(
                self.calculate_future_price(year, current_capacity, demand_limit)
            )
        
        # 悲观情景：容量增长快，竞争激烈，补贴退坡
        self.capacity_growth_rate = 0.20
        self.competition_factor = 0.20
        self.policy_factor = -0.03
        for year in years_list:
            scenarios['悲观情景（竞争加剧+补贴退坡）'].append(
                self.calculate_future_price(year, current_capacity, demand_limit)
            )
        
        # 恢复原始参数
        self.demand_growth_rate = original_params['demand_growth_rate']
        self.capacity_growth_rate = original_params['capacity_growth_rate']
        self.competition_factor = original_params['competition_factor']
        self.policy_factor = original_params['policy_factor']
        
        return scenarios

  #申报容量
def simple_fixed_sum_offers_vec(plant_cap, dist_demand, cap_ratio_upper, cap_ratio_lower, Q_lb, target_sum, n_plants, n_sims, rng, is_non_indep=False):
    if is_non_indep:
        upper = 150
        lower = 100
        initial_offers = rng.uniform(lower, upper, size=(n_sims, n_plants))
    else:
        upper = np.minimum(plant_cap, dist_demand * cap_ratio_upper)
        lower_candidate = np.maximum(0.2 * plant_cap, Q_lb)
        lower = np.minimum(lower_candidate, dist_demand * cap_ratio_lower)
        lower = np.minimum(lower, upper)
    
        initial_offers = rng.triangular(lower, upper, upper, size=(n_sims, n_plants))
    
    sums = np.sum(initial_offers, axis=1, keepdims=True)
    # 避免除以零
    sums = np.where(sums > 0, sums, 1)
    
    scale_factor = target_sum / sums
    scaled_offers = initial_offers * scale_factor
    scaled_offers = np.clip(scaled_offers, lower, upper)
    
    final_sums = np.sum(scaled_offers, axis=1, keepdims=True)
    diffs = target_sum - final_sums
    
    # 计算每台电站可增加的空间
    slack = upper - scaled_offers
    # 对每次模拟，找到最接近下限的电站索引，调整其申报容量以修正总和
    min_idx = np.argmin(scaled_offers, axis=1)
    batch_idx = np.arange(n_sims)
    
    # 只调整那些差异较大的情况，避免微小数值误差导致不必要的调整
    mask = np.abs(diffs.flatten()) > 1e-6
    if np.any(mask):
        #只在slack内分摊调整量，避免超过上限
        adjustment = np.minimum(diffs.flatten()[mask], slack[batch_idx[mask], min_idx[mask]])
        scaled_offers[batch_idx[mask], min_idx[mask]] += adjustment
        #scaled_offers[batch_idx[mask], min_idx[mask]] += diffs.flatten()[mask]
        
    return scaled_offers


def monte_carlo_simulation(
    total_capacity,          # 总储能容量 (MW)
    max_demand,              # 全省调频需求上限 (MW)  # 默认 1300
    dist_demand_ratio,       # 本资源区需求占比 (0~1)
    bid_mean,                # 报价均值 (元/MW)
    bid_std_ratio,           # 报价波动系数
    n_sims,                  # 模拟次数
    k_mean, k_std,           # 排序性能指标 k 的均值和标准差
    m_mean, m_std,           # 结算性能系数 m 的基准均值和标准差
    m_competition_factor,    # 优胜劣汰增强系数
    m_max,                   # m 值上限
    m_unknown_ratio,     # m 值不确定系数
    lambda_dis, lambda_ch,   # 放电和充电的电价（用于成本计算）
    N_day_mean, N_day_std,   # 日响应次数均值和标准差
    d_mean, d_std,           # 调节深度均值和标准差
    U_x, U_y,                # 边际替代率曲线参数
    cap_ratio_upper,         # 申报容量上限比例 (如 0.2)
    cap_ratio_lower,         # 申报容量下限比例 (如 0.15)
    Q_lb,                    # 保底容量 (MW)
    price_min, price_max,    # 报价上下限 (元/MW)
    non_indep_capacity=5000,  # 非独立储能总容量 (MW)
    non_indep_bid_mean=5.0,  # 非独立储能报价均值 (元/MW)
    non_indep_k_mean=1.0,    # 非独立储能k值均值
    return_detailed_stats=False,  # 是否返回详细统计（用于敏感性分析）
    target_bid=None,         # 目标电站的特定报价（用于价格敏感性分析）
    target_k=None,           # 目标电站的特定k值（用于k值敏感性分析）
    export_target_capacity=None,  # 触发调试导出的目标总容量
    export_target_bid=None,       # 触发调试导出的目标报价
    export_target_k=None,          # 触发调试导出的目标K值
    seed=42
):
    """
    完全符合南方区域调频市场 2025 年版细则的蒙特卡洛模拟。
    优胜劣汰机制：当总容量超过需求上限时，m 均值按比例提高，但不超过 m_max。
    
    新增功能：
    - 跟踪不同报价水平下的中标概率
    - 考虑 M 值对中标概率的影响（通过排序价格公式）
    - 可选返回详细统计数据
    - 支持目标电站特定报价分析（用于价格敏感性分析）
    """
    # 设置随机种子
    rng = np.random.default_rng(seed)
    # 平均单站容量设为 150 MW
    avg_plant_size = 150.0
    n_plants = max(1, int(np.ceil(total_capacity / avg_plant_size)))    #站点数量。
    plant_cap = total_capacity / n_plants                             # 每个电站的额定容量
    
    # 计算本资源区需求
    dist_demand = max_demand * dist_demand_ratio
    
    # 优胜劣汰调整：计算供需比，调整 m 均值，并应用上限
    #supply_demand_ratio = total_capacity / max_demand
    #if supply_demand_ratio > 1.0:
    #    adjusted_m_mean = m_mean * (1 + m_competition_factor * (supply_demand_ratio - 1))
    #else:
    #    adjusted_m_mean = m_mean
    #adjusted_m_mean = min(adjusted_m_mean, m_max)  # 应用上限
    
    # ----- 生成每个电站的随机参数 -----
    # 报价：如果指定了目标报价，第一个电站使用目标报价，其他电站使用随机报价
    if target_bid is not None:
        # 竞争对手报价围绕市场均价 bid_mean 分布
        competitor_bids = rng.normal(bid_mean, bid_std_ratio * bid_mean ,size=(n_sims, n_plants - 1))
        #competitor_bids = rng.triangular(3.5, bid_mean, 15.0, n_plants - 1) 
        #competitor_bids = rng.uniform(bid_mean - 2*bid_std_ratio*bid_mean, bid_mean + 2*bid_std_ratio*bid_mean, n_plants - 1)  # 使用均匀分布生成竞争对手报价
        #competitor_bids = rng.uniform(bid_mean - 2*bid_std_ratio*bid_mean, bid_mean + 2*bid_std_ratio*bid_mean, size=(n_sims, n_plants - 1))
        competitor_bids = np.clip(competitor_bids, price_min, price_max)
        bids = np.column_stack([np.full((n_sims, 1), target_bid), competitor_bids])
        
        # 排序性能指标 k (用于排序)
        # 如果指定了目标报价，第一个电站使用特定k值或平均性能，其他电站随机
        #competitor_k = rng.uniform(k_mean - k_std, k_mean + k_std, size=(n_sims, n_plants - 1))
        competitor_k = rng.normal(k_mean, k_std, size=(n_sims, n_plants - 1))  # 使用正态分布生成竞争对手的k值
        competitor_k = np.clip(competitor_k, 0.1, 3)
        # 使用target_k（如果提供）或k_mean作为目标电站的k值
        target_k_value = target_k if target_k is not None else k_mean
        k = np.column_stack([np.full((n_sims, 1), target_k_value), competitor_k])
        
        # 结算性能系数 m (用于收益) —— 使用调整后的均值，并截断至 [1.2, m_max]
        #competitor_m = rng.normal(adjusted_m_mean, m_std, size=(n_sims, n_plants - 1))
        #competitor_m = np.clip(competitor_m, 1.2, m_max)
        #m = np.column_stack([np.full((n_sims, 1), adjusted_m_mean), competitor_m])
    else:
        bids = rng.normal(bid_mean, bid_std_ratio * bid_mean, size=(n_sims, n_plants))
        bids = np.clip(bids, price_min, price_max)
        
        k = rng.uniform(k_mean - k_std, k_mean + k_std, size=(n_sims, n_plants))
        k = np.clip(k, 0.1, 3)
        
        #m = rng.normal(adjusted_m_mean, m_std, size=(n_sims, n_plants))
        #m = np.clip(m, 1.2, m_max)
    
    # 基础线性映射
    m_base = 1.2 + 0.8 * (k - 0.1) / 2.9

    # 添加噪声（噪声大小可调）
    noise_scale = 0.01  # 噪声标准差，控制相关性强度
    m_noise = rng.normal(0, noise_scale, size=k.shape)
    m = m_base + m_noise
    m = m * m_unknown_ratio

    # 裁剪到有效范围
    m = np.clip(m, 1.2, 2.0)

    # 日响应次数
    N_day = rng.uniform(200, 400, size=(n_sims, n_plants))
    #N_day = np.clip(N_day, 200, 400)  # 限制在合理范围内
    N_day = np.round(N_day).astype(int) # 取整
    # 调节深度
    d = rng.normal(d_mean, d_std, size=(n_sims, n_plants))
    d = np.clip(d, 0.3, 0.5)
    # 申报容量
    offers = simple_fixed_sum_offers_vec(plant_cap, dist_demand, cap_ratio_upper, cap_ratio_lower, Q_lb, total_capacity, n_plants, n_sims, rng)
    
    # ===== 新增：非独立储能数据生成 =====
    if non_indep_capacity > 0:
        n_plants_ni = max(1, int(np.ceil(non_indep_capacity / 600)))
        plant_cap_ni = non_indep_capacity / n_plants_ni
        
        # 报价均值 4-6，正态分布
        bids_ni = rng.normal(non_indep_bid_mean, bid_std_ratio * non_indep_bid_mean, size=(n_sims, n_plants_ni))
        bids_ni = np.clip(bids_ni, price_min, price_max)
        
        # k值 0.8-2
        k_ni = rng.normal(non_indep_k_mean, k_std, size=(n_sims, n_plants_ni))
        k_ni = np.clip(k_ni, 0.8, 2.0)
        
        # 非独立储能申报上限为7.5%，下限为3%
        offers_ni = simple_fixed_sum_offers_vec(plant_cap_ni, dist_demand, 0.075, 0.03, Q_lb, non_indep_capacity, n_plants_ni, n_sims, rng, is_non_indep=True)
        
        # 合并全场数据以求全局排名
        bids_all = np.column_stack([bids, bids_ni])
        k_all = np.column_stack([k, k_ni])
        offers_all = np.column_stack([offers, offers_ni])
    else:
        n_plants_ni = 0
        bids_all = bids
        k_all = k
        offers_all = offers

    # ----- 归一化排序性能指标 P_i (基于全场) -----
    k_max_arr = np.max(k_all, axis=1, keepdims=True)
    k_max_arr = np.maximum(k_max_arr, 1e-6)
    P_all = k_all / k_max_arr
    
    # ----- 先计算全场内部价格（用于各自排序） -----
    internal_price_all = bids_all / P_all
    
    # ----- 边际系数 Fm：仅独立储能参与，按累计占比线性衰减到 0 -----
    # 规则：累计占比 <= U_x 时 Fm = 1 - cum_ratio / U_x；超过 U_x 时 Fm=0
    # 这样独立储能在累计占比超过阈值后排序价格会趋于无穷大，为非独立储能保留容量空间。
    Fm_all = np.ones_like(internal_price_all)
    internal_price_ind = internal_price_all[:, :n_plants]
    ratios_ind = offers_all[:, :n_plants] / dist_demand
    sorted_idx_ind = np.argsort(internal_price_ind, axis=1)
    ratios_sorted_ind = np.take_along_axis(ratios_ind, sorted_idx_ind, axis=1)
    cum_ratio_sorted_ind = np.cumsum(ratios_sorted_ind, axis=1)

    safe_Ux = max(U_x, 1e-6)
    Fm_sorted_ind = np.where(
        cum_ratio_sorted_ind <= safe_Ux,
        U_y * (1 - cum_ratio_sorted_ind / safe_Ux),
        0.0
    )
    Fm_ind = np.ones_like(internal_price_ind)
    np.put_along_axis(Fm_ind, sorted_idx_ind, Fm_sorted_ind, axis=1)
    Fm_all[:, :n_plants] = Fm_ind

    # 非独立储能不考虑边际系数，恒为 1.0
    if n_plants_ni > 0:
        Fm_all[:, n_plants:] = 1.0

    safe_Fm_all = np.where(Fm_all > 0, Fm_all, 1e-6)
    
    # ----- 排序价格 -----
    sort_price_all = bids_all / (P_all * safe_Fm_all)
    
    # ----- 市场出清 -----
    order_all = np.argsort(sort_price_all, axis=1)
    offers_sorted_all = np.take_along_axis(offers_all, order_all, axis=1)
    
    accumulated_all = np.cumsum(offers_sorted_all, axis=1)
    prev_accumulated_all = accumulated_all - offers_sorted_all
    
    assigned_sorted_all = np.clip(dist_demand - prev_accumulated_all, 0, offers_sorted_all)

    # 提取重新排队后的 Fm 值
    Fm_for_clearing = np.take_along_axis(Fm_all, order_all, axis=1)
    assigned_sorted_all = np.where(Fm_for_clearing > 0, assigned_sorted_all, 0)
    
    
    cleared_cap_all = np.zeros_like(assigned_sorted_all)
    np.put_along_axis(cleared_cap_all, order_all, assigned_sorted_all, axis=1)
    
    sort_price_sorted_all = np.take_along_axis(sort_price_all, order_all, axis=1)
    
    # ----- 出清价格（全场边际机组口径）-----
    # 按全场最终排序价格计算总累积容量占比，取累计占比不超过 1 的最高排序价格
    cum_ratio_market = np.cumsum(offers_sorted_all / dist_demand, axis=1)
    market_eligible = cum_ratio_market <= 1.0
    clear_price_per_sim = np.max(np.where(market_eligible, sort_price_sorted_all, 0), axis=1)

    # 极端保护：若没有满足累计占比<=1的机组，则回退到全场最低排序价
    no_eligible_mask = ~np.any(market_eligible, axis=1)
    if np.any(no_eligible_mask):
        clear_price_per_sim[no_eligible_mask] = sort_price_sorted_all[no_eligible_mask, 0]

    clear_price = np.minimum(clear_price_per_sim, price_max)

    # ----- 剥离结果：重新截取仅属于【独立储能】的数据作结算和评估 -----
    cleared_cap = cleared_cap_all[:, :n_plants]
    sort_price = sort_price_all[:, :n_plants]
    Fm = Fm_all[:, :n_plants]
    internal_price = internal_price_all[:, :n_plants]
    cum_ratio_sorted = cum_ratio_sorted_ind
    
    # ==========================
    # 统一计算收益、成本矩阵 (重构后)
    # ==========================
    # 中标容量 × 每天频次 × 深度 × 330
    coeff = cleared_cap * N_day * d * 330.0
    cost = coeff * (0.0333 * (0.85 * lambda_dis - lambda_ch) / 10.0)
    daily_rev = coeff * clear_price[:, None] * m / np.float64(10000.0)
    
    # 统一为期望收益算法: (每日收益 + 成本增量) / 换算到每兆瓦/每月的维度
    plant_monthly_rev_matrix = (daily_rev + cost) / plant_cap / 12.0
    
    # ----- 月收益（万元/MW/月） -----
    avg_revenues = np.sum(plant_monthly_rev_matrix * plant_cap, axis=1) / total_capacity
    
    detailed_stats = None
    if return_detailed_stats:
        detailed_stats = {}
        if target_bid is not None:
            # 统一提取第一台（目标号机）数据
            target_cleared = cleared_cap[:, 0]
            detailed_stats['target_monthly_revs'] = plant_monthly_rev_matrix[:, 0].tolist()
            
            if n_sims > 0:
                limit = min(n_sims, 500)  # 只记录前 500 次以减少内存
                detailed_stats['bid_levels'] = bids[:limit, 0].tolist()
                detailed_stats['m_values'] = m[:limit, 0].tolist()
                detailed_stats['revenues'] = (target_cleared[:limit] * clear_price[:limit] * m[:limit, 0]).tolist()
                detailed_stats['win_rates'] = (target_cleared[:limit] > 0.01).astype(float).tolist()
                
                safe_offers = np.where(offers[:limit, 0] > 0.01, offers[:limit, 0], 1)
                detailed_stats['cap_utilization'] = np.where(offers[:limit, 0] > 0.01, target_cleared[:limit] / safe_offers, 0).tolist()
                detailed_stats['cleared_capacity'] = target_cleared[:limit].tolist()
                
    # ----- 调试打印与导出：重构后的动态参数导出xlsx -----
    if (export_target_capacity is not None and 
        export_target_bid is not None and 
        export_target_k is not None and 
        total_capacity == export_target_capacity and 
        target_bid == export_target_bid and 
        target_k == export_target_k):
        
        print(f"【调试输出】触发导出参数: 总容量={total_capacity}, 目标电站报价={target_bid}, 竞争电站均价={bid_mean}, 目标性能K={target_k}, 对手性能均K={k_mean}")
        print("="*90)
        print_sims = min(5, n_sims)

        debug_records = []
        for sim in range(print_sims):
            print(f"\n--- 第 {sim + 1} 次模拟结果 ---")
            print(f"全场统一出清价: {clear_price[sim]:.4f} 元/MW")

            # 使用全场最终排序后的索引，确保导出覆盖独立+非独立全部机组
            sorted_indices_all = order_all[sim]
            for rank, p_all in enumerate(sorted_indices_all):
                is_indep = p_all < n_plants

                if is_indep:
                    plant_label = "目标电站" if p_all == 0 else f"竞争对手_{p_all}"
                    unit_type = "独立储能"
                    plant_cost_wan = cost[sim, p_all] / plant_cap / 12.0
                    plant_rev_matrix_val = plant_monthly_rev_matrix[sim, p_all]
                    d_val = d[sim, p_all]
                    n_day_val = N_day[sim, p_all]
                    m_val = m[sim, p_all]
                    # 找到当前机组在独立储能排序中的位置
                    sorted_idx = sorted_idx_ind[sim]
                    # 逆序映射：机组p_all在排序中的位置
                    indep_rank = np.where(sorted_idx == p_all)[0][0]
                    indep_cum_ratio = cum_ratio_sorted[sim, indep_rank]
                else:
                    ni_idx = p_all - n_plants
                    plant_label = f"非独立储能_{ni_idx}"
                    unit_type = "非独立储能"
                    plant_cost_wan = None
                    plant_rev_matrix_val = None
                    d_val = None
                    n_day_val = None
                    m_val = None
                    indep_cum_ratio = None

                debug_records.append({
                    "模拟批次": sim + 1,
                    "统一出清价(元/MW)": clear_price[sim],
                    "排名": rank + 1,
                    "机组类型": unit_type,
                    "电站标识": plant_label,
                    "排序价格(元/MW)": sort_price_all[sim, p_all],
                    "中标容量(MW)": cleared_cap_all[sim, p_all],
                    "性能指标K": k_all[sim, p_all],
                    "真实报价(元/MW)": bids_all[sim, p_all],
                    "内部价格(报价/K)": internal_price_all[sim, p_all],
                    "初始申报容量(MW)": offers_all[sim, p_all],
                    "Fm系数": Fm_all[sim, p_all],
                    "独储累积容量占比": indep_cum_ratio,
                    "最终累积容量占比(cum_ratio)": cum_ratio_market[sim, rank],
                    "调节深度": d_val,
                    "日响应次数": n_day_val,
                    "结算M值": m_val,
                    "成本(万元)": plant_cost_wan,
                    "预期月收益(万元/MW)": plant_rev_matrix_val,
                })
        print("="*90 + "\n")

        if detailed_stats is None:
            detailed_stats = {}
        detailed_stats['debug_records'] = debug_records

    return avg_revenues, detailed_stats


def run_single_simulation(
    total_capacity, max_demand, dist_demand_ratio,
    bid_mean, bid_std_ratio, n_sims,
    k_mean, k_std, m_mean, m_std, m_competition_factor, m_max, m_unknown_ratio,
    lambda_dis, lambda_ch,
    N_day_mean, N_day_std, d_mean, d_std,
    U_x, U_y,
    cap_ratio_upper, cap_ratio_lower, Q_lb,
    price_min, price_max,
    non_indep_capacity, non_indep_bid_mean, non_indep_k_mean
):
    """单次模拟，返回图表和统计表"""
    avg_revs, _ = monte_carlo_simulation(
        total_capacity, max_demand, dist_demand_ratio,
        bid_mean, bid_std_ratio, n_sims,
        k_mean, k_std, m_mean, m_std, m_competition_factor, m_max, m_unknown_ratio,
        lambda_dis, lambda_ch,
        N_day_mean, N_day_std, d_mean, d_std,
        U_x, U_y,
        cap_ratio_upper, cap_ratio_lower, Q_lb,
        price_min, price_max,
        non_indep_capacity, non_indep_bid_mean, non_indep_k_mean,
        return_detailed_stats=False
    )

    p50 = np.percentile(avg_revs, 50)
    p10 = np.percentile(avg_revs, 10)
    p90 = np.percentile(avg_revs, 90)
    mean_val = np.mean(avg_revs)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(avg_revs, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(p50, color='red', linestyle='--', linewidth=2, label=f'P50 = {p50:.4f}')
    ax.axvline(p10, color='orange', linestyle='--', linewidth=2, label=f'P10 = {p10:.4f}')
    ax.axvline(p90, color='green', linestyle='--', linewidth=2, label=f'P90 = {p90:.4f}')
    ax.set_xlabel('平均收益 (万元/MW/月)')
    ax.set_ylabel('频次')
    ax.set_title(f'平均收益分布 (总容量={total_capacity} MW, 报价均值={bid_mean}元/MW)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    stats_df = pd.DataFrame({
        "指标": ["期望收益（平均收益）", "P50", "P10", "P90"],
        "数值": [
            f"{mean_val:.4f} 万元/MW/月",
            f"{p50:.4f} 万元/MW/月",
            f"{p10:.4f} 万元/MW/月",
            f"{p90:.4f} 万元/MW/月"
        ]
    })

    return fig, stats_df


def run_capacity_scan(
    cap_start, cap_end, cap_step, scan_n_sims,
    max_demand, dist_demand_ratio,
    bid_mean, bid_std_ratio,
    k_mean, k_std, m_mean, m_std, m_competition_factor, m_max, m_unknown_ratio,
    lambda_dis, lambda_ch,
    N_day_mean, N_day_std, d_mean, d_std,
    U_x, U_y,
    cap_ratio_upper, cap_ratio_lower, Q_lb,
    price_min, price_max,
    non_indep_capacity, non_indep_bid_mean, non_indep_k_mean
):
    """
    容量扫描：对一系列总容量进行蒙特卡洛模拟，绘制收益 - 容量曲线和 m 均值曲线，
    并计算每步的下降量。
    
    核心逻辑：
    - 全省调频需求上限不变（固定为 max_demand）
    - 总储能容量增加（新参与者建设电站）
    - 供过于求 → M 值上升（优胜劣汰机制）
    - 竞争加剧 → 出清价格下降
    - 综合影响：单位收益 = f(价格↓, M↑)
    """
    capacities = np.arange(cap_start, cap_end + 1, cap_step)
    mean_returns = []
    p50_returns = []
    adj_m_means = []  # 截断后的 m 均值
    clear_prices_mean = []  # 平均出清价格
    clear_prices_std = []   # 出清价格标准差

    for C in capacities:
        # 计算截断后的理论 m 均值（用于绘图）
        supply_demand_ratio = C / max_demand
        if supply_demand_ratio > 1.0:
            raw_m = m_mean * (1 + m_competition_factor * (supply_demand_ratio - 1))
        else:
            raw_m = m_mean
        adj_m = min(raw_m, m_max)
        adj_m_means.append(adj_m)

        # 运行模拟
        avg_revs_result, _ = monte_carlo_simulation(
            total_capacity=C,
            max_demand=max_demand,
            dist_demand_ratio=dist_demand_ratio,
            bid_mean=bid_mean,
            bid_std_ratio=bid_std_ratio,
            n_sims=scan_n_sims,
            k_mean=k_mean, k_std=k_std,
            m_mean=m_mean, m_std=m_std,
            m_competition_factor=m_competition_factor,
            m_max=m_max, m_unknown_ratio=m_unknown_ratio,
            lambda_dis=lambda_dis, lambda_ch=lambda_ch,
            N_day_mean=N_day_mean, N_day_std=N_day_std,
            d_mean=d_mean, d_std=d_std,
            U_x=U_x, U_y=U_y,
            cap_ratio_upper=cap_ratio_upper,
            cap_ratio_lower=cap_ratio_lower,
            Q_lb=Q_lb,
            price_min=price_min, price_max=price_max,
            non_indep_capacity=non_indep_capacity, non_indep_bid_mean=non_indep_bid_mean, non_indep_k_mean=non_indep_k_mean,
            return_detailed_stats=False
        )
        mean_returns.append(np.mean(avg_revs_result))
        p50_returns.append(np.percentile(avg_revs_result, 50))

    # 绘制收益-容量曲线
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(capacities, mean_returns, 'b-o', label='期望收益（平均收益）')
    ax1.plot(capacities, p50_returns, 'r--s', label='P50收益')
    ax1.set_xlabel('总容量 (MW)')
    ax1.set_ylabel('平均收益 (万元/MW/月)')
    ax1.set_title('收益随容量变化曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()

    # 绘制调整后 m 均值曲线（已截断）
    fig2, ax2_m = plt.subplots(figsize=(10, 5))
    ax2_m.plot(capacities, adj_m_means, 'g-^', linewidth=2.5, markersize=8, 
             label='调整后 m 均值（截断后）')
    ax2_m.axhline(y=m_max, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                label=f'm 上限={m_max}')
    ax2_m.axhline(y=m_mean, color='blue', linestyle='--', linewidth=1.5, alpha=0.5, 
                label=f'm 基准值={m_mean}')
        
    # 标记 m 值开始上升的点
    if capacities[0] <= max_demand <= capacities[-1]:
        ax2_m.axvline(x=max_demand, color='gray', linestyle=':', linewidth=2, 
                    alpha=0.7, label=f'供需平衡点={max_demand:.0f}MW')
        
    ax2_m.set_xlabel('总储能容量 (MW)', fontsize=13, fontweight='bold')
    ax2_m.set_ylabel('结算性能系数 m', fontsize=13, fontweight='bold')
    ax2_m.set_title('优胜劣汰机制：M 值随容量增加而上升（供过于求时启动）', fontsize=14, fontweight='bold')
    ax2_m.legend(loc='upper right', fontsize=11)
    ax2_m.grid(True, alpha=0.3)
    plt.tight_layout()

    # 计算下降量表格
    delta_table = []
    for i in range(1, len(capacities)):
        cap_prev = capacities[i-1]
        cap_cur = capacities[i]
        mean_prev = mean_returns[i-1]
        mean_cur = mean_returns[i]
        p50_prev = p50_returns[i-1]
        p50_cur = p50_returns[i]
        delta_mean = mean_cur - mean_prev
        delta_p50 = p50_cur - p50_prev
        delta_mean_percent = (delta_mean / mean_prev) * 100 if mean_prev != 0 else 0
        delta_p50_percent = (delta_p50 / p50_prev) * 100 if p50_prev != 0 else 0
        delta_table.append({
            "容量区间 (MW)": f"{cap_prev:.0f} → {cap_cur:.0f}",
            "期望收益下降 (万元/MW/月)": f"{delta_mean:+.4f}",
            "期望收益下降 (%)": f"{delta_mean_percent:+.2f}%",
            "P50 下降 (万元/MW/月)": f"{delta_p50:+.4f}",
            "P50 下降 (%)": f"{delta_p50_percent:+.2f}%"
        })
    delta_df = pd.DataFrame(delta_table)
    
    # 生成关键指标解读
    critical_idx = np.argmin(np.abs(capacities - max_demand))
    critical_revenue = mean_returns[critical_idx]
    base_revenue = mean_returns[0]
    final_revenue = mean_returns[-1]
    
    # 计算 M 值达到上限的容量点
    m_max_idx = None
    for i, m_val in enumerate(adj_m_means):
        if abs(m_val - m_max) < 0.01:
            m_max_idx = i
            break
    
    analysis_summary = f"""
【容量扫描分析关键结论】

📊 市场参数：
- 全省调频需求上限：{max_demand:.0f} MW（固定不变）
- 容量扫描范围：{capacities[0]:.0f} ~ {capacities[-1]:.0f} MW
- 当前报价均值：{bid_mean:.1f} 元/MW
- M 值范围：{m_mean:.2f} ~ {m_max:.2f}
- 优胜劣汰增强系数γ：{m_competition_factor:.2f}

🎯 关键转折点：
- 供需平衡点：{max_demand:.0f} MW（此时单位收益={critical_revenue:.4f} 万元/MW/月）
- 相比基准收益下降：{(1-critical_revenue/base_revenue)*100:.2f}%
"""
    
    if m_max_idx is not None:
        m_max_capacity = capacities[m_max_idx]
        m_max_revenue = mean_returns[m_max_idx]
        analysis_summary += f"""
- M 值达到上限点：{m_max_capacity:.0f} MW（此时 M={m_max:.2f}，收益={m_max_revenue:.4f} 万元/MW/月）
"""
    
    analysis_summary += f"""
📉 极端情况：
- 容量={capacities[-1]:.0f} MW 时，单位收益={final_revenue:.4f} 万元/MW/月
- 累计下降幅度：{(1-final_revenue/base_revenue)*100:.2f}%

💡 趋势解读：
1️⃣ 当总容量 < {max_demand:.0f} MW 时：
   - 市场竞争温和，M 值保持{m_mean:.2f}不变
   - 出清价格稳定在{bid_mean:.1f}元/MW 附近
   - 单位收益相对稳定

2️⃣ 当总容量 > {max_demand:.0f} MW 时：
   - 启动优胜劣汰机制，M 值开始上升
   - 但竞争加剧导致出清价格下降
   - M 值上升部分抵消价格下降影响，收益加速下滑

3️⃣ 当 M 值达到上限{m_max:.2f}后{f'（容量>{m_max_capacity:.0f}MW）' if m_max_idx else ''}：
   - M 值不再变化，失去缓冲作用
   - 价格持续下降，收益线性快速下滑

❓ 常见疑问解答：
Q1: 为什么当前报价 11 元/MW，最优报价却是 13.57 元/MW？
A1: 因为容量过剩时 M 值上升（1.45→2.0），提高了高价中标的收益回报。
    虽然高价降低了中标概率，但 M 值的放大效应使期望收益最大化。
    计算公式：期望收益 = 中标概率 × (报价 × M - 成本)
    
Q2: 敢报高价吗？会不会亏本？
A2: 需要权衡利弊：
    ✅ 优势：M 值提升放大收益（最高 +38%），适合性能好的电站
    ❌ 风险：报价过高会显著降低中标概率
    💡 建议：根据自身的 k、m 指标动态调整，不要盲目追高
    
Q3: 收益为何"先高后低"？是真的先升高再降低吗？
A3: 实际上收益是单调下降的！看似"先高后低"的错觉是因为：
    - 容量<1300MW 时：M 值不变，收益下降缓慢（视觉上的"高"）
    - 容量 1300~2000MW：M 值上升缓冲，但仍难抵价格下跌（过渡期）
    - 容量>2000MW 后：M 值失效（已达上限 2.0），收益直线下降（真实的"低"）
    ⚠️ 注意：如果看到收益曲线有波动，可能是蒙特卡洛模拟的统计误差
       建议增加模拟次数到 1000-2000 次以平滑曲线

📌 策略建议：
✅ 当前容量接近{max_demand:.0f} MW 时：尽快投产，抢占最后的高收益窗口期
⚠️  规划容量超过{max_demand*1.5:.0f} MW 时：谨慎投资，预期收益将下降 50% 以上
🔍 关注政策变化：若需求上限提升，可延缓收益下降趋势
📈 差异化竞争：提升性能指标 k 和 m，在同等报价下获得更高排序
🎯 报价策略：参考"价格敏感性分析"标签页，找出自身的最优报价点
🔢 投资决策：使用"容量扫描分析"预测不同容量下的收益水平
"""

    return fig1, fig2, delta_df, analysis_summary


def run_fixed_revenue_analysis(
    base_capacity,           # 基准容量 (MW)
    total_revenue_wan,       # 总收益 (万元)
    cap_start, cap_end, cap_step,  # 容量扫描范围
    max_demand, dist_demand_ratio,
    bid_mean, bid_std_ratio,
    k_mean, k_std, m_mean, m_std, m_competition_factor, m_max, m_unknown_ratio,
    lambda_dis, lambda_ch,
    N_day_mean, N_day_std, d_mean, d_std,
    U_x, U_y,
    cap_ratio_upper, cap_ratio_lower, Q_lb,
    price_min, price_max,
    non_indep_capacity, non_indep_bid_mean, non_indep_k_mean
):
    """
    固定总收益分析：在总收益不变的情况下，分析容量增加对单位收益的影响。
    基于实际市场数据：总收益8.11亿元，基准容量1280MW。
    """
    # 计算基准单位收益
    base_unit_revenue = total_revenue_wan / base_capacity / 12  # 万元/MW/月
    
    capacities = np.arange(cap_start, cap_end + 1, cap_step)
    
    results = []
    for C in capacities:
        # 总收益不变，计算新的单位收益
        new_unit_revenue = total_revenue_wan / C / 12  # 万元/MW/月
        
        # 计算相对于基准的变化
        unit_revenue_change = new_unit_revenue - base_unit_revenue
        unit_revenue_change_pct = (unit_revenue_change / base_unit_revenue) * 100 if base_unit_revenue != 0 else 0
        
        # 计算总收益分配比例（用于模拟市场竞争强度）
        supply_demand_ratio = C / max_demand
        
        results.append({
            "总容量 (MW)": f"{C:.0f}",
            "总收益 (万元)": f"{total_revenue_wan:,.2f}",
            "单位收益 (万元/MW/月)": f"{new_unit_revenue:.4f}",
            "单位收益变化 (万元/MW/月)": f"{unit_revenue_change:+.4f}",
            "单位收益变化 (%)": f"{unit_revenue_change_pct:+.2f}%",
            "供需比": f"{supply_demand_ratio:.2f}"
        })
    
    results_df = pd.DataFrame(results)
    
    # 绘制单位收益随容量变化曲线
    unit_revenues = [total_revenue_wan / C for C in capacities]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 主曲线
    ax.plot(capacities, unit_revenues, 'b-o', linewidth=2, markersize=8, label='单位收益')
    
    # 标记基准点
    ax.axvline(x=base_capacity, color='red', linestyle='--', linewidth=1.5, alpha=0.7, 
               label=f'基准容量={base_capacity}MW')
    ax.axhline(y=base_unit_revenue, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'基准单位收益={base_unit_revenue:.4f}万元/MW/月')
    
    # 标记当前点
    ax.scatter([base_capacity], [base_unit_revenue], color='red', s=150, zorder=5, marker='*')
    
    ax.set_xlabel('总容量 (MW)', fontsize=12)
    ax.set_ylabel('单位收益 (万元/MW/月)', fontsize=12)
    ax.set_title(f'固定总收益={total_revenue_wan/10000:.2f}亿元时，单位收益随容量变化曲线', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 添加数据标签
    for i, (cap, rev) in enumerate(zip(capacities, unit_revenues)):
        ax.annotate(f'{rev:.3f}', 
                   xy=(cap, rev), 
                   xytext=(0, 10), 
                   textcoords='offset points',
                   ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # 创建汇总信息
    summary_text = f"""
    【固定总收益分析汇总】
        
    基准参数：
    - 基准容量：{base_capacity} MW
    - 总收益：{total_revenue_wan:,.2f} 万元 ({total_revenue_wan/10000:.2f} 亿元)
    - 基准单位收益：{base_unit_revenue:.4f} 万元/MW/月
        
    当前市场情况（12 个电站）：
    - 最高单位收益：{16.6735/12:.4f} 万元/MW/月 (万羚储能站)
    - 最低单位收益：{0.1021/12:.4f} 万元/MW/月 (广合储能站)
    - 加权平均单位收益：{5.6720/12:.4f} 万元/MW/月
        
    分析说明：
    当总容量增加而总收益不变时，单位收益将按反比例下降。
    容量翻倍，单位收益减半。
    """
    
    return fig, results_df, summary_text


def run_price_trend_analysis(
    base_capacity, max_demand,
    base_price, target_price,
    demand_growth_rate, capacity_growth_rate,
    learning_rate, competition_factor, policy_factor,
    forecast_years, n_sims_per_year,
    # 其他模拟参数
    dist_demand_ratio, bid_std_ratio,
    k_mean, k_std, m_mean, m_std, m_competition_factor, m_max, m_unknown_ratio,
    lambda_dis, lambda_ch,
    N_day_mean, N_day_std, d_mean, d_std,
    U_x, U_y,
    cap_ratio_upper, cap_ratio_lower, Q_lb,
    price_min, price_max,
    non_indep_capacity, non_indep_bid_mean, non_indep_k_mean
):
    """
    价格趋势分析：预测未来价格走势及对应的收益变化
    """
    # 创建价格演化模型
    price_model = PriceEvolutionModel(
        base_price=base_price,
        target_price=target_price,
        demand_growth_rate=demand_growth_rate,
        capacity_growth_rate=capacity_growth_rate,
        learning_rate=learning_rate,
        competition_factor=competition_factor,
        policy_factor=policy_factor
    )
    
    # 生成年份列表
    years = list(range(forecast_years + 1))
    
    # 生成多情景价格预测
    scenarios = price_model.generate_price_scenarios(years, base_capacity, max_demand)
    
    # 为每个情景计算收益
    scenario_results = {}
    colors = {'乐观情景（政策利好+需求快速增长）': 'green',
              '基准情景（市场自然演化）': 'blue',
              '悲观情景（竞争加剧+补贴退坡）': 'red'}
    
    for scenario_name, prices in scenarios.items():
        revenues = []
        capacities = []
        for year, price in zip(years, prices):
            # 计算该年的容量
            year_capacity = base_capacity * ((1 + capacity_growth_rate) ** year)
            capacities.append(year_capacity)
            
            # 运行蒙特卡洛模拟
            avg_revs_result, _ = monte_carlo_simulation(
                total_capacity=year_capacity,
                max_demand=max_demand,  # 需求上限固定不变
                dist_demand_ratio=dist_demand_ratio,
                bid_mean=price,
                bid_std_ratio=bid_std_ratio,
                n_sims=n_sims_per_year,
                k_mean=k_mean, k_std=k_std,
                m_mean=m_mean, m_std=m_std,
                m_competition_factor=m_competition_factor,
                m_max=m_max, m_unknown_ratio=m_unknown_ratio,
                lambda_dis=lambda_dis, lambda_ch=lambda_ch,
                N_day_mean=N_day_mean, N_day_std=N_day_std,
                d_mean=d_mean, d_std=d_std,
                U_x=U_x, U_y=U_y,
                cap_ratio_upper=cap_ratio_upper,
                cap_ratio_lower=cap_ratio_lower,
                Q_lb=Q_lb,
                price_min=price_min, price_max=price_max,
                non_indep_capacity=non_indep_capacity, non_indep_bid_mean=non_indep_bid_mean, non_indep_k_mean=non_indep_k_mean,
                return_detailed_stats=False
            )
            revenues.append(np.mean(avg_revs_result))
        
        scenario_results[scenario_name] = {
            'years': years,
            'prices': prices,
            'capacities': capacities,
            'revenues': revenues
        }
    
    # 绘制价格趋势图
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 价格趋势
    for scenario_name, data in scenario_results.items():
        ax1.plot(data['years'], data['prices'], 'o-', 
                label=scenario_name, color=colors[scenario_name], linewidth=2, markersize=6)
    ax1.axhline(y=base_price, color='gray', linestyle='--', alpha=0.5, label=f'当前价格={base_price}元/MW')
    ax1.set_xlabel('年份')
    ax1.set_ylabel('预测价格 (元/MW)')
    ax1.set_title('未来价格走势预测（多情景分析）')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 收益趋势
    for scenario_name, data in scenario_results.items():
        ax2.plot(data['years'], data['revenues'], 's-', 
                label=scenario_name, color=colors[scenario_name], linewidth=2, markersize=6)
    ax2.set_xlabel('年份')
    ax2.set_ylabel('单位收益 (万元/MW/月)')
    ax2.set_title('未来收益走势预测')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 生成结果表格
    table_data = []
    for scenario_name, data in scenario_results.items():
        for i, year in enumerate(data['years']):
            table_data.append({
                '情景': scenario_name,
                '年份': year,
                '预测价格(元/MW)': f"{data['prices'][i]:.2f}",
                '预测容量(MW)': f"{data['capacities'][i]:.0f}",
                '预测收益(万元/MW/月)': f"{data['revenues'][i]:.4f}"
            })
    
    results_df = pd.DataFrame(table_data)
    
    # 生成分析文本
    analysis_text = f"""
【价格趋势分析汇总】

基准参数：
- 当前价格：{base_price} 元/MW
- 当前容量：{base_capacity} MW
- 预测年限：{forecast_years} 年

各情景 {forecast_years} 年后预测：
"""
    for scenario_name, data in scenario_results.items():
        final_price = data['prices'][-1]
        final_revenue = data['revenues'][-1]
        price_change = ((final_price - base_price) / base_price) * 100
        analysis_text += f"\n{scenario_name}:"
        analysis_text += f"\n  - 价格：{final_price:.2f} 元/MW ({price_change:+.1f}%)"
        analysis_text += f"\n  - 收益：{final_revenue:.4f} 万元/MW/月"
    
    analysis_text += f"""

关键趋势：
1. 价格下行压力大：随着储能装机快速增长，价格普遍呈下降趋势
2. 收益分化明显：不同情景下收益差距可达 30-50%
3. 政策影响显著：政策支持力度直接影响价格下降速度

💡 核心机制说明：
- 全省调频需求上限固定为 {max_demand} MW（不随年份增长）
- 容量过剩 → M 值提升（1.45→2.0，最多 +38%）
- 价格下跌 → 出清价降低，但**中标率会提升**（低价优势）
- 综合效应 = 价格跌幅 vs (M 值提升 × 中标率提升)

⚠️ 为何某些情景下收益可能增长：
- 虽然价格下跌 50% 以上，但如果：
  a) 容量增长较慢（如乐观情景 10%/年）→ 供需比不过度恶化
  b) M 值充分提升（1.45→2.0，+38%）→ 放大单次收益
  c) 低价带来中标率大幅提升（如 60%→90%，+50%）
  d) 三者叠加：0.5(价格) × 1.38(M 值) × 1.5(中标率) = 1.035（略增）
- 这种情况下的"增长"实质是**以量补价**策略的体现
- 但悲观情景下（容量暴增 20%），收益必然大幅下降

📊 各情景解析：
- 乐观情景：容量增长慢 (10%)，价格跌幅小，M 值稳定 → 收益**持平或微增**
- 基准情景：容量增长快 (15%)，价格中等跌幅 → 收益**小幅下降 10-20%**
- 悲观情景：容量暴增 (20%)，价格暴跌 → 收益**大幅下滑 30-50%**

建议：
- 短期（1-2 年）：把握当前较高价格窗口期，尽快投产
- 中期（3-5 年）：
  • 做好价格下降 30-50% 的准备
  • 预期收益下降 10-40%（取决于竞争态势）
  • 通过提升性能指标（k/m 值）获取竞争优势
- 长期（5 年+）：
  • 成本领先战略：降低投资成本，适应低价竞争
  • 差异化战略：提升 k/m 指标，在同等报价下获得更高排序
  • 关注政策变化：若需求上限提升，可延缓收益下降趋势
"""
    
    return fig1, results_df, analysis_text


def run_price_sensitivity_analysis(
    total_capacity, max_demand, dist_demand_ratio,
    base_bid_mean, price_range_pct, price_steps,
    bid_std_ratio, n_sims,
    k_mean, k_std, m_mean, m_std, m_competition_factor, m_max, m_unknown_ratio,
    lambda_dis, lambda_ch,
    N_day_mean, N_day_std, d_mean, d_std,
    U_x, U_y,
    cap_ratio_upper, cap_ratio_lower, Q_lb,
    price_min, price_max,
    non_indep_capacity, non_indep_bid_mean, non_indep_k_mean
):
    """
    价格敏感性分析：分析不同容量和报价组合下的中标概率和收益
    新增功能：
    - 二维矩阵分析：容量 × 报价
    - 显示不同容量下的 M 值变化
    - 显示不同容量 + 报价组合下的中标概率
    - 提供最优报价策略建议
    """
    # 生成容量范围（从当前容量的 80% 到 150%）
    capacity_start = total_capacity 
    capacity_end = total_capacity * 1.5
    capacity_steps = 6
    capacities = np.linspace(capacity_start, capacity_end, capacity_steps)
    
    # 生成价格范围
    price_min_analysis = base_bid_mean * (1 - price_range_pct / 100)
    price_max_analysis = base_bid_mean * (1 + price_range_pct / 100)
    prices = np.linspace(price_min_analysis, price_max_analysis, price_steps)
    
    results = []
    
    # 定义k值范围（1.2-2.0，步长0.2）
    k_values = [1.2, 1.4, 1.6, 1.8, 2.0]
    
    # 存储不同k值的中标概率结果
    k_win_rate_results = {}
    
    for k_val in k_values:
        k_win_rate_results[k_val] = []
    
    for cap in capacities:
        # 计算该容量下的理论 M 值
        supply_demand_ratio = cap / max_demand
        if supply_demand_ratio > 1.0:
            theoretical_m = m_mean * (1 + m_competition_factor * (supply_demand_ratio - 1))
        else:
            theoretical_m = m_mean
        theoretical_m = min(theoretical_m, m_max)
        
        for price in prices:
            # 使用 target_bid 参数指定目标电站的报价，竞争对手报价围绕 base_bid_mean 分布
            avg_revs, detailed_stats = monte_carlo_simulation(
                total_capacity=cap,
                max_demand=max_demand,
                dist_demand_ratio=dist_demand_ratio,
                bid_mean=base_bid_mean,  # 竞争对手围绕基准报价分布
                bid_std_ratio=bid_std_ratio,
                n_sims=n_sims,
                k_mean=k_mean, k_std=k_std,
                m_mean=m_mean, m_std=m_std,
                m_competition_factor=m_competition_factor,
                m_max=m_max, m_unknown_ratio=m_unknown_ratio,
                lambda_dis=lambda_dis, lambda_ch=lambda_ch,
                N_day_mean=N_day_mean, N_day_std=N_day_std,
                d_mean=d_mean, d_std=d_std,
                U_x=U_x, U_y=U_y,
                cap_ratio_upper=cap_ratio_upper,
                cap_ratio_lower=cap_ratio_lower,
                Q_lb=Q_lb,
                price_min=price_min, price_max=price_max,
                non_indep_capacity=non_indep_capacity, non_indep_bid_mean=non_indep_bid_mean, non_indep_k_mean=non_indep_k_mean,
                return_detailed_stats=True,
                target_bid=price  # 目标电站使用当前分析的报价
            )
            
            # 计算中标概率和容量利用率
            if detailed_stats and len(detailed_stats['win_rates']) > 0:
                win_array = np.array(detailed_stats['win_rates'])
                avg_win_rate = np.mean(win_array)  # 中标概率（是否中标）
                cap_util_array = np.array(detailed_stats['cap_utilization'])
                avg_cap_util = np.mean(cap_util_array)  # 容量利用率（中标容量/申报容量）
            else:
                avg_win_rate = 0.0
                avg_cap_util = 0.0
            
            my_expected_rev = np.mean(detailed_stats['target_monthly_revs']) if detailed_stats and 'target_monthly_revs' in detailed_stats else np.mean(avg_revs)
            
            results.append({
                '总容量': cap,
                '报价': price,
                'M 值': theoretical_m,
                '期望收益': np.mean(avg_revs),
                '目标电站期望收益': my_expected_rev,
                '中标概率': avg_win_rate,
                '容量利用率': avg_cap_util,
                '收益标准差': np.std(avg_revs)
            })
            
            # 对不同k值分别计算中标概率
            for k_val in k_values:
                avg_revs_k, detailed_stats_k = monte_carlo_simulation(
                    total_capacity=cap,
                    max_demand=max_demand,
                    dist_demand_ratio=dist_demand_ratio,
                    bid_mean=base_bid_mean,
                    bid_std_ratio=bid_std_ratio,
                    n_sims=n_sims,
                    k_mean=k_mean, k_std=k_std,  # 竞争对手使用平均k值
                    m_mean=m_mean, m_std=m_std,
                    m_competition_factor=m_competition_factor,
                    m_max=m_max, m_unknown_ratio=m_unknown_ratio,
                    lambda_dis=lambda_dis, lambda_ch=lambda_ch,
                    N_day_mean=N_day_mean, N_day_std=N_day_std,
                    d_mean=d_mean, d_std=d_std,
                    U_x=U_x, U_y=U_y,
                    cap_ratio_upper=cap_ratio_upper,
                    cap_ratio_lower=cap_ratio_lower,
                    Q_lb=Q_lb,
                    price_min=price_min, price_max=price_max,
                    non_indep_capacity=non_indep_capacity, non_indep_bid_mean=non_indep_bid_mean, non_indep_k_mean=non_indep_k_mean,
                    return_detailed_stats=True,
                    target_bid=price,
                    target_k=k_val  # 目标电站使用特定k值
                )
                
                if detailed_stats_k and len(detailed_stats_k['win_rates']) > 0:
                    win_array_k = np.array(detailed_stats_k['win_rates'])
                    avg_win_rate_k = np.mean(win_array_k)
                else:
                    avg_win_rate_k = 0.0
                
                k_win_rate_results[k_val].append({
                    '总容量': cap,
                    '报价': price,
                    '中标概率': avg_win_rate_k
                })
    
    results_df = pd.DataFrame(results)
    
    # 绘制敏感性分析图 - 1×5 垂直排列布局，增加高度和上下间距避免重叠
    fig, axes = plt.subplots(5, 1, figsize=(16, 40))
    fig.subplots_adjust(hspace=0.5, top=0.96, bottom=0.04)
    
    # 子图 1：M 值随容量变化曲线
    unique_caps = sorted(results_df['总容量'].unique())
    m_values = [results_df[results_df['总容量'] == cap]['M 值'].iloc[0] for cap in unique_caps]
    
    ax1 = axes[0]
    ax1.plot(unique_caps, m_values, 'b-o', linewidth=2.5, markersize=10)
    ax1.axvline(x=max_demand, color='red', linestyle='--', linewidth=2, alpha=0.6,
               label=f'需求上限={max_demand:.0f}MW')
    ax1.axhline(y=m_mean, color='gray', linestyle=':', linewidth=2, alpha=0.6,
               label=f'M 基准值={m_mean}')
    ax1.axhline(y=m_max, color='orange', linestyle=':', linewidth=2, alpha=0.6,
               label=f'M 上限={m_max}')
    ax1.set_xlabel('总储能容量 (MW)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('结算性能系数 M', fontsize=14, fontweight='bold')
    ax1.set_title('【性能系数】M 值随总容量变化（容量过剩时提升）', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=12)
    
    # 标注供需比
    for i, cap in enumerate(unique_caps):
        ratio = cap / max_demand
        ax1.annotate(f'供需比={ratio:.2f}', xy=(cap, m_values[i]), xytext=(0, 12), 
                    textcoords='offset points', ha='center', fontsize=11, fontweight='bold')
    
    # 子图 2：不同容量下的中标概率热力图对比（横排）
    ax2_main = axes[1]
    ax2_main.axis('off')  # 隐藏主透明网格边框
    ax2_main.set_title('【中标概率】不同总容量下的热力图对比（性能 K × 报价）', fontsize=16, fontweight='bold', pad=30)
    
    # 汇总数据
    all_k_results = []
    for k_val, items in k_win_rate_results.items():
        for item in items:
            all_k_results.append({
                'k': k_val,
                '总容量': item['总容量'],
                '报价': item['报价'],
                '中标概率': item['中标概率']
            })
    df_all_k = pd.DataFrame(all_k_results)
    unique_caps = sorted(df_all_k['总容量'].unique())
    n_caps = len(unique_caps)
    
    gs_ax2 = ax2_main.get_subplotspec().subgridspec(1, n_caps, wspace=0.15)
    
    ax2_list = []
    for idx, cap in enumerate(unique_caps):
        ax2_sub = fig.add_subplot(gs_ax2[0, idx])
        ax2_list.append(ax2_sub)
        
        cap_df = df_all_k[df_all_k['总容量'] == cap]
        pivot_cap = cap_df.pivot_table(index='报价', columns='k', values='中标概率')
        
        # 热力图加上 origin='lower' 实现从下往上升序
        im_cap = ax2_sub.imshow(pivot_cap.values, cmap='RdYlGn_r', aspect='auto', 
                                vmin=0, vmax=1, interpolation='nearest', origin='lower')
        
        ax2_sub.set_xticks(range(len(pivot_cap.columns)))
        ax2_sub.set_yticks(range(0, len(pivot_cap.index), max(1, len(pivot_cap.index)//5)))
        
        if idx == 0:
            ax2_sub.set_yticklabels([f'{p:.1f}' for p in pivot_cap.index[::max(1, len(pivot_cap.index)//5)]], fontsize=9)
            ax2_sub.set_ylabel('报价 (元/MW)', fontsize=10, fontweight='bold')
        else:
            ax2_sub.set_yticklabels([])
            
        ax2_sub.set_xticklabels([f'{k_:.1f}' for k_ in pivot_cap.columns], fontsize=8)
        if idx == n_caps // 2:
            ax2_sub.set_xlabel('性能指标 K', fontsize=10, fontweight='bold')
            
        ratio = cap / max_demand
        ax2_sub.set_title(f'容量={cap:.0f}\n(供求比={ratio:.2f})', fontsize=11, fontweight='bold', pad=5)
        
        # 居中打印概率值
        for i in range(len(pivot_cap.index)):
            for j in range(len(pivot_cap.columns)):
                val = pivot_cap.values[i, j]
                text_color = 'white' if val > 0.5 else 'black'
                font_size = 7 if len(pivot_cap.index) > 8 else 9
                ax2_sub.text(j, i, f'{val*100:.1f}', ha='center', va='center', 
                            fontsize=font_size, fontweight='bold', color=text_color)
                            
        if idx == n_caps - 1:
            cbar = plt.colorbar(im_cap, ax=ax2_sub, fraction=0.046, pad=0.04)
            cbar.set_label('中标概率', fontsize=9, fontweight='bold')
            cbar.ax.tick_params(labelsize=8)

    # ==========================================
    # 子图 3：不同性能系数 k 下的热力图对比（横排）
    # ==========================================
    ax3_main = axes[2]
    ax3_main.axis('off')
    ax3_main.set_title('【中标概率】不同性能系数 k 下的热力图对比（容量 × 报价）', fontsize=16, fontweight='bold', pad=30)
    
    k_values = [1.2, 1.4, 1.6, 1.8, 2.0]
    gs_ax3 = ax3_main.get_subplotspec().subgridspec(1, len(k_values), wspace=0.15)
    
    ax3_list = []
    for idx, k_val in enumerate(k_values):
        ax3_sub = fig.add_subplot(gs_ax3[0, idx])
        ax3_list.append(ax3_sub)
        
        k_df = pd.DataFrame(k_win_rate_results[k_val])
        pivot_k = k_df.pivot_table(index='报价', columns='总容量', values='中标概率')
        
        # 同样加设 origin='lower' 让子图3方向一致
        im_k = ax3_sub.imshow(pivot_k.values, cmap='RdYlGn_r', aspect='auto', 
                             vmin=0, vmax=1, interpolation='nearest', origin='lower')
        
        ax3_sub.set_xticks(range(len(pivot_k.columns)))
        ax3_sub.set_yticks(range(0, len(pivot_k.index), max(1, len(pivot_k.index)//5)))
        
        if idx == 0:
            ax3_sub.set_yticklabels([f'{p:.1f}' for p in pivot_k.index[::max(1, len(pivot_k.index)//5)]], fontsize=9)
            ax3_sub.set_ylabel('报价(元/MW)', fontsize=10, fontweight='bold')
        else:
            ax3_sub.set_yticklabels([])
        
        ax3_sub.set_xticklabels([f'{c:.0f}' for c in pivot_k.columns], fontsize=8, rotation=45)
        if idx == len(k_values) // 2:
            ax3_sub.set_xlabel('总容量(MW)', fontsize=10, fontweight='bold')
        
        performance_level = ['低', '中低', '中', '中高', '高'][idx]
        ax3_sub.set_title(f'k={k_val}\n({performance_level}性能)', fontsize=11, fontweight='bold', pad=5)
        
        for i in range(len(pivot_k.index)):
            for j in range(len(pivot_k.columns)):
                val = pivot_k.values[i, j]
                text_color = 'white' if val > 0.5 else 'black'
                font_size = 7 if len(pivot_k.index) > 8 else 9
                ax3_sub.text(j, i, f'{val*100:.0f}', ha='center', va='center', 
                            fontsize=font_size, fontweight='bold', color=text_color)
        
        if idx == len(k_values) - 1:
            cbar_k = plt.colorbar(im_k, ax=ax3_sub, fraction=0.046, pad=0.04)
            cbar_k.set_label('中标概率', fontsize=9, fontweight='bold')
            cbar_k.ax.tick_params(labelsize=8)
    
    # ==========================================
    # 子图 4：期望收益热力图 (容量×报价组合)
    # ==========================================
    pivot_revenue = results_df.pivot_table(index='报价', columns='总容量', values='期望收益')
    ax4 = axes[3]
    # 给综合热力图也加上 origin='lower' 同步Y方向
    im4 = ax4.imshow(pivot_revenue.values, cmap='YlOrRd', aspect='auto',
                    interpolation='nearest', origin='lower')
    
    ax4.set_xticks(range(len(pivot_revenue.columns)))
    ax4.set_yticks(range(len(pivot_revenue.index)))
    ax4.set_xticklabels([f'{c:.0f}' for c in pivot_revenue.columns], fontsize=12)
    ax4.set_yticklabels([f'{p:.1f}' for p in pivot_revenue.index], fontsize=12)
    ax4.set_xlabel('总储能容量 (MW)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('报价 (元/MW)', fontsize=14, fontweight='bold')
    ax4.set_title('【期望收益】万元/MW/月（容量×报价组合）', fontsize=16, fontweight='bold', pad=15)
    
    for i in range(len(pivot_revenue.index)):
        for j in range(len(pivot_revenue.columns)):
            val = pivot_revenue.values[i, j]
            ax4.text(j, i, f'{val:.3f}', ha='center', va='center', 
                    fontsize=12, fontweight='bold', color='black')
    
    cbar4 = plt.colorbar(im4, ax=ax4)
    cbar4.set_label('期望收益 (万元/MW/月)', fontsize=13, fontweight='bold')
    cbar4.ax.tick_params(labelsize=11)
    ax4.grid(False)
    
    # 子图 4：最优报价策略分析
    optimal_bids = []
    optimal_returns = []
    for cap in unique_caps:
        cap_data = results_df[results_df['总容量'] == cap]
        best_idx = cap_data['期望收益'].idxmax()
        best_row = cap_data.loc[best_idx]
        optimal_bids.append(best_row['报价'])
        optimal_returns.append(best_row['期望收益'])
    
    # 计算全局最优报价和收益
    max_revenue_idx = results_df['期望收益'].idxmax()
    optimal_price = results_df.loc[max_revenue_idx, '报价']
    optimal_revenue = results_df.loc[max_revenue_idx, '期望收益']
    
    ax5 = axes[4]
    ax5.plot(unique_caps, optimal_bids, 'g-s', linewidth=2.5, markersize=10, 
            label='最优报价')
    ax5.set_xlabel('总储能容量 (MW)', fontsize=14, fontweight='bold')
    ax5.set_ylabel('最优报价 (元/MW)', fontsize=14, color='green', fontweight='bold')
    ax5.tick_params(axis='y', labelcolor='green', labelsize=12)
    ax5.tick_params(axis='x', labelsize=12)
    ax5.set_title('【最优策略】不同容量下的最优报价与对应收益', fontsize=15, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='upper left', fontsize=11)
    
    ax5b = ax5.twinx()
    ax5b.plot(unique_caps, optimal_returns, 'r-^', linewidth=2.5, markersize=10,
             label='最优收益')
    ax5b.set_ylabel('最优期望收益 (万元/MW/月)', fontsize=14, color='red', fontweight='bold')
    ax5b.tick_params(axis='y', labelcolor='red', labelsize=12)
    ax5b.legend(loc='upper right', fontsize=11)
    
    # 注意：由于第2张子图使用了手动添加的axes，tight_layout可能不兼容
    # 这里跳过tight_layout以避免警告
    current_revenue = results_df[results_df['报价'] == base_bid_mean]['期望收益'].values
    if len(current_revenue) == 0:
        # 找到最接近的
        idx = (results_df['报价'] - base_bid_mean).abs().idxmin()
        current_revenue = results_df.loc[idx, '期望收益']
    else:
        current_revenue = current_revenue[0]
    
    # 找出高价区和低价区的代表点
    low_price_mask = results_df['报价'] < results_df['报价'].quantile(0.3)
    high_price_mask = results_df['报价'] > results_df['报价'].quantile(0.7)
    
    if low_price_mask.any():
        low_price_data = results_df[low_price_mask].iloc[0]
        low_price = low_price_data['报价']
        low_win_rate = low_price_data['中标概率']
        low_exp_return = low_price_data['期望收益']
    else:
        low_price, low_win_rate, low_exp_return = None, None, None
    
    if high_price_mask.any():
        high_price_data = results_df[high_price_mask].iloc[0]
        high_price = high_price_data['报价']
        high_win_rate = high_price_data['中标概率']
        high_exp_return = high_price_data['期望收益']
    else:
        high_price, high_win_rate, high_exp_return = None, None, None
    
    analysis_text = f"""
【价格敏感性分析汇总】

当前状态：
- 报价：{base_bid_mean} 元/MW
- 期望收益：{current_revenue:.4f} 万元/MW/月

最优报价策略：
- 最优报价：{optimal_price:.2f} 元/MW
- 对应收益：{optimal_revenue:.4f} 万元/MW/月
- 相比当前：{(optimal_revenue/current_revenue - 1)*100:+.2f}%

价格弹性分析：
- 价格区间：{price_min_analysis:.2f} ~ {price_max_analysis:.2f} 元/MW
- 收益区间：{results_df['期望收益'].min():.4f} ~ {results_df['期望收益'].max():.4f} 万元/MW/月

策略建议：
"""
    
    if optimal_price < base_bid_mean:
        analysis_text += "- 当前报价偏高，适当降价可提高中标概率和收益\n"
    elif optimal_price > base_bid_mean:
        analysis_text += "- 当前报价偏低，有提价空间\n"
    else:
        analysis_text += "- 当前报价接近最优水平\n"
    
    # 新增：高价与低价策略对比分析
    if low_price and high_price:
        analysis_text += f"""
📊 高价 vs 低价策略对比：

【低价策略】（报价≈{low_price:.2f}元/MW）
✅ 优势：
  - 中标概率高：{low_win_rate*100:.1f}%，基本都能中标
  - 风险低：收益波动小，现金流稳定
  - 适合：资金紧张、需要保证基本收益的电站
❌ 劣势：
  - 单位收益低：期望收益{low_exp_return:.4f}万元/MW/月
  - 利润空间薄：难以覆盖运营成本
  - 被动竞争：容易陷入价格战

【高价策略】（报价≈{high_price:.2f}元/MW）
✅ 优势：
  - 单次收益高：中标后收益率是低价策略的{((high_price/low_price - 1)*100):.1f}%倍以上
  - M 值红利：容量过剩时 M 值提升，进一步放大收益
  - 适合：性能好（k/m 指标优）、成本低的优质电站
❌ 劣势：
  - 中标概率低：仅{high_win_rate*100:.1f}%，大部分情况下无法中标
  - 风险高：收益波动大，现金流不稳定
  - 机会成本：未中标时容量完全浪费

【最优策略】（报价≈{optimal_price:.2f}元/MW）
💡 平衡点：中标概率×中标后收益率 最大化
- 期望收益：{optimal_revenue:.4f}万元/MW/月
- 中标概率：{results_df.loc[max_revenue_idx, '中标概率']*100:.1f}%
- 风险提示：需根据实际中标概率调整
"""
    
    analysis_text += """
- 报价决策需平衡中标概率与单位收益
- 建议根据竞争对手报价动态调整

📊 中标概率指标说明：
- 中标概率：电站至少中标一部分容量的概率（0%~100%）
- 容量利用率：实际中标容量/申报容量的平均值（0%~100%）
- 当总容量过剩时，中标概率和容量利用率都会显著下降
- 高价策略会降低中标概率（排序靠后，难以中标）

💡 四子图解读：
1. 【性能系数】（第1张）：展示 M 值如何受供需比影响（与报价无关）
2. 【中标概率】（第2张）：展示不同容量×报价组合下的中标概率热力图
3. 【期望收益】（第3张）：展示不同容量×报价组合下的期望收益热力图
4. 【最优策略】（第4张）：展示不同容量下的最优报价与对应收益

🔍 核心影响因素：
- **报价水平**：直接影响排序价格和中标概率
  - 报价越低，排序价格越低，中标概率越高
  - 报价越高，排序价格越高，中标概率越低
- **总储能容量**：总容量超过需求上限 → M 值提升
  - 容量增加 → 竞争加剧 → 中标概率下降
  - 容量增加 → M 值上升 → 中标后收益放大
- **性能系数 M**：M 值提升放大收益，但不影响中标概率
- **综合效应**：期望收益 = 中标概率 × 容量利用率 × 报价 × M

💡 收益率指标解读：
- 期望收益 = 中标概率 × 容量利用率 × 报价 × M
- 中标后收益率 = 假设全额中标时的平均收益水平
- 最优报价 = 使「中标概率 × 容量利用率 × 报价 × M」最大化
"""
    
    return fig, results_df, analysis_text

#三维全空间中标概率测算
def _calc_win_rate_worker(args):
    cap, k_val, bid, base_kwargs = args
    kwargs = base_kwargs.copy()

    node_seed = hash((cap, k_val, bid)) % (2**32)
    kwargs['seed'] = node_seed

    kwargs['total_capacity'] = cap
    kwargs['target_bid'] = bid
    kwargs['target_k'] = k_val
    
    _, detailed_stats = monte_carlo_simulation(**kwargs)
    
    if detailed_stats and 'target_monthly_revs' in detailed_stats:
        win_rate = np.mean(np.array(detailed_stats['target_monthly_revs']) > 0)
        expected_rev = np.mean(detailed_stats['target_monthly_revs'])
        debug_records = detailed_stats.get('debug_records', None)
    else:
        win_rate = 0.0
        expected_rev = 0.0
        debug_records = None
        
    return {
        '容量': cap, '性能K': k_val, '报价': bid,
        '中标概率': win_rate, '期望收益': expected_rev,
        'debug_records': debug_records
    }

def calc_win_rate_matrix_data(
    cap_start, cap_end, cap_steps,
    price_start, price_end, price_steps,
    k_start, k_end, k_steps,
    n_sims, 
    max_demand, dist_demand_ratio, bid_mean, bid_std_ratio,
    k_mean, k_std, m_mean, m_std, m_competition_factor, m_max, m_unknown_ratio,
    lambda_dis, lambda_ch,
    N_day_mean, N_day_std, d_mean, d_std,
    U_x, U_y,                                                                   #边际替代率
    cap_ratio_upper, cap_ratio_lower, Q_lb,                                     #申报容量上下限。QLB
    limit_price_min, limit_price_max,
    non_indep_capacity, non_indep_bid_mean, non_indep_k_mean,
    export_target_capacity=None, export_target_bid=None, export_target_k=None
):
    """
    全解空间中标概率矩阵分析计算：
    完全遍历 总容量 × 性能K × 报价 三维空间，返回包含不同维度指标的DataFrame。
    """
    capacities = np.linspace(cap_start, cap_end, int(cap_steps))    # 仿真容量
    bids = np.linspace(price_start, price_end, int(price_steps))    # 价格
    k_values = np.linspace(k_start, k_end, int(k_steps)) # k值
    
    base_kwargs = dict(
        max_demand=max_demand, dist_demand_ratio=dist_demand_ratio,
        bid_mean=bid_mean, bid_std_ratio=bid_std_ratio, n_sims=int(n_sims),
        k_mean=k_mean, k_std=k_std, m_mean=m_mean, m_std=m_std,
        m_competition_factor=m_competition_factor, m_max=m_max, m_unknown_ratio=m_unknown_ratio,
        lambda_dis=lambda_dis, lambda_ch=lambda_ch,
        N_day_mean=N_day_mean, N_day_std=N_day_std, d_mean=d_mean, d_std=d_std,
        U_x=U_x, U_y=U_y, cap_ratio_upper=cap_ratio_upper, cap_ratio_lower=cap_ratio_lower,
        Q_lb=Q_lb, price_min=limit_price_min, price_max=limit_price_max,
        non_indep_capacity=non_indep_capacity, non_indep_bid_mean=non_indep_bid_mean, non_indep_k_mean=non_indep_k_mean,
        return_detailed_stats=True,
        export_target_capacity=export_target_capacity,
        export_target_bid=export_target_bid,
        export_target_k=export_target_k
    )
    
    tasks = [(cap, k_val, bid, base_kwargs) for cap in capacities for k_val in k_values for bid in bids]
    results = []
    debug_df = None

    cpu_count = os.cpu_count() or 4
    optimal_chunksize = max(1, len(tasks) // (cpu_count * 4))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for res in executor.map(_calc_win_rate_worker, tasks, chunksize=optimal_chunksize):
            if res.get('debug_records') is not None:
                # 当有 worker 匹配了标点条件时，提取记录转为 DataFrame
                debug_df = pd.DataFrame(res['debug_records'])
            
            # 删除由于进程传输带来的该冗余字典避免干扰正常的矩阵作图
            del res['debug_records']
            results.append(res)
            
    df = pd.DataFrame(results)
    return df, debug_df

def plot_win_rate_matrix_data(df, metric_choice):
    """
    根据给定的 df 和选择的指标，绘制热力图矩阵
    """
    if df is None or df.empty:
        return None
        
    fig = plt.figure(figsize=(16, 14))
    fig.subplots_adjust(hspace=0.4, wspace=0.15, top=0.92, bottom=0.08, left=0.06, right=0.91)
    
    title_metric = '中标概率' if metric_choice == '中标概率' else '期望收益 (万元/MW/月)'
    fig.suptitle(f'各维度交叉{title_metric}热力图', fontsize=18, fontweight='bold')
    
    k_values = sorted(df['性能K'].unique())
    capacities = sorted(df['容量'].unique())
    
    n_k = len(k_values)
    n_cap = len(capacities)
    
    # 颜色和范围配置
    cmap_str = 'RdYlGn_r' 
    vmin = 0 if metric_choice == '中标概率' else df['期望收益'].min()
    vmax = 1 if metric_choice == '中标概率' else df['期望收益'].max()
    
    # 矩阵 A：不同 K 视角下的 [容量 × 报价] 剖面
    gs1 = fig.add_gridspec(1, n_k, top=0.85, bottom=0.55)
    for i, k_val in enumerate(k_values):
        ax = fig.add_subplot(gs1[0, i])
        sub_df = df[df['性能K'] == k_val].pivot(index='报价', columns='容量', values=metric_choice)
        
        im = ax.imshow(sub_df.values, cmap=cmap_str, aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(f'K = {k_val:.2f}', fontsize=12, fontweight='bold')
        
        ax.set_xticks(range(len(sub_df.columns)))
        ax.set_xticklabels([f"{c:.0f}" for c in sub_df.columns], rotation=45, fontsize=9)
        ax.set_yticks(range(len(sub_df.index)))
        
        if i == 0:
            ax.set_yticklabels([f"{p:.1f}" for p in sub_df.index], fontsize=9)
            ax.set_ylabel('出价 (元)', fontsize=11, fontweight='bold')
        else:
            ax.set_yticklabels([])
            
        if i == n_k // 2:
            ax.set_xlabel('总电网容量 (MW)', fontsize=11, fontweight='bold')
            
        for y in range(len(sub_df.index)):
            for x in range(len(sub_df.columns)):
                val = sub_df.values[y, x]
                # Determine text color to ensure readability
                tC = 'white' if metric_choice == '中标概率' and val > 0.5 else ('white' if metric_choice != '中标概率' and val > (vmin + vmax)/2 else 'black')
                if metric_choice == '中标概率':
                    text_str = f"{val*100:.0f}"
                else:
                    text_str = f"{val:.2f}"
                ax.text(x, y, text_str, ha='center', va='center', color=tC, fontsize=8)

    # 矩阵 B：不同 容量 视角下的 [K × 报价] 剖面
    gs2 = fig.add_gridspec(1, n_cap, top=0.42, bottom=0.12)
    for i, cap in enumerate(capacities):
        ax = fig.add_subplot(gs2[0, i])
        sub_df = df[df['容量'] == cap].pivot(index='报价', columns='性能K', values=metric_choice)
        
        im2 = ax.imshow(sub_df.values, cmap=cmap_str, aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(f'总容量 = {cap:.0f} MW', fontsize=12, fontweight='bold')
        
        ax.set_xticks(range(len(sub_df.columns)))
        ax.set_xticklabels([f"{k_:.2f}" for k_ in sub_df.columns], fontsize=9)
        ax.set_yticks(range(len(sub_df.index)))
        
        if i == 0:
            ax.set_yticklabels([f"{p:.1f}" for p in sub_df.index], fontsize=9)
            ax.set_ylabel('出价 (元)', fontsize=11, fontweight='bold')
        else:
            ax.set_yticklabels([])
            
        if i == n_cap // 2:
            ax.set_xlabel('性能指标 K', fontsize=11, fontweight='bold')
            
        for y in range(len(sub_df.index)):
            for x in range(len(sub_df.columns)):
                val = sub_df.values[y, x]
                tC = 'white' if metric_choice == '中标概率' and val > 0.5 else ('white' if metric_choice != '中标概率' and val > (vmin + vmax)/2 else 'black')
                if metric_choice == '中标概率':
                    text_str = f"{val*100:.0f}"
                else:
                    text_str = f"{val:.2f}"
                ax.text(x, y, text_str, ha='center', va='center', color=tC, fontsize=8)
                
    # 全局 Colorbar
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label(title_metric, fontsize=12, fontweight='bold')
    
    return fig

def export_debug_handler(debug_df):
    """处理按钮点击，如果状态里存在 DF 就用临时盘生成给用户下发"""
    if debug_df is None or len(debug_df) == 0:
        raise gr.Error("并未匹配到调试数据！请确认您填入的三个导出目标参数位于扫描节点上并运行了矩阵生成。")
    tmp_path = os.path.join(tempfile.gettempdir(), f"模拟明细结果_{int(time.time()*1000)}.xlsx")
    debug_df.to_excel(tmp_path, index=False)
    return gr.update(value=tmp_path, visible=True)

def run_win_rate_matrix_analysis_calc(
    cap_start, cap_end, cap_steps,
    price_start, price_end, price_steps,
    k_start, k_end, k_steps,
    n_sims, 
    max_demand, dist_demand_ratio, bid_mean, bid_std_ratio,
    k_mean, k_std, m_mean, m_std, m_competition_factor, m_max, m_unknown_ratio,
    lambda_dis, lambda_ch,
    N_day_mean, N_day_std, d_mean, d_std,
    U_x, U_y,                                                                   #边际替代率
    cap_ratio_upper, cap_ratio_lower, Q_lb,                                     #申报容量上下限。QLB
    limit_price_min, limit_price_max,
    non_indep_capacity, non_indep_bid_mean, non_indep_k_mean,
    export_target_capacity=None, export_target_bid=None, export_target_k=None
):
    """运行中标概率矩阵分析的计算流程，返回中标概率和期望收益的两个 Figure"""
    df, debug_df = calc_win_rate_matrix_data(
        cap_start, cap_end, cap_steps,
        price_start, price_end, price_steps,
        k_start, k_end, k_steps,
        n_sims, 
        max_demand, dist_demand_ratio, bid_mean, bid_std_ratio,
        k_mean, k_std, m_mean, m_std, m_competition_factor, m_max, m_unknown_ratio,
        lambda_dis, lambda_ch,
        N_day_mean, N_day_std, d_mean, d_std,
        U_x, U_y, cap_ratio_upper, cap_ratio_lower, Q_lb, limit_price_min, limit_price_max,
        non_indep_capacity, non_indep_bid_mean, non_indep_k_mean,
        export_target_capacity, export_target_bid, export_target_k
    )
    
    # 自动保存为 Excel
    #excel_path = "模拟三维全空间中标结果_临时导出文件.xlsx"
    #df.to_excel(excel_path, index=False)
    
    fig_win_rate = plot_win_rate_matrix_data(df, '中标概率')
    fig_revenue = plot_win_rate_matrix_data(df, '期望收益')
    
    return fig_win_rate, fig_revenue, debug_df


# ===== 构建Gradio界面 =====
with gr.Blocks(title="储能市场竞争蒙特卡洛模拟（2025细则+优胜劣汰+容量扫描）") as demo:
    gr.HTML("""<style>footer {display:none !important;}</style>""")
    gr.Markdown(r"""
    # ⚡ 储能市场竞争蒙特卡洛模拟（完全符合2025年南方区域调频细则 + 优胜劣汰机制 + 容量扫描）
    本模型严格按照《南方区域调频辅助服务市场交易实施细则（2025年版）》设计，并引入优胜劣汰机制：
    - 当总容量超过需求上限时，结算性能系数 \(m\) 的均值按比例提高，但不超过设定的上限 \(m_{\max}\)。
    - 分离排序性能指标 \(k\) 与结算性能系数 \(m\)
    - 动态边际替代率系数 \(F_m\) 计算（依赖所有储能报价与容量占比）
    - 统一出清价格（边际价格与15元/MW取小）
    - 申报容量上下限：\(\min(P_{\text{rated}}, D_{\text{dist}} \times 20\%)\) 与 \(\min(\max(20\% P_{\text{rated}}, 5), D_{\text{dist}} \times 15\%)\)
    - 报价范围 3.5～15 元/MW
    - **默认全省调频需求上限设为1300 MW**
    """)

    with gr.Tabs():
        # 标签页1：单次模拟
        with gr.TabItem("📊 单次模拟"):
            with gr.Row():
                with gr.Column(scale=1):
                    # 容量与需求参数
                    total_cap = gr.Slider(500, 5000, 1200, step=10, label="总储能容量 (MW)")
                    max_demand = gr.Slider(500, 3000, 1300, step=50, label="全省调频需求上限 (MW)")
                    dist_demand_ratio = gr.Slider(0.1, 1.0, 1.0, step=0.05, label="本资源区需求占比")

                    gr.Markdown("### 非独立储能参数")
                    non_indep_capacity = gr.Slider(0, 3000, 520, step=10, label="非独立储能规模 (MW)")
                    non_indep_bid_mean = gr.Slider(1.0, 15.0, 5.0, step=0.1, label="非独立储能报价均值")
                    non_indep_k_mean = gr.Slider(0.5, 3.0, 1.4, step=0.05, label="非独立储能调频性能均值")

                    # 报价参数
                    bid_mean = gr.Number(value=11.0, label="报价均值 (元/MW)")
                    bid_std_ratio = gr.Slider(0.1, 1.0, 0.3, step=0.05, label="报价波动系数")
                    price_min = gr.Number(value=3.5, label="报价下限 (元/MW)")
                    price_max = gr.Number(value=15.0, label="报价上限 (元/MW)")

                    gr.Markdown("### 性能指标")
                    k_mean = gr.Slider(0.5, 3.0, 1.8, step=0.05, label="排序性能指标 k 均值")
                    k_std = gr.Slider(0.05, 0.5, 0.2, step=0.01, label="排序性能指标 k 标准差")
                    m_mean = gr.Slider(1.2, 2.0, 1.45, step=0.01, label="结算性能系数 m 基准均值")
                    m_std = gr.Slider(0.05, 0.2, 0.1, step=0.01, label="结算性能系数 m 标准差")
                    m_unknow_ratio = gr.Slider(0.2, 1.0, 0.9, step=0.05, label="m值 确定系数")
                    m_competition_factor = gr.Slider(0.0, 2.0, 0.5, step=0.05, label="优胜劣汰增强系数 γ",
                                                     info="γ>0时，容量超过需求越多，m均值提高越快")
                    m_max = gr.Slider(1.3, 2.0, 2.0, step=0.05, label="结算性能系数 m 上限",
                                      info="m 值不能超过此上限，默认 2.0（符合细则）")

                    gr.Markdown("### 调频运营参数")
                    N_day_mean = gr.Slider(50, 500, 300, step=10, label="日响应次数均值")
                    N_day_std = gr.Slider(10, 200, 10, step=10, label="日响应次数标准差")
                    d_mean = gr.Slider(0.3, 0.8, 0.5, step=0.05, label="调节深度均值")
                    d_std = gr.Slider(0.05, 0.3, 0.05, step=0.01, label="调节深度标准差")
                    lambda_dis = gr.Slider(0.01, 0.5, 0.3206, step=0.01, label="放电均价 ")
                    lambda_ch = gr.Slider(0.01, 0.5, 0.3051, step=0.01, label="充电均价 ")

                    gr.Markdown("### 边际替代率曲线")
                    U_x = gr.Slider(0.1, 1.0, 0.6, step=0.05, label="U_x (容量占比)")
                    U_y = gr.Slider(1.0, 4.0, 2.5, step=0.1, label="U_y (最大替代系数)")

                    gr.Markdown("### 申报容量约束")
                    cap_ratio_upper = gr.Slider(0.05, 0.5, 0.2, step=0.01, label="申报上限比例")
                    cap_ratio_lower = gr.Slider(0.05, 0.5, 0.15, step=0.01, label="申报下限比例")
                    Q_lb = gr.Number(value=5.0, label="保底容量 Q_lb (MW)")

                    n_sims_single = gr.Slider(100, 5000, 2000, step=100, label="模拟次数")
                    run_btn = gr.Button("🚀 运行模拟", variant="primary")

                with gr.Column(scale=2):
                    plot_output = gr.Plot(label="平均收益分布")
                    stats_output = gr.Dataframe(label="统计结果")

        # 标签页 2：容量扫描
        with gr.TabItem("📈 容量扫描分析"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(r"""
                    ### 📊 场景说明
                            
                    **市场背景：**
                    - 全省调频需求上限固定（默认 1300 MW）
                    - 储能参与者增多 → 总装机容量增加
                    - 供过于求 → 启动优胜劣汰机制
                            
                    **核心逻辑：**
                    1. 当总容量 > 需求上限时，M 值上升
                    2. 竞争加剧导致报价下降
                    3. M 值上升部分抵消价格下降影响
                    4. 最终单位收益变化取决于两者博弈
                            
                    **预期结果：**
                    - 短期：M 值提升缓冲收益下降
                    - 中期：收益加速下滑
                    - 长期：M 值达上限后线性下降
                    """)
                            
                    gr.Markdown("### 🔧 扫描范围")
                    cap_start = gr.Number(value=1000, label="起始容量 (MW)",
                                         info="当前市场总容量约 1280MW")
                    cap_end = gr.Number(value=3000, label="结束容量 (MW)",
                                       info="预测未来最大容量")
                    cap_step = gr.Number(value=200, label="步长 (MW)",
                                        info="容量递增间隔")
        
                    gr.Markdown("### ⚙️ 扫描模拟次数")
                    scan_n_sims = gr.Slider(100, 2000, 500, step=50, label="每个容量点的模拟次数",
                                           info="建议 500 次以上以保证统计显著性")
        
                    gr.Markdown(r"""
                    ### 📝 参数说明
                    使用左侧「单次模拟」标签页中的市场参数：
                    - 报价均值：11.0 元/MW
                    - m 基准均值：1.45
                    - 优胜劣汰增强系数γ：0.5
                    - m 上限：2.0
                    - 全省调频需求上限：1300 MW
                    """)
        
                    scan_btn = gr.Button("📉 运行扫描", variant="primary")
        
                with gr.Column(scale=2):
                    scan_plot1 = gr.Plot(label="收益 - 价格 - 容量综合曲线")
                    scan_plot2 = gr.Plot(label="M 值变化曲线")
                    scan_table = gr.Dataframe(label="收益下降量统计表")
                    scan_summary = gr.Textbox(label="📊 关键结论解读", lines=22, visible=True)

        # 标签页3：固定总收益分析
        with gr.TabItem("💰 固定总收益分析"):
            gr.Markdown(r"""
            ### 固定总收益分析
            基于实际市场数据：调频总收益 **8.11亿元**，当前参与电站总容量 **1280MW**。
            在总收益不变的情况下，分析调频电站总容量增加后，单位收益的变化情况。
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 基准参数")
                    fixed_base_capacity = gr.Number(value=1280, label="基准容量 (MW)", 
                                                    info="当前12个电站总容量")
                    fixed_total_revenue = gr.Number(value=81100, label="总收益 (万元)", 
                                                   info="8.11亿元 = 81100万元")
                    
                    gr.Markdown("### 扫描范围")
                    fixed_cap_start = gr.Number(value=1000, label="起始容量 (MW)")
                    fixed_cap_end = gr.Number(value=3000, label="结束容量 (MW)")
                    fixed_cap_step = gr.Number(value=200, label="步长 (MW)")
                    
                    gr.Markdown("### 其他参数")
                    gr.Markdown("使用左侧「单次模拟」标签页中的市场参数（需求上限、报价参数等）")
                    
                    fixed_revenue_btn = gr.Button("📊 运行固定收益分析", variant="primary")
                
                with gr.Column(scale=2):
                    fixed_plot = gr.Plot(label="单位收益随容量变化曲线")
                    fixed_table = gr.Dataframe(label="分析结果表")
                    fixed_summary = gr.Textbox(label="汇总信息", lines=12)

        # 标签页4：价格趋势预测
        with gr.TabItem("📉 价格趋势预测"):
            gr.Markdown(r"""
            ### 储能调频市场价格趋势预测
            基于供需关系、技术进步、竞争态势和政策环境，预测未来价格走势。
            当前价格约 **11元/MW**，预计未来 5 年可能下降至 **5-8元/MW**。
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 价格演化参数")
                    trend_base_price = gr.Number(value=11.0, label="当前基准价格 (元/MW)")
                    trend_target_price = gr.Number(value=5.0, label="长期均衡价格目标 (元/MW)")
                    
                    gr.Markdown("### 市场增长参数")
                    trend_demand_growth = gr.Slider(0.0, 0.10, 0.02, step=0.01, 
                                                   label="需求年增长率", 
                                                   info="调频需求年增长预期")
                    trend_capacity_growth = gr.Slider(0.05, 0.30, 0.15, step=0.01,
                                                     label="储能容量年增长率",
                                                     info="储能装机年增长预期")
                    
                    gr.Markdown("### 价格影响因素")
                    trend_learning_rate = gr.Slider(0.0, 0.10, 0.05, step=0.01,
                                                   label="学习效应系数",
                                                   info="技术进步导致的成本下降速度")
                    trend_competition = gr.Slider(0.0, 0.30, 0.10, step=0.01,
                                                 label="竞争加剧系数",
                                                 info="参与者增多对价格的压制")
                    trend_policy = gr.Slider(-0.10, 0.10, 0.0, step=0.01,
                                            label="政策调整因子",
                                            info="正值表示政策支持，负值表示补贴退坡")
                    
                    gr.Markdown("### 预测设置")
                    trend_forecast_years = gr.Slider(1, 10, 5, step=1, label="预测年限")
                    trend_n_sims = gr.Slider(100, 1000, 300, step=50, 
                                            label="每年模拟次数")
                    
                    gr.Markdown("### 其他参数")
                    gr.Markdown("使用「单次模拟」标签页中的市场参数")
                    
                    trend_btn = gr.Button("📈 运行价格趋势预测", variant="primary")
                
                with gr.Column(scale=2):
                    trend_plot = gr.Plot(label="价格与收益趋势预测")
                    trend_table = gr.Dataframe(label="预测结果表")
                    trend_summary = gr.Textbox(label="分析汇总", lines=15)

        # 标签页5：价格敏感性分析
        with gr.TabItem("🔍 价格敏感性分析"):
            gr.Markdown(r"""
            ### 报价策略敏感性分析
            分析不同报价水平对收益的影响，找出最优报价策略。
            帮助决策：在当前市场环境下，应该报高价还是低价？
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 分析参数")
                    sens_base_price = gr.Number(value=11.0, label="基准报价 (元/MW)")
                    sens_price_range = gr.Slider(10, 50, 30, step=5,
                                                label="价格浮动范围 (%)",
                                                info="相对于基准价的上下浮动百分比")
                    sens_price_steps = gr.Slider(10, 40, 20, step=2,
                                                label="价格分析点数")
                    sens_n_sims = gr.Slider(100, 5000, 2000, step=100,
                                           label="模拟次数")
                    
                    gr.Markdown("### 其他参数")
                    gr.Markdown("使用「单次模拟」标签页中的市场参数")
                    
                    sens_btn = gr.Button("🔍 运行敏感性分析", variant="primary")
                
                with gr.Column(scale=2):
                    sens_plot = gr.Plot(label="价格敏感性分析")
                    sens_table = gr.Dataframe(label="分析结果表")
                    sens_summary = gr.Textbox(label="策略建议", lines=12)
        
        
        #标签页7：三维全空间中标概率测算
        with gr.Tab("🎲 三维全空间中标概率测算"):
            gr.Markdown("自定义遍历【总申报容量】×【电站性能K】×【申报报价】三个维度的正交空间，生成各切面的指标热力图。")
            
            with gr.Row():
                with gr.Column():
                    grid_cap_start = gr.Number(value=750, label="起步仿真容量(MW)")
                    grid_cap_end = gr.Number(value=1500, label="终端仿真容量(MW)")
                    grid_cap_steps = gr.Slider(3, 8, 5, step=1, label="容量插值列数")
                with gr.Column():
                    grid_price_start = gr.Number(value=3.5, label="最低测算报价")
                    grid_price_end = gr.Number(value=15.0, label="最高测算报价")
                    grid_price_steps = gr.Slider(5, 20, 8, step=1, label="报价插值行数")
                with gr.Column():
                    grid_k_start = gr.Number(value=0.8, label="最低K值")
                    grid_k_end = gr.Number(value=3.0, label="最高K值")
                    grid_k_steps = gr.Slider(3, 8, 6, step=1, label="K值插值列数")
                with gr.Column():
                    grid_bid_mean = gr.Number(value=11.0, label="对手报价均值 (元/MW)")
                    grid_k_mean = gr.Slider(0.5, 3.0, 1.8, step=0.05, label="对手排序性能指标 k 均值")
                    grid_n_sims = gr.Slider(100, 5000, 2000, step=100, label="模拟次数")
                with gr.Column():
                    grid_lambda_dis = gr.Slider(0.01, 0.5, 0.3206, step=0.01, label="放电均价 ")
                    grid_lambda_ch = gr.Slider(0.01, 0.5, 0.3051, step=0.01, label="充电均价 ")
                    grid_m_unknow = gr.Slider(0.5, 1.0, 0.9, step=0.05, label="M值不确定系数")
                with gr.Column():
                    #gr.Markdown("### 非独立储能参数")
                    non_indep_capacity = gr.Slider(0, 10000, 5000, step=10, label="非独立储能规模 (MW)")
                    non_indep_bid_mean = gr.Slider(3.5, 15.0, 4.5, step=0.1, label="非独立储能报价均值")
                    non_indep_k_mean = gr.Slider(0.8, 2.0, 1.4, step=0.05, label="非独立储能排序性能k均值")
                with gr.Column():
                    grid_d_mean = gr.Slider(0.3, 0.8, 0.5, step=0.05, label="调节深度均值")
                    grid_max_demand = gr.Slider(500, 3000, 1300, step=50, label="全省调频需求上限 (MW)")
                    
            
            gr.Markdown("可选参数：需要导出**目标场景**细节的明细结果时，请填写下列三个条件")
            with gr.Row():
                export_target_capacity_ui = gr.Number(value=None, label="导出目标总容量(MW)")
                export_target_bid_ui = gr.Number(value=None, label="导出目标报价(元/MW)")
                export_target_k_ui = gr.Number(value=None, label="导出目标K值")
            
            with gr.Row():
                gen_grid_btn = gr.Button("生成热力图", variant="primary")

            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=150):
                    download_debug_btn = gr.Button("⬇️ 生成并下载调试结果", size="sm")
                    down_file_card = gr.File(label="获取的详细记录：", interactive=False, visible=False)
                with gr.Column(scale=5):
                    gr.Markdown(" ")
                
            # 用于缓存生成的那个 DataFrame 
            grid_debug_state = gr.State(None)
            
            grid_plot_win_rate = gr.Plot(label="中标概率热力图")
            grid_plot_revenue = gr.Plot(label="期望收益热力图")
        
    

    # 绑定单次模拟按钮
    run_btn.click(
        fn=run_single_simulation,
        inputs=[
            total_cap, max_demand, dist_demand_ratio,
            bid_mean, bid_std_ratio, n_sims_single,
            k_mean, k_std, m_mean, m_std, m_competition_factor, m_max, m_unknow_ratio,
            lambda_dis, lambda_ch,
            N_day_mean, N_day_std, d_mean, d_std,
            U_x, U_y,
            cap_ratio_upper, cap_ratio_lower, Q_lb,
            price_min, price_max,
            non_indep_capacity, non_indep_bid_mean, non_indep_k_mean
        ],
        outputs=[plot_output, stats_output]
    )

    # 绑定扫描按钮
    scan_btn.click(
        fn=run_capacity_scan,
        inputs=[
            cap_start, cap_end, cap_step, scan_n_sims,
            max_demand, dist_demand_ratio,
            bid_mean, bid_std_ratio,
            k_mean, k_std, m_mean, m_std, m_competition_factor, m_max, m_unknow_ratio,
            lambda_dis, lambda_ch,
            N_day_mean, N_day_std, d_mean, d_std,
            U_x, U_y,
            cap_ratio_upper, cap_ratio_lower, Q_lb,
            price_min, price_max,
            non_indep_capacity, non_indep_bid_mean, non_indep_k_mean
        ],
        outputs=[scan_plot1, scan_plot2, scan_table, scan_summary]
    )

    # 绑定固定总收益分析按钮
    fixed_revenue_btn.click(
        fn=run_fixed_revenue_analysis,
        inputs=[
            fixed_base_capacity, fixed_total_revenue,
            fixed_cap_start, fixed_cap_end, fixed_cap_step,
            max_demand, dist_demand_ratio,
            bid_mean, bid_std_ratio,
            k_mean, k_std, m_mean, m_std, m_competition_factor, m_max, m_unknow_ratio,
            lambda_dis, lambda_ch,
            N_day_mean, N_day_std, d_mean, d_std,
            U_x, U_y,
            cap_ratio_upper, cap_ratio_lower, Q_lb,
            price_min, price_max,
            non_indep_capacity, non_indep_bid_mean, non_indep_k_mean
        ],
        outputs=[fixed_plot, fixed_table, fixed_summary]
    )

    # 绑定价格趋势预测按钮
    trend_btn.click(
        fn=run_price_trend_analysis,
        inputs=[
            total_cap, max_demand,
            trend_base_price, trend_target_price,
            trend_demand_growth, trend_capacity_growth,
            trend_learning_rate, trend_competition, trend_policy,
            trend_forecast_years, trend_n_sims,
            dist_demand_ratio, bid_std_ratio,
            k_mean, k_std, m_mean, m_std, m_competition_factor, m_max, m_unknow_ratio,
            lambda_dis, lambda_ch,
            N_day_mean, N_day_std, d_mean, d_std,
            U_x, U_y,
            cap_ratio_upper, cap_ratio_lower, Q_lb,
            price_min, price_max,
            non_indep_capacity, non_indep_bid_mean, non_indep_k_mean
        ],
        outputs=[trend_plot, trend_table, trend_summary]
    )

    # 绑定价格敏感性分析按钮
    sens_btn.click(
        fn=run_price_sensitivity_analysis,
        inputs=[
            total_cap, max_demand, dist_demand_ratio,
            sens_base_price, sens_price_range, sens_price_steps,
            bid_std_ratio, sens_n_sims,
            k_mean, k_std, m_mean, m_std, m_competition_factor, m_max, m_unknow_ratio,
            lambda_dis, lambda_ch,
            N_day_mean, N_day_std, d_mean, d_std,
            U_x, U_y,
            cap_ratio_upper, cap_ratio_lower, Q_lb,
            price_min, price_max,
            non_indep_capacity, non_indep_bid_mean, non_indep_k_mean
        ],
        outputs=[sens_plot, sens_table, sens_summary]
    )

    # 绑定生成三维中标率切面矩阵按钮
    gen_grid_btn.click(
        fn=run_win_rate_matrix_analysis_calc,
        inputs=[
            grid_cap_start, grid_cap_end, grid_cap_steps,                 #容量起始值、终端值、插值列。
            grid_price_start, grid_price_end, grid_price_steps,
            grid_k_start, grid_k_end, grid_k_steps,
            grid_n_sims,                                                  # 复用之前的基础模拟次数变量即可
            grid_max_demand, dist_demand_ratio, grid_bid_mean, bid_std_ratio,
            grid_k_mean, k_std, m_mean, m_std, m_competition_factor, m_max, grid_m_unknow,
            grid_lambda_dis, grid_lambda_ch,
            N_day_mean, N_day_std, grid_d_mean, d_std,
            U_x, U_y, cap_ratio_upper, cap_ratio_lower, Q_lb, price_min, price_max,
            non_indep_capacity, non_indep_bid_mean, non_indep_k_mean,
            export_target_capacity_ui, export_target_bid_ui, export_target_k_ui
        ],
        outputs=[grid_plot_win_rate, grid_plot_revenue, grid_debug_state]
    )
    download_debug_btn.click(
        fn=export_debug_handler,
        inputs=[grid_debug_state],    
        outputs=[down_file_card]     
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", theme=gr.themes.Soft(),server_port=7861,share=True)
    #demo.launch(theme=gr.themes.Soft(),share=True)