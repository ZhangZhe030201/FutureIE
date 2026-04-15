from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

SIM_THRESHOLD = 0.7

tqdm.pandas()


def _normalize_text(text: object) -> str:
    if not isinstance(text, str):
        return str(text)
    return text.lower().replace("-", " ").replace("_", " ").strip()


def _safe_extract_entities(triple_str: object) -> Optional[list[str]]:
    try:
        triples = ast.literal_eval(str(triple_str))

        if not isinstance(triples, list):
            return None

        ents: list[str] = []

        for item in triples:
            if isinstance(item, list) and len(item) > 0 and isinstance(item[0], list):
                ent = _normalize_text(" ".join(item[0]))
                if ent:
                    ents.append(ent)

        return ents if ents else None
    except Exception:
        return None


def analyze_entity_newness_tfidf_pro_sim(data_csv: str, mapping_csv: str, output_csv: str, logger=None) -> None:
    if logger is not None:
        logger.info("========== Step1: 加载实体年份映射表 ==========")

    map_df = pd.read_csv(mapping_csv)

    map_df["year"] = pd.to_numeric(map_df["year"], errors="coerce")
    map_df = map_df.dropna(subset=["year", "entity"])
    map_df["clean_key"] = map_df["entity"].apply(_normalize_text)
    map_df = map_df.sort_values("year")

    entity_year_map = (
        map_df.drop_duplicates(subset=["clean_key"], keep="first").set_index("clean_key")["year"].to_dict()
    )
    mapping_keys = list(entity_year_map.keys())

    if logger is not None:
        logger.info("映射表实体数量: %s", len(mapping_keys))
        logger.info("========== Step2: 读取数据 ==========")

    df = pd.read_csv(data_csv)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    if logger is not None:
        logger.info("解析实体中...")

    df["fws_entities"] = df["fws_triple"].progress_apply(_safe_extract_entities)
    df["abs_entities"] = df["abstract_triple"].progress_apply(_safe_extract_entities)

    df_clean = df.dropna(subset=["fws_entities", "abs_entities", "year"]).copy()

    if logger is not None:
        logger.info("有效论文数: %s", len(df_clean))
        logger.info("========== Step3: 实体匹配 ==========")

    all_fws = [e for row in df_clean["fws_entities"] for e in row]
    all_abs = [e for row in df_clean["abs_entities"] for e in row]
    unique_entities = list(set(all_fws + all_abs))

    cache_entity_year: dict[str, float] = {}
    missing_entities: list[str] = []

    for ent in unique_entities:
        if ent in entity_year_map:
            cache_entity_year[ent] = entity_year_map[ent]
        else:
            missing_entities.append(ent)

    if logger is not None:
        logger.info("精确匹配: %s", len(cache_entity_year))
        logger.info("待模糊匹配: %s", len(missing_entities))

    fuzzy_match_details: dict[str, str] = {}

    if missing_entities:
        vectorizer = TfidfVectorizer(min_df=1, analyzer="char_wb", ngram_range=(2, 3))
        vectorizer.fit(mapping_keys)

        tfidf_candidates = vectorizer.transform(mapping_keys)
        tfidf_queries = vectorizer.transform(missing_entities)

        nbrs = NearestNeighbors(n_neighbors=1, metric="cosine", n_jobs=-1).fit(tfidf_candidates)
        dists, idxs = nbrs.kneighbors(tfidf_queries)

        if logger is not None:
            logger.info("开始模糊匹配...")

        for i, ent in enumerate(tqdm(missing_entities)):
            dist = dists[i][0]
            idx = idxs[i][0]
            best_match = mapping_keys[idx]
            similarity = 1.0 - dist

            if similarity >= SIM_THRESHOLD:
                cache_entity_year[ent] = entity_year_map[best_match]
                fuzzy_match_details[ent] = f"['{ent}', '{best_match}', {similarity:.4f}]"

        if logger is not None:
            logger.info("有效模糊匹配: %s", len(fuzzy_match_details))

    if logger is not None:
        logger.info("========== Step4: 计算年龄 ==========")

    fws_avg_list = []
    abs_avg_list = []
    sim_list = []

    iterator = tqdm(zip(df_clean["fws_entities"], df_clean["abs_entities"], df_clean["year"]), total=len(df_clean))

    for fws_ents, abs_ents, year in iterator:
        sims = []
        fws_ages = []
        abs_ages = []

        for ent in fws_ents:
            start_year = cache_entity_year.get(ent)
            if start_year is not None:
                age = max(0, year - start_year)
                fws_ages.append(age)
            if ent in fuzzy_match_details:
                sims.append(fuzzy_match_details[ent])

        for ent in abs_ents:
            start_year = cache_entity_year.get(ent)
            if start_year is not None:
                age = max(0, year - start_year)
                abs_ages.append(age)
            if ent in fuzzy_match_details:
                sims.append(fuzzy_match_details[ent])

        fws_avg = np.mean(fws_ages) if fws_ages else None
        abs_avg = np.mean(abs_ages) if abs_ages else None

        fws_avg_list.append(fws_avg)
        abs_avg_list.append(abs_avg)
        sim_list.append("; ".join(sorted(set(sims))))

    df_clean["fws_avg_age"] = fws_avg_list
    df_clean["abs_avg_age"] = abs_avg_list
    df_clean["sim"] = sim_list

    if logger is not None:
        logger.info("========== Step5: 保存结果 ==========")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False, encoding="utf-8-sig")

    if logger is not None:
        logger.info("输出文件: %s", output_path)


def final_comprehensive_statistics(input_csv: str, logger=None) -> None:
    if logger is not None:
        logger.info("正在读取文件: %s ...", input_csv)

    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        if logger is not None:
            logger.error("错误: 找不到文件 %s，请确保上一步已执行成功。", input_csv)
        return

    fws_series = df["fws_avg_age"]
    abs_series = df["abs_avg_age"]

    stats_data = {
        "统计项": ["有效样本数 (Count)", "平均年份差 (Mean)", "中位数年份差 (Median)", "标准差 (Std)"],
        "fws_triple (全文)": [
            fws_series.count(),
            fws_series.mean(),
            fws_series.median(),
            fws_series.std(),
        ],
        "abstract_triple (摘要)": [
            abs_series.count(),
            abs_series.mean(),
            abs_series.median(),
            abs_series.std(),
        ],
    }

    stats_df = pd.DataFrame(stats_data)
    pd.options.display.float_format = "{:.2f}".format

    if logger is not None:
        logger.info("%s", "=" * 40)
        logger.info("       实体新旧程度综合统计报告")
        logger.info("%s", "=" * 40)
        logger.info("注：'年份差' = 论文发表年份 - 实体最早出现年份")
        logger.info("    数值越【小】，说明实体越【新】。")
        logger.info("\n%s", stats_df.to_string(index=False))
        logger.info("%s", "-" * 40)

    mean_fws = fws_series.mean()
    mean_abs = abs_series.mean()

    if logger is not None:
        logger.info("【最终结论】:")

    if pd.isna(mean_fws) or pd.isna(mean_abs):
        if logger is not None:
            logger.info("数据不足，无法计算平均值进行比较。")
    else:
        diff = abs(mean_fws - mean_abs)
        if mean_fws < mean_abs:
            if logger is not None:
                logger.info("🏆 [fws_triple] 胜出！")
                logger.info("原因: 全文提取实体的平均“年龄差”为 %.2f 年，", mean_fws)
                logger.info("      比摘要提取实体 (%.2f 年) 更小。", mean_abs)
                logger.info("      这意味着全文中包含了更多较新的技术或概念 (平均新 %.2f 年)。", diff)
        elif mean_abs < mean_fws:
            if logger is not None:
                logger.info("🏆 [abstract_triple] 胜出！")
                logger.info("原因: 摘要提取实体的平均“年龄差”为 %.2f 年，", mean_abs)
                logger.info("      比全文提取实体 (%.2f 年) 更小。", mean_fws)
                logger.info("      这意味着摘要中包含了更多较新的技术或概念 (平均新 %.2f 年)。", diff)
        else:
            if logger is not None:
                logger.info("平局。两列实体的平均新旧程度完全一致。")


def run_multiple_statistical_tests(input_csv: str, logger=None) -> None:
    if logger is not None:
        logger.info("正在读取数据: %s ...", input_csv)

    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        if logger is not None:
            logger.error("错误: 找不到文件 %s", input_csv)
        return

    data = df[["fws_avg_age", "abs_avg_age"]].dropna()
    n = len(data)

    if logger is not None:
        logger.info("有效成对样本数 (N): %s", n)

    if n < 5:
        if logger is not None:
            logger.info("样本太少，无法进行统计检验。")
        return

    diff = data["fws_avg_age"] - data["abs_avg_age"]

    shapiro_stat, shapiro_p = stats.shapiro(diff)
    dist_desc = "符合正态分布" if shapiro_p > 0.05 else "不符合正态分布"

    if logger is not None:
        logger.info("%s", "=" * 60)
        logger.info("数据分布检查 (Shapiro-Wilk): p=%.4e -> %s", shapiro_p, dist_desc)
        logger.info("%s", "=" * 60)

    results = []

    t_stat, p_t = stats.ttest_rel(data["fws_avg_age"], data["abs_avg_age"])
    mean_diff = diff.mean()
    std_diff = diff.std()
    cohens_d = mean_diff / std_diff if std_diff != 0 else 0
    results.append(
        {
            "检验名称": "配对 T 检验 (均值)",
            "统计量": f"t={t_stat:.3f}",
            "P值": p_t,
            "效应量": f"d={cohens_d:.3f}",
            "结论": "显著" if p_t < 0.05 else "不显著",
        }
    )

    w_stat, p_w = stats.wilcoxon(data["fws_avg_age"], data["abs_avg_age"])
    results.append(
        {
            "检验名称": "Wilcoxon 符号秩检验 (中位数)",
            "统计量": f"W={w_stat:.1f}",
            "P值": p_w,
            "效应量": "-",
            "结论": "显著" if p_w < 0.05 else "不显著",
        }
    )

    n_positive = np.sum(diff > 0)
    n_negative = np.sum(diff < 0)
    n_ties = np.sum(diff == 0)
    binom_res = stats.binomtest(n_positive, n - n_ties, p=0.5)
    p_sign = binom_res.pvalue
    win_rate = (n_negative / (n - n_ties)) if (n - n_ties) != 0 else np.nan
    results.append(
        {
            "检验名称": "符号检验 (方向胜率)",
            "统计量": f"Pos={n_positive}/Neg={n_negative}",
            "P值": p_sign,
            "效应量": f"Win%={win_rate:.1%}(FWS)" if not np.isnan(win_rate) else "Win%=nan(FWS)",
            "结论": "显著" if p_sign < 0.05 else "不显著",
        }
    )

    res_df = pd.DataFrame(results)
    if logger is not None:
        logger.info("综合统计检验报告:")
        logger.info("\n%s", res_df.to_string(index=False))
        logger.info("%s", "-" * 60)
        logger.info("【详细解读】")

    alpha = 0.05
    significant_count = sum(1 for p in [p_t, p_w, p_sign] if p < alpha)

    if significant_count >= 2:
        if logger is not None:
            logger.info("✅ 结论非常稳健：多种检验均表明两组数据存在统计学显著差异。")
            if diff.median() < 0:
                logger.info("   方向：[fws_triple] (全文) 包含的实体显著更“新”。")
            else:
                logger.info("   方向：[abstract_triple] (摘要) 包含的实体显著更“新”。")
    elif significant_count == 0:
        if logger is not None:
            logger.info("❌ 结论：两组数据没有统计学显著差异。")
    else:
        if logger is not None:
            logger.info("⚠️ 结论存在分歧：部分检验显著，部分不显著。")
            logger.info("   建议以 [Wilcoxon] 检验结果为准 (因为它对非正态分布更稳健)。")

    plt.figure(figsize=(10, 6))
    sns.histplot(diff, kde=True, bins=30, color="purple")
    plt.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero Difference")
    plt.axvline(x=diff.mean(), color="blue", linestyle="-", linewidth=2, label=f"Mean Diff ({diff.mean():.2f})")
    plt.title("Distribution of Age Differences (Full Text - Abstract)")
    plt.xlabel("Difference in Years (Negative = Full Text is Newer)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_dir = os.path.dirname(input_csv)
    out_img_path = os.path.join(output_dir, "statistical_tests_distribution.png")
    plt.savefig(out_img_path)
    plt.close()

    if logger is not None:
        logger.info("分布图已保存至: %s", out_img_path)


def run_additional_statistical_tests(input_csv: str, logger=None) -> None:
    if logger is not None:
        logger.info("正在读取数据: %s ...", input_csv)

    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        if logger is not None:
            logger.error("错误: 找不到文件 %s", input_csv)
        return

    data = df[["fws_avg_age", "abs_avg_age"]].dropna()
    n = len(data)

    if logger is not None:
        logger.info("有效成对样本数 (N): %s", n)

    if n < 2:
        if logger is not None:
            logger.info("样本不足，无法计算。")
        return

    diff = data["fws_avg_age"] - data["abs_avg_age"]
    results = []

    n_fws_newer = np.sum(diff < 0)
    n_abs_newer = np.sum(diff > 0)
    observed = [n_fws_newer, n_abs_newer]
    expected = [(n_fws_newer + n_abs_newer) / 2] * 2
    chi2_stat, p_chi2 = stats.chisquare(f_obs=observed, f_exp=expected)
    results.append(
        {
            "检验类型": "卡方检验 (Chi-Square)",
            "检验对象": "胜负比例 (Counts)",
            "统计量": f"χ²={chi2_stat:.3f}",
            "P值": p_chi2,
            "备注": f"FWS胜:{n_fws_newer} vs ABS胜:{n_abs_newer}",
        }
    )

    mean_d = diff.mean()
    std_d = diff.std(ddof=1)
    std_error = std_d / np.sqrt(n)
    z_stat = mean_d / std_error
    p_z = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    results.append(
        {
            "检验类型": "Z 检验 (Paired Z-test)",
            "检验对象": "均值差异 (Means)",
            "统计量": f"Z={z_stat:.3f}",
            "P值": p_z,
            "备注": f"均值差={mean_d:.3f} (负数代表FWS更新)",
        }
    )

    var_fws = np.var(data["fws_avg_age"], ddof=1)
    var_abs = np.var(data["abs_avg_age"], ddof=1)
    if var_fws > var_abs:
        f_stat = var_fws / var_abs
        df1 = n - 1
        df2 = n - 1
        variance_msg = "FWS方差 > ABS方差"
    else:
        f_stat = var_abs / var_fws
        df1 = n - 1
        df2 = n - 1
        variance_msg = "ABS方差 > FWS方差"

    p_f = 2 * (1 - stats.f.cdf(f_stat, df1, df2))
    results.append(
        {
            "检验类型": "F 检验 (F-test)",
            "检验对象": "方差/稳定性 (Variances)",
            "统计量": f"F={f_stat:.3f}",
            "P值": p_f,
            "备注": variance_msg,
        }
    )

    res_df = pd.DataFrame(results)
    if logger is not None:
        logger.info("%s", "=" * 80)
        logger.info("                 三项额外统计检验报告")
        logger.info("%s", "=" * 80)
        pd.set_option("display.max_colwidth", None)
        logger.info("\n%s", res_df.to_string(index=False))
        logger.info("%s", "-" * 80)
        logger.info("【结果解读指南】")
        logger.info("1. 卡方检验 (P=%.2e):", p_chi2)

    if logger is not None:
        if p_chi2 < 0.05:
            winner = "全文 (fws)" if n_fws_newer > n_abs_newer else "摘要 (abs)"
            logger.info("   ✅ 显著。两者的胜负比例不是随机的 (50/50)。[%s] 的实体更新频率显著更高。", winner)
        else:
            logger.info("   ❌ 不显著。两者互有胜负，比例接近五五开。")

        logger.info("2. Z 检验 (P=%.2e):", p_z)
        if p_z < 0.05:
            direction = "全文 (fws) 更 '新'" if mean_d < 0 else "摘要 (abs) 更 '新'"
            logger.info("   ✅ 显著。两个来源提取实体的平均年份差存在本质区别。")
            logger.info("   方向: %s。", direction)
        else:
            logger.info("   ❌ 不显著。平均来看，两者的实体新旧程度没有区别。")

        logger.info("3. F 检验 (P=%.2e):", p_f)
        if p_f < 0.05:
            stable = "摘要 (abs)" if var_fws > var_abs else "全文 (fws)"
            logger.info("   ✅ 显著。两者的数据波动性不同。")
            logger.info("   [%s] 的方差更小，说明它提取到的实体年份分布更加【稳定/一致】。", stable)
        else:
            logger.info("   ❌ 不显著。两者的方差相似，稳定性一致。")
