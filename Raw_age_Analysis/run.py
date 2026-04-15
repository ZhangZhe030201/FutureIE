from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from entity_novelty_analysis.logging_utils import setup_file_logger
from entity_novelty_analysis.pipeline import (
    analyze_entity_newness_tfidf_pro_sim,
    final_comprehensive_statistics,
    run_additional_statistical_tests,
    run_multiple_statistical_tests,
)

# =========================
# Default file paths
# 为了保持与原始脚本一致，这里保留了原脚本中的默认路径与执行顺序。
# 如需运行，请按你的本地环境修改这些路径。
# =========================
ANALYSIS_INPUT_DATA_FILE = r"D:\future_work_sentence\entity_novelty_analysis\data\FutureIE.csv"
MAPPING_FILE = r"D:\future_work_sentence\entity_novelty_analysis\data\ents_year_mapping.csv"
ANALYSIS_OUTPUT_FILE = r"D:\future_work_sentence\entity_novelty_analysis\output\diff.csv"


STATS_INPUT_FILE = ANALYSIS_OUTPUT_FILE

def main() -> None:
    logger, log_path = setup_file_logger(log_dir=PROJECT_ROOT / "logs")
    logger.info("脚本开始运行。")
    logger.info("日志文件: %s", log_path)

    analyze_entity_newness_tfidf_pro_sim(
        data_csv=ANALYSIS_INPUT_DATA_FILE,
        mapping_csv=MAPPING_FILE,
        output_csv=ANALYSIS_OUTPUT_FILE,
        logger=logger,
    )

    final_comprehensive_statistics(
        input_csv=STATS_INPUT_FILE,
        logger=logger,
    )

    run_multiple_statistical_tests(
        input_csv=STATS_INPUT_FILE,
        logger=logger,
    )

    run_additional_statistical_tests(
        input_csv=STATS_INPUT_FILE,
        logger=logger,
    )

    logger.info("脚本运行完成。")


if __name__ == "__main__":
    main()
