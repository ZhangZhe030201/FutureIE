"""Entity newness analysis package."""

from .pipeline import (
    analyze_entity_newness_tfidf_pro_sim,
    final_comprehensive_statistics,
    run_multiple_statistical_tests,
    run_additional_statistical_tests,
)

__all__ = [
    "analyze_entity_newness_tfidf_pro_sim",
    "final_comprehensive_statistics",
    "run_multiple_statistical_tests",
    "run_additional_statistical_tests",
]
