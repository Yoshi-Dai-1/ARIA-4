"""ARIA Taxonomy Mapping 生成エグゼキューター

GitHub Actions から呼ばれ、taxonomy_mapping.parquet を生成する。

環境変数:
  TARGET_YEARS: カンマ区切りの年度リスト（省略時は全年度）
    例: "2025,2026"
"""

import os
import sys

# PYTHONPATH を通すための相対パス処理
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _base_dir not in sys.path:
    sys.path.insert(0, _base_dir)

from engines.taxonomy_engine import TaxonomyMappingEngine  # noqa: E402


def main():
    print("=== ARIA Taxonomy Mapping Generator ===")
    try:
        # 環境変数から対象年度を取得（GitHub Actions の input 経由）
        target_years_str = os.environ.get("TARGET_YEARS", "").strip()
        if target_years_str:
            target_years = [y.strip() for y in target_years_str.split(",")]
            print(f"  指定年度: {target_years}")
        else:
            target_years = None  # 全年度
            print("  対象: 全年度")

        engine = TaxonomyMappingEngine(target_years=target_years)
        df = engine.generate_mapping_dataframe()
        engine.upsert_to_parquet(df)
        print("=== FINISHED ===")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
