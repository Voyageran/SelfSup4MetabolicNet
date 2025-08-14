import pandas as pd
import numpy as np

def main():
    # ========== 路径配置 ==========
    input_path = "G:/project1_obesity_pathway/common_codes/SUV/SUV1185/combined_results/SULmax_1185sub_cleaned_labeled.csv"
    output_path = "G:/project1_obesity_pathway/project1_2EJMMN_obesityPathway/codes/UnsupervisedClassification_subgroup/diagnostic_group12_benchmark_py.csv"

    # ========== 读取数据 ==========
    df = pd.read_csv(input_path)
    assert all(col in df.columns for col in ['Patient', 'BMI', 'age', 'FBG']), "缺少必要列"

    # ========== 三等分 BMI ==========
    df["BMI_bin"] = pd.qcut(df["BMI"], q=3, labels=False)

    # ========== 初始化 ==========
    df["age_bin"] = -1
    df["FBG_bin"] = -1

    # ========== 分层构造标签 ==========
    for bmi_bin in range(3):
        subset = df[df["BMI_bin"] == bmi_bin]
        age_median = subset["age"].median()
        fbg_median = subset["FBG"].median()

        df.loc[subset.index, "age_bin"] = (subset["age"] > age_median).astype(int)
        df.loc[subset.index, "FBG_bin"] = (subset["FBG"] > fbg_median).astype(int)

    # ========== 构造最终12类标签 ==========
    df["y_cls"] = df["BMI_bin"] * 4 + df["age_bin"] * 2 + df["FBG_bin"]

    # ========== 保存结果 ==========
    df_out = df[["Patient", "y_cls"]]
    df_out.to_csv(output_path, index=False)
    print(f"✅ 生成标签文件，保存至：{output_path}")
    print(df_out["y_cls"].value_counts().sort_index())

if __name__=="__main__":
    main()