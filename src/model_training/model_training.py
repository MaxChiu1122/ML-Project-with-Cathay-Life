def shap_interaction_selection(model, X, y=None, top_percent=0.2, top_interactions=10):
    """
    使用 SHAP 計算特徵重要性和交互作用項，並篩選出重要特徵和交互作用特徵

    參數：
    - model: 訓練好的機器學習模型 (支持 predict 方法)
    - X: 特徵資料 (DataFrame)
    - y: 目標變數 (Series)，可選
    - top_percent: 保留的特徵重要性百分比 (0~1)
    - top_interactions: 保留的交互作用特徵數量

    Return：
    - top_features: 篩選後的重要特徵名稱 (List)
    - top_interaction_pairs: 篩選後的交互作用特徵對 (List of tuples)
    - interaction_df: 所有交互作用特徵值 (DataFrame)
    """

    # === 計算 SHAP 值 ===
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # 計算特徵重要性 (取絕對值平均)
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    feature_importance = pd.Series(shap_importance, index=X.columns).sort_values(ascending=False)

    # === 篩選重要特徵 ===
    threshold = feature_importance.quantile(1 - top_percent)
    top_features = feature_importance[feature_importance > threshold].index.tolist()

    print(f"\n SHAP 特徵重要性 (Top {int(top_percent * 100)}%):")
    print(feature_importance.head(10))

    # === 計算 SHAP 交互作用 ===
    print("\n 計算交互作用效果：")
    interaction_values = shap.TreeExplainer(model).shap_interaction_values(X)

    # 生成交互作用特徵的 DataFrame
    interaction_df = pd.DataFrame(np.abs(interaction_values).mean(axis=0), columns=X.columns, index=X.columns)

    # 篩選交互作用強的特徵對
    interaction_pairs = interaction_df.unstack().sort_values(ascending=False)
    interaction_pairs = interaction_pairs[interaction_pairs.index.get_level_values(0) != interaction_pairs.index.get_level_values(1)]

    # 取得前 N 個重要交互作用特徵
    top_interaction_pairs = interaction_pairs.head(top_interactions).index.tolist()

    print(f"\n 交互作用項 (Top {top_interactions}):")
    for pair in top_interaction_pairs:
        print(f"{pair[0]} x {pair[1]}: {interaction_df.loc[pair[0], pair[1]]:.4f}")

    return top_features, top_interaction_pairs, interaction_df