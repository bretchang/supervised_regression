import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.datasets import load_diabetes, fetch_california_housing, load_linnerud, fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 設置頁面配置
st.set_page_config(
    page_title="監督式學習-回歸互動教學平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義CSS樣式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .small-text {
        font-size: 0.7rem !important;
        line-height: 1.2 !important;
    }
</style>
""", unsafe_allow_html=True)

# 側邊欄導航
st.sidebar.title("🎯 課程導航")
page = st.sidebar.radio(
    "選擇學習模塊：",
    [
        "🏠 監督式學習概述",
        "📊 數據集探索", 
        "📈 線性回歸",
        "🌀 多項式回歸",
        "⚖️ 正則化回歸",
        "🌳 決策樹回歸",
        "🌲 隨機森林回歸", 
        "🚀 梯度提升回歸",
        "🎯 支持向量回歸",
        "📏 評價指標詳解",
        "🔄 交叉驗證與穩定性",
        "🏆 模型綜合比較"
    ]
)

# 數據集選擇放在側邊欄
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 數據集選擇")
dataset_choice = st.sidebar.selectbox("選擇數據集：", [
    "生理測量", "能源效率", "糖尿病進展", "紅酒品質", "鮑魚年齡", "加州房價"
])

# 數據集簡介
dataset_info = {
    "生理測量": "🏃‍♂️ 適合線性回歸基礎學習 (1KB)",
    "能源效率": "🏢 建築能耗預測，適合特徵工程 (8KB)",
    "糖尿病進展": "🩺 經典回歸問題，適合各種算法 (11KB)", 
    "紅酒品質": "🍷 適合樹模型和隨機森林 (84KB)",
    "鮑魚年齡": "🐚 適合複雜特徵關係學習 (196KB)",
    "加州房價": "🏠 大數據集，適合模型性能比較 (1.3MB)"
}

st.sidebar.markdown("### 📝 數據集特點")
for dataset, description in dataset_info.items():
    if dataset == dataset_choice:
        st.sidebar.markdown(f'<div class="small-text">✅ <strong>{dataset}</strong>: {description}</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f'<div class="small-text"><strong>{dataset}</strong>: {description}</div>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 作者信息")
st.sidebar.info("**This tutorial was made by CCChang18** 🚀")

# 數據載入函數
@st.cache_data
def load_datasets():
    datasets = {}
    
    # 糖尿病數據集 (11KB)
    diabetes = load_diabetes()
    diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    diabetes_df['target'] = diabetes.target.astype(float)
    datasets['糖尿病進展'] = diabetes_df
    
    # 生理測量數據集 (1KB)
    linnerud = load_linnerud()
    linnerud_df = pd.DataFrame(linnerud.data, columns=linnerud.feature_names)
    # 使用第一個目標變量
    linnerud_df['target'] = linnerud.target[:, 0].astype(float)  # Weight
    datasets['生理測量'] = linnerud_df
    
    # 鮑魚年齡數據集 (196KB)
    try:
        abalone = fetch_openml(data_id=183, parser='auto')
        abalone_df = pd.DataFrame(abalone.data, columns=abalone.feature_names)
        abalone_df['target'] = pd.to_numeric(abalone.target, errors='coerce')
        # 處理分類特徵 Sex
        if 'Sex' in abalone_df.columns:
            abalone_df['Sex'] = pd.get_dummies(abalone_df['Sex'], drop_first=True).iloc[:, 0]
        # 移除缺失值
        abalone_df = abalone_df.dropna()
        datasets['鮑魚年齡'] = abalone_df
    except:
        # 如果加載失敗，創建假數據
        np.random.seed(42)
        abalone_df = pd.DataFrame({
            'Length': np.random.uniform(0.1, 0.8, 1000),
            'Diameter': np.random.uniform(0.1, 0.6, 1000),
            'Height': np.random.uniform(0.1, 0.3, 1000),
            'Whole_weight': np.random.uniform(0.1, 2.5, 1000),
            'target': np.random.randint(1, 30, 1000).astype(float)
        })
        datasets['鮑魚年齡'] = abalone_df
    
    # 能源效率數據集 (替換汽車油耗) (8KB)
    try:
        # 使用能源效率數據集 (Energy Efficiency Dataset)
        energy = fetch_openml(data_id=242, parser='auto')
        energy_df = pd.DataFrame(energy.data, columns=energy.feature_names)
        energy_df['target'] = pd.to_numeric(energy.target, errors='coerce')
        # 處理缺失值和非數值特徵
        energy_df = energy_df.dropna()
        # 確保所有特徵都是數值型
        for col in energy_df.columns:
            if col != 'target':
                energy_df[col] = pd.to_numeric(energy_df[col], errors='coerce')
        energy_df = energy_df.dropna()
        datasets['能源效率'] = energy_df
    except:
        # 如果加載失敗，創建高質量的假數據
        np.random.seed(42)
        n_samples = 768
        energy_df = pd.DataFrame({
            'relative_compactness': np.random.uniform(0.6, 1.0, n_samples),
            'surface_area': np.random.uniform(514, 808, n_samples),
            'wall_area': np.random.uniform(245, 416, n_samples),
            'roof_area': np.random.uniform(110, 220, n_samples),
            'overall_height': np.random.uniform(3.5, 7.0, n_samples),
            'orientation': np.random.choice([2, 3, 4, 5], n_samples),
            'glazing_area': np.random.uniform(0, 0.4, n_samples),
            'glazing_area_distribution': np.random.choice([0, 1, 2, 3, 4, 5], n_samples),
        })
        # 目標變量：建築物冷卻負荷 (合理的物理關係)
        energy_df['target'] = (
            energy_df['relative_compactness'] * 15 +
            energy_df['surface_area'] * 0.02 +
            energy_df['wall_area'] * 0.05 +
            energy_df['roof_area'] * 0.1 +
            energy_df['overall_height'] * 3 +
            energy_df['glazing_area'] * 25 +
            np.random.normal(0, 2, n_samples)
        )
        datasets['能源效率'] = energy_df
    
    # 紅酒品質數據集 (84KB)
    try:
        wine_quality = fetch_openml(name='wine-quality-red', version=1, parser='auto')
        wine_df = pd.DataFrame(wine_quality.data, columns=wine_quality.feature_names)
        wine_df['target'] = pd.to_numeric(wine_quality.target, errors='coerce')
        # 移除缺失值
        wine_df = wine_df.dropna()
        datasets['紅酒品質'] = wine_df
    except:
        # 如果加載失敗，創建假數據
        np.random.seed(42)
        wine_df = pd.DataFrame({
            'fixed_acidity': np.random.uniform(4, 16, 1000),
            'volatile_acidity': np.random.uniform(0.1, 1.6, 1000),
            'citric_acid': np.random.uniform(0, 1, 1000),
            'residual_sugar': np.random.uniform(0.9, 16, 1000),
            'chlorides': np.random.uniform(0.01, 0.6, 1000),
            'free_sulfur_dioxide': np.random.uniform(1, 72, 1000),
            'total_sulfur_dioxide': np.random.uniform(6, 290, 1000),
            'density': np.random.uniform(0.99, 1.01, 1000),
            'pH': np.random.uniform(2.7, 4.0, 1000),
            'sulphates': np.random.uniform(0.3, 2.0, 1000),
            'alcohol': np.random.uniform(8.4, 15, 1000),
            'target': np.random.randint(3, 9, 1000).astype(float)
        })
        datasets['紅酒品質'] = wine_df
    
    # 加州房價數據集 (1.3MB)
    california = fetch_california_housing()
    california_df = pd.DataFrame(california.data, columns=california.feature_names)
    california_df['target'] = california.target.astype(float)
    datasets['加州房價'] = california_df
    
    return datasets

all_datasets = load_datasets()

# 通用數據獲取函數
def get_current_data():
    try:
        current_dataset = all_datasets[dataset_choice]
        if current_dataset is None or len(current_dataset) == 0:
            st.error(f"❌ 數據集 '{dataset_choice}' 為空或無法加載")
            return pd.DataFrame(), pd.Series(dtype=float)
        
        X = current_dataset.drop('target', axis=1)
        y = current_dataset['target']
        
        # 確保數據類型正確
        X = X.select_dtypes(include=[np.number])  # 只選擇數值型特徵
        
        if len(X.columns) == 0:
            st.error(f"❌ 數據集 '{dataset_choice}' 沒有數值型特徵")
            return pd.DataFrame(), pd.Series(dtype=float)
            
        y = pd.to_numeric(y, errors='coerce')  # 確保目標變量是數值型
        
        # 移除任何缺失值
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        if len(X) == 0:
            st.error(f"❌ 數據集 '{dataset_choice}' 在移除缺失值後為空")
            return pd.DataFrame(), pd.Series(dtype=float)
        
        return X, y
        
    except Exception as e:
        st.error(f"❌ 加載數據集 '{dataset_choice}' 時發生錯誤: {str(e)}")
        return pd.DataFrame(), pd.Series(dtype=float)

# 頁面內容
if page == "🏠 監督式學習概述":
    st.markdown('<h1 class="main-header">監督式學習(Supervised Learning)-回歸 互動教學平台</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🏅 什麼是監督式學習？")
    
    st.markdown("""
    **監督式學習**是機器學習的一個重要分支，其特點是：
    
    1. **有標籤的訓練數據**：每個樣本都有對應的正確答案
    2. **學習映射關係**：從輸入特徵到輸出標籤的函數關係
    3. **預測新數據**：用學到的模型對未見過的數據進行預測
    """)
    
    st.markdown("## 🎯 什麼是回歸？")
    
    st.markdown("""
    回歸(Regression)是監督式學習的重要分支，目標是預測連續的數值輸出：
    
    1. **連續輸出**：預測結果是實數值，可以是任意精度的數字
    2. **函數映射**：學習從輸入特徵到連續輸出的數學函數關係
    3. **趨勢預測**：捕捉數據中的趨勢和模式來進行數值預測
    """)
    
    st.markdown("### 🔍 回歸 vs 分類")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **回歸(Regression)**：預測**連續數值**
        - 房價預測
        - 溫度預測  
        - 股價預測
        - 銷售額預測
        """)
    
    with col2:
        st.markdown("""
        **分類(Classification)**：預測**離散類別**
        - 垃圾郵件偵測
        - 疾病診斷
        - 圖像識別
        - 情感分析
        """)
    
    st.markdown("### 📈 回歸示例：線性關係")
    
    # 創建簡單的示意圖
    fig = go.Figure()
    x = np.linspace(0, 10, 50)
    y = 2 * x + 1 + np.random.normal(0, 1, 50)
    
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers',
        name='訓練數據', marker=dict(color='blue', size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=x, y=2*x+1, mode='lines',
        name='回歸線', line=dict(color='red', width=3)
    ))
    
    fig.update_layout(
        title="回歸示例：尋找輸入與輸出間的關係",
        xaxis_title="輸入特徵 (X)",
        yaxis_title="輸出標籤 (Y)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("## 📚 本課程學習內容")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔧 核心算法
        - 線性回歸
        - 多項式回歸  
        - 正則化回歸
        - 決策樹回歸
        - 隨機森林回歸
        - 支持向量回歸
        """)
    
    with col2:
        st.markdown("""
        ### 📏 評價指標
        - R² (決定係數)
        - RMSE (均方根誤差)
        - MAE (平均絕對誤差)
        - MSE (均方誤差)
        """)
    
    with col3:
        st.markdown("""
        ### 🔄 驗證方法
        - 交叉驗證
        - K-fold驗證
        - 模型穩定性分析
        - 過擬合檢測
        """)
    
    st.markdown("## 🎯 學習目標")
    st.info("""
    通過本課程，您將能夠：
    1. 理解不同回歸算法的原理和適用場景
    2. 掌握模型參數調優的方法
    3. 學會選擇合適的評價指標
    4. 進行模型比較和選擇
    5. 識別和處理過擬合問題
    """)

elif page == "📊 數據集探索":
    st.markdown('<h1 class="main-header">📊 數據集探索</h1>', unsafe_allow_html=True)
    
    st.info("💡 您可以在左側選擇不同的數據集來探索其特性")
    
    # 獲取當前選擇的數據集
    current_dataset = all_datasets[dataset_choice]
    
    # 數據集信息映射
    dataset_descriptions = {
        "糖尿病進展": {
            "title": "🏥 糖尿病進展預測數據集",
            "target_desc": "一年後的糖尿病進展程度",
            "source": "scikit-learn內建數據集",
            "features": {
                "age": "年齡", "sex": "性別", "bmi": "身體質量指數", 
                "bp": "平均血壓", "s1-s6": "6項血清測量值"
            },
            "color": "lightblue"
        },
        "生理測量": {
            "title": "🏃‍♂️ 生理測量數據集", 
            "target_desc": "體重 (Weight)",
            "source": "scikit-learn內建數據集",
            "features": {
                "Chins": "引體向上次數", "Situps": "仰臥起坐次數", "Jumps": "跳躍次數"
            },
            "color": "lightcoral"
        },
        "鮑魚年齡": {
            "title": "🐚 鮑魚年齡預測數據集",
            "target_desc": "鮑魚年齡 (年)",
            "source": "UCI機器學習庫",
            "features": {
                "Length": "長度", "Diameter": "直徑", "Height": "高度",
                "Whole_weight": "整體重量", "Sex": "性別"
            },
            "color": "lightseagreen"
        },
        "能源效率": {
            "title": "🏢 建築能源效率預測數據集",
            "target_desc": "建築物冷卻負荷 (kWh/m²)",
            "source": "UCI機器學習庫",
            "features": {
                "relative_compactness": "相對緊密度", "surface_area": "表面積", 
                "wall_area": "牆體面積", "roof_area": "屋頂面積",
                "overall_height": "總高度", "orientation": "方向",
                "glazing_area": "玻璃面積", "glazing_area_distribution": "玻璃分布"
            },
            "color": "lightgoldenrodyellow"
        },
        "紅酒品質": {
            "title": "🍷 紅酒品質預測數據集",
            "target_desc": "紅酒品質評分 (0-10)",
            "source": "UCI機器學習庫",
            "features": {
                "fixed_acidity": "固定酸度", "volatile_acidity": "揮發性酸度", 
                "citric_acid": "檸檬酸", "residual_sugar": "殘糖",
                "alcohol": "酒精度", "pH": "酸鹼值"
            },
            "color": "lightpink"
        },
        "加州房價": {
            "title": "🏠 加州房價預測數據集",
            "target_desc": "房屋中位數價格 (單位: 10萬美元)",
            "source": "1990年加州人口普查",
            "features": {
                "MedInc": "家庭收入中位數", "HouseAge": "房屋年齡中位數",
                "AveRooms": "平均房間數", "AveBedrms": "平均臥室數",
                "Population": "區域人口", "AveOccup": "平均入住率",
                "Latitude": "緯度", "Longitude": "經度"
            },
            "color": "lightgreen"
        }
    }
    
    desc = dataset_descriptions[dataset_choice]
    st.markdown(f"## {desc['title']}")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📋 數據集資訊")
        st.info(f"""
        - **樣本數量**: {len(current_dataset)} 個樣本
        - **特徵數量**: {len(current_dataset.columns)-1} 個特徵
        - **目標變數**: {desc['target_desc']}
        - **數據來源**: {desc['source']}
        """)
    
    with col2:
        st.markdown("### 🔬 特徵說明")
        for feature, description in desc['features'].items():
            st.markdown(f"- **{feature}**: {description}")
    
    # 目標變數分布單獨拉出來
    st.markdown("### 📈 目標變數分布")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=current_dataset['target'],
        nbinsx=30,
        name='目標變數',
        marker_color=desc['color']
    ))
    fig.update_layout(
        title="目標變數分布",
        xaxis_title=desc['target_desc'],
        yaxis_title="頻率",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 特徵相關性熱力圖
    st.markdown("### 🔥 特徵相關性分析")
    corr_matrix = current_dataset.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',
        zmid=0
    ))
    fig.update_layout(
        title="特徵間相關性熱力圖",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 特殊可視化 - 僅針對加州房價
    if dataset_choice == "加州房價":
        st.markdown("### 🗺️ 地理位置與房價關係")
        fig = px.scatter(
            current_dataset, 
            x='Longitude', 
            y='Latitude',
            color='target',
            color_continuous_scale='Viridis',
            title="加州各地區房價分布",
            labels={'target': '房價 (10萬美元)'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

elif page == "📈 線性回歸":
    st.markdown('<h1 class="main-header">📈 線性回歸 (Linear Regression)</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🧮 數學原理")
    
    st.markdown("### 📐 線性回歸公式")
    st.latex(r'''
    \hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n
    ''')
    
    st.markdown("其中：")
    st.latex(r'''
    \begin{align}
    \hat{y} &= \text{預測值} \\
    \beta_0 &= \text{截距項 (bias)} \\
    \beta_i &= \text{第i個特徵的係數} \\
    x_i &= \text{第i個特徵值}
    \end{align}
    ''')
    
    st.markdown("### 🎯 目標函數 (最小平方法)")
    st.latex(r'''
    \min_{\beta} \sum_{i=1}^m (y_i - \hat{y}_i)^2 = \min_{\beta} \sum_{i=1}^m (y_i - \beta_0 - \sum_{j=1}^n \beta_j x_{ij})^2
    ''')
    
    st.markdown("### 📊 解析解 (Normal Equation)")
    st.latex(r'''
    \beta = (X^T X)^{-1} X^T y
    ''')
    
    # 優缺點並排顯示
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("### ✅ 優點")
        st.markdown("""
        - 📍 **簡單易懂**：數學原理清晰
        - ⚡ **計算速度快**：有解析解
        - 🔍 **可解釋性強**：係數意義明確
        - 🛡️ **不易過擬合**：模型簡單
        - 📊 **基礎算法**：其他算法的基礎
        """)
    
    with col2:
        st.error("### ❌ 缺點")
        st.markdown("""
        - 📏 **假設線性關係**：無法處理非線性
        - 🎯 **對離群值敏感**：影響整體效果
        - 🔗 **多重共線性問題**：特徵相關性高
        - 🔢 **特徵限制**：需要數值化處理
        - 📉 **表達能力有限**：複雜關係難以建模
        """)
    
    st.markdown("## 🎛️ 互動式實驗")
    
    X, y = get_current_data()
    
    # 特徵選擇
    selected_features = st.multiselect(
        "選擇用於建模的特徵：",
        X.columns.tolist(),
        default=X.columns.tolist()[:3]
    )
    
    if len(selected_features) > 0 and len(X) > 0:
        X_selected = X[selected_features]
        
        # 數據分割
        test_size = st.slider("測試集比例：", 0.1, 0.5, 0.2, 0.05)
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=test_size, random_state=42
        )
        
        # 特徵標準化選項
        normalize = st.checkbox("是否進行特徵標準化？", value=True)
        
        if normalize:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
        
        # 訓練模型
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # 預測
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # 評估指標
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # 顯示結果
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Train-Data R²", f"{train_r2:.4f}")
        with col2:
            st.metric("Test-Data R²", f"{test_r2:.4f}")
        with col3:
            st.metric("Train-Data RMSE", f"{train_rmse:.2f}")
        with col4:
            st.metric("Test-Data RMSE", f"{test_rmse:.2f}")
        
        # 預測 vs 真實值圖 (改進標籤)
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_train, y=y_train_pred,
                mode='markers', name='Train-Data',
                marker=dict(color='blue', size=6, opacity=0.6)
            ))
            fig.add_trace(go.Scatter(
                x=[y_train.min(), y_train.max()],
                y=[y_train.min(), y_train.max()],
                mode='lines', name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                title=f"Train-Data: Predict vs Actual (R² = {train_r2:.4f})",
                xaxis_title="Actual",
                yaxis_title="Predict"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test, y=y_test_pred,
                mode='markers', name='Test-Data',
                marker=dict(color='green', size=6, opacity=0.6)
            ))
            fig.add_trace(go.Scatter(
                x=[y_test.min(), y_test.max()],
                y=[y_test.min(), y_test.max()],
                mode='lines', name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                title=f"Test-Data: Predict vs Actual (R² = {test_r2:.4f})",
                xaxis_title="Actual",
                yaxis_title="Predict"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 新增時間序列風格的預測圖
        st.markdown("### 📈 預測結果時間序列視圖")
        
        # 合併數據以便繪圖
        train_size = len(y_train)
        total_size = len(y_train) + len(y_test)
        
        # 創建索引
        train_indices = np.arange(train_size)
        test_indices = np.arange(train_size, total_size)
        
        fig = go.Figure()
        
        # 真實值 (藍色)
        fig.add_trace(go.Scatter(
            x=train_indices, y=y_train,
            mode='markers', name='Actual (Train)',
            marker=dict(color='blue', size=4)
        ))
        fig.add_trace(go.Scatter(
            x=test_indices, y=y_test,
            mode='markers', name='Actual (Test)',
            marker=dict(color='blue', size=4)
        ))
        
        # 訓練預測 (綠色)
        fig.add_trace(go.Scatter(
            x=train_indices, y=y_train_pred,
            mode='markers', name='Predict (Train)',
            marker=dict(color='green', size=4)
        ))
        
        # 測試預測 (紅色)
        fig.add_trace(go.Scatter(
            x=test_indices, y=y_test_pred,
            mode='markers', name='Predict (Test)',
            marker=dict(color='red', size=4)
        ))
        
        # 添加分割線
        fig.add_vline(
            x=train_size-0.5, 
            line_dash="dash", 
            line_color="black",
            annotation_text="Train/Test Split"
        )
        
        fig.update_layout(
            title="Model Predictions vs Actual Values",
            xaxis_title="Index",
            yaxis_title="Value",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 係數分析
        st.markdown("### 📊 模型係數分析")
        coef_df = pd.DataFrame({
            '特徵': selected_features,
            '係數': model.coef_,
            '絕對值': np.abs(model.coef_)
        }).sort_values('絕對值', ascending=False)
        
        fig = go.Figure(data=[
            go.Bar(x=coef_df['特徵'], y=coef_df['係數'], 
                   marker_color=['red' if x < 0 else 'blue' for x in coef_df['係數']])
        ])
        fig.update_layout(
            title="特徵係數大小 (正負表示影響方向)",
            xaxis_title="特徵",
            yaxis_title="係數值"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"**截距項 (β₀)**: {model.intercept_:.4f}")

# 繼續其他頁面...
elif page == "🌀 多項式回歸":
    st.markdown('<h1 class="main-header">🌀 多項式回歸 (Polynomial Regression)</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🧮 數學原理")
    
    st.markdown("### 📐 多項式回歸公式")
    st.latex(r'''
    \hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + ... + \beta_d x^d
    ''')
    
    st.markdown("對於多特徵的情況，包含交互項：")
    st.latex(r'''
    \hat{y} = \beta_0 + \sum_{i=1}^n \beta_i x_i + \sum_{i=1}^n \beta_{ii} x_i^2 + \sum_{i<j} \beta_{ij} x_i x_j + ...
    ''')
    
    st.markdown("### 🔄 特徵變換")
    st.markdown("多項式回歸本質上是線性回歸的擴展，通過特徵工程創造新特徵：")
    st.latex(r'''
    \begin{align}
    \phi_1(x) &= x \\
    \phi_2(x) &= x^2 \\
    \phi_3(x) &= x^3 \\
    &\vdots \\
    \phi_d(x) &= x^d
    \end{align}
    ''')
    
    # 優缺點並排顯示
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("### ✅ 優點")
        st.markdown("""
        - 🔄 **捕捉非線性關係**：處理複雜模式
        - 📊 **仍是線性模型**：易於優化求解
        - 🔧 **易於實現**：基於線性回歸
        - ⚙️ **可控複雜度**：調整多項式階數
        - 🎯 **靈活性強**：適應各種曲線
        """)
    
    with col2:
        st.error("### ❌ 缺點")
        st.markdown("""
        - ⚠️ **容易過擬合**：高階項風險大
        - 💥 **特徵爆炸**：組合特徵數量急增
        - 🎯 **離群值敏感**：影響更加嚴重
        - 📉 **數值不穩定**：高次項計算問題
        - 🔗 **特徵相關性**：多重共線性加劇
        """)
    
    st.markdown("## 🎛️ 互動式實驗")
    
    X, y = get_current_data()
    
    # 參數設置
    col1, col2 = st.columns(2)
    
    with col1:
        degree = st.slider("多項式階數 (degree)：", 1, 5, 2)
        include_bias = st.checkbox("包含偏置項", value=True)
    
    with col2:
        interaction_only = st.checkbox("僅包含交互項 (不包含高次項)", value=False)
        test_size = st.slider("測試集比例：", 0.1, 0.5, 0.2, 0.05)
    
    # 特徵選擇（限制數量避免特徵爆炸）
    max_features = 3 if degree > 2 else 5
    selected_features = st.multiselect(
        f"選擇特徵 (建議最多{max_features}個，避免特徵爆炸)：",
        X.columns.tolist(),
        default=X.columns.tolist()[:min(max_features, len(X.columns))]
    )
    
    if len(selected_features) > 0:
        X_selected = X[selected_features]
        
        # 數據分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=test_size, random_state=42
        )
        
        # 特徵標準化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 多項式特徵變換
        poly = PolynomialFeatures(
            degree=degree, 
            include_bias=include_bias,
            interaction_only=interaction_only
        )
        X_train_poly = poly.fit_transform(X_train_scaled)
        X_test_poly = poly.transform(X_test_scaled)
        
        # 顯示特徵變換後的信息
        st.info(f"""
        **特徵變換結果：**
        - 原始特徵數：{X_train_scaled.shape[1]}
        - 變換後特徵數：{X_train_poly.shape[1]}
        - 特徵增加了 {X_train_poly.shape[1] - X_train_scaled.shape[1]} 個
        """)
        
        # 警告特徵過多
        if X_train_poly.shape[1] > 50:
            st.warning("⚠️ 特徵數量過多，可能導致過擬合和計算緩慢！")
        
        # 訓練模型
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        
        # 預測
        y_train_pred = model.predict(X_train_poly)
        y_test_pred = model.predict(X_test_poly)
        
        # 評估指標
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # 顯示結果
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Train-Data R²", f"{train_r2:.4f}")
        with col2:
            st.metric("Test-Data R²", f"{test_r2:.4f}")
        with col3:
            st.metric("Train-Data RMSE", f"{train_rmse:.2f}")
        with col4:
            st.metric("Test-Data RMSE", f"{test_rmse:.2f}")
        
        # 過擬合檢測
        overfitting_gap = train_r2 - test_r2
        if overfitting_gap > 0.1:
            st.warning(f"⚠️ 可能存在過擬合！訓練集與測試集R²差距：{overfitting_gap:.4f}")
        elif overfitting_gap < -0.05:
            st.info("ℹ️ 模型可能欠擬合，可以考慮增加複雜度")
        else:
            st.success("✅ 模型拟合良好！")
        
        # 預測 vs 真實值圖
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_train, y=y_train_pred,
                mode='markers', name='Train-Data',
                marker=dict(color='blue', size=6, opacity=0.6)
            ))
            fig.add_trace(go.Scatter(
                x=[y_train.min(), y_train.max()],
                y=[y_train.min(), y_train.max()],
                mode='lines', name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                title=f"Train-Data: Predict vs Actual (R² = {train_r2:.4f})",
                xaxis_title="Actual",
                yaxis_title="Predict"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test, y=y_test_pred,
                mode='markers', name='Test-Data',
                marker=dict(color='green', size=6, opacity=0.6)
            ))
            fig.add_trace(go.Scatter(
                x=[y_test.min(), y_test.max()],
                y=[y_test.min(), y_test.max()],
                mode='lines', name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                title=f"Test-Data: Predict vs Actual (R² = {test_r2:.4f})",
                xaxis_title="Actual",
                yaxis_title="Predict"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 新增時間序列風格的預測圖
        st.markdown("### 📈 預測結果時間序列視圖")
        
        # 合併數據以便繪圖
        train_size = len(y_train)
        total_size = len(y_train) + len(y_test)
        
        # 創建索引
        train_indices = np.arange(train_size)
        test_indices = np.arange(train_size, total_size)
        
        fig = go.Figure()
        
        # 真實值 (藍色)
        fig.add_trace(go.Scatter(
            x=train_indices, y=y_train,
            mode='markers', name='Actual (Train)',
            marker=dict(color='blue', size=4)
        ))
        fig.add_trace(go.Scatter(
            x=test_indices, y=y_test,
            mode='markers', name='Actual (Test)',
            marker=dict(color='blue', size=4)
        ))
        
        # 訓練預測 (綠色)
        fig.add_trace(go.Scatter(
            x=train_indices, y=y_train_pred,
            mode='markers', name='Predict (Train)',
            marker=dict(color='green', size=4)
        ))
        
        # 測試預測 (紅色)
        fig.add_trace(go.Scatter(
            x=test_indices, y=y_test_pred,
            mode='markers', name='Predict (Test)',
            marker=dict(color='red', size=4)
        ))
        
        # 添加分割線
        fig.add_vline(
            x=train_size-0.5, 
            line_dash="dash", 
            line_color="black",
            annotation_text="Train/Test Split"
        )
        
        fig.update_layout(
            title="Polynomial Regression: Predictions vs Actual Values",
            xaxis_title="Index",
            yaxis_title="Value",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 特徵重要性（係數絕對值）
        if X_train_poly.shape[1] <= 20:  # 只有特徵不太多時才顯示
            st.markdown("### 📊 多項式特徵係數分析")
            feature_names = poly.get_feature_names_out(selected_features)
            coef_df = pd.DataFrame({
                '特徵': feature_names,
                '係數': model.coef_,
                '絕對值': np.abs(model.coef_)
            }).sort_values('絕對值', ascending=False).head(10)
            
            fig = go.Figure(data=[
                go.Bar(x=coef_df['特徵'], y=coef_df['係數'],
                       marker_color=['red' if x < 0 else 'blue' for x in coef_df['係數']])
            ])
            fig.update_layout(
                title="前10個最重要特徵的係數",
                xaxis_title="特徵",
                yaxis_title="係數值",
                xaxis_tickangle=45
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ 請選擇至少一個特徵進行多項式回歸實驗。")

elif page == "⚖️ 正則化回歸":
    st.markdown('<h1 class="main-header">⚖️ 正則化回歸 (Regularized Regression)</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🧮 數學原理")
    
    st.markdown("### 📐 正則化回歸的基本思想")
    st.markdown("正則化通過在損失函數中添加懲罰項來控制模型複雜度，防止過擬合：")
    
    st.latex(r'''
    \text{損失函數} = \text{原始損失} + \lambda \times \text{正則化項}
    ''')
    
    st.markdown("### 🎯 Ridge回歸 (L2正則化)")
    st.latex(r'''
    \min_{\beta} \sum_{i=1}^m (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^n \beta_j^2
    ''')
    
    st.markdown("### 🎯 Lasso回歸 (L1正則化)")
    st.latex(r'''
    \min_{\beta} \sum_{i=1}^m (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^n |\beta_j|
    ''')
    
    st.markdown("### 🎯 ElasticNet (L1 + L2正則化)")
    st.latex(r'''
    \min_{\beta} \sum_{i=1}^m (y_i - \hat{y}_i)^2 + \lambda_1 \sum_{j=1}^n |\beta_j| + \lambda_2 \sum_{j=1}^n \beta_j^2
    ''')
    
    # 優缺點並排顯示
    st.markdown("### 📊 三種正則化方法比較")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🔵 Ridge回歸")
        st.markdown("**優點：**")
        st.markdown("""
        - 處理多重共線性
        - 係數收縮但不為0
        - 數值穩定性好
        - 適合特徵多的情況
        """)
        st.markdown("**缺點：**")
        st.markdown("""
        - 不進行特徵選擇
        - 所有特徵都保留
        """)
    
    with col2:
        st.markdown("#### 🟡 Lasso回歸")
        st.markdown("**優點：**")
        st.markdown("""
        - 自動特徵選擇
        - 係數可以為0
        - 模型解釋性強
        - 適合稀疏特徵
        """)
        st.markdown("**缺點：**")
        st.markdown("""
        - 相關特徵中隨機選擇
        - 數值不穩定
        """)
    
    with col3:
        st.markdown("#### 🟢 ElasticNet")
        st.markdown("**優點：**")
        st.markdown("""
        - 結合L1和L2優點
        - 處理相關特徵組
        - 平衡特徵選擇和穩定性
        """)
        st.markdown("**缺點：**")
        st.markdown("""
        - 參數調優複雜
        - 計算成本較高
        """)
    
    st.markdown("## 🎛️ 互動式實驗")
    
    X, y = get_current_data()
    
    # 參數設置
    col1, col2 = st.columns(2)
    
    with col1:
        reg_type = st.selectbox("選擇正則化類型：", ["Ridge", "Lasso", "ElasticNet"])
        
        # 使用對數尺度的滑動條，更容易調整小數值
        alpha_log = st.slider(
            "正則化強度 (α) - 對數尺度：", 
            min_value=-4.0,  # 10^-4 = 0.0001
            max_value=2.0,   # 10^1 = 10
            value=-1.0,      # 10^-1 = 0.1 (更合理的默認值)
            step=0.1,
            help="滑動條為對數尺度：-4表示0.0001，0表示1.0，1表示10"
        )
        alpha = 10 ** alpha_log
        
        # 顯示實際的α值
        st.write(f"實際 α 值：{alpha:.4f}")
        
        # 根據α值給出建議
        if alpha < 0.001:
            st.info("💡 極低正則化：適合高維數據或需要保留所有特徵")
        elif alpha < 0.1:
            st.info("💡 低正則化：適合輕微過擬合的情況")
        elif alpha < 1.0:
            st.info("💡 中等正則化：平衡偏差和方差")
        else:
            st.warning("⚠️ 高正則化：可能導致欠擬合，謹慎使用")
    
    with col2:
        test_size = st.slider("測試集比例：", 0.1, 0.5, 0.2, 0.05)
        if reg_type == "ElasticNet":
            l1_ratio = st.slider("L1比例 (l1_ratio)：", 0.0, 1.0, 0.5, 0.1)
    
    # 特徵選擇
    selected_features = st.multiselect(
        "選擇用於建模的特徵：",
        X.columns.tolist(),
        default=X.columns.tolist()
    )
    
    if len(selected_features) > 0:
        X_selected = X[selected_features]
        
        # 數據分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=test_size, random_state=42
        )
        
        # 特徵標準化（正則化回歸必須標準化）
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 選擇模型
        if reg_type == "Ridge":
            model = Ridge(alpha=alpha)
        elif reg_type == "Lasso":
            model = Lasso(alpha=alpha, max_iter=2000)
        else:  # ElasticNet
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000)
        
        # 訓練模型
        model.fit(X_train_scaled, y_train)
        
        # 預測
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # 評估指標
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # 計算特徵數量（非零係數）
        non_zero_features = np.sum(np.abs(model.coef_) > 1e-6)
        
        # 顯示結果
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Train-Data R²", f"{train_r2:.4f}")
        with col2:
            st.metric("Test-Data R²", f"{test_r2:.4f}")
        with col3:
            st.metric("Train-Data RMSE", f"{train_rmse:.2f}")
        with col4:
            st.metric("Test-Data RMSE", f"{test_rmse:.2f}")
        with col5:
            st.metric("特徵數量", f"{non_zero_features}/{len(model.coef_)}")
        
        # 改善的正則化效果分析
        # 首先檢查模型是否有效
        if test_r2 < 0 or train_r2 < 0:
            st.error("❌ 模型效果極差！R²為負值，請調整參數")
            if alpha > 1.0:
                st.warning("💡 建議：正則化強度過高，嘗試降低α值")
            elif non_zero_features == 0:
                st.warning("💡 建議：所有特徵被正則化歸零，嘗試降低α值")
        elif test_r2 < 0.1 and train_r2 < 0.1:
            st.warning("⚠️ 模型整體表現不佳，考慮調整正則化參數或增加特徵")
        else:
            # 檢查過擬合/欠擬合
            regularization_gap = train_r2 - test_r2
            if regularization_gap > 0.15:
                st.warning(f"⚠️ 可能過擬合！建議增加正則化強度 (當前α={alpha})")
                st.markdown(f"R²差距：{regularization_gap:.4f}")
            elif regularization_gap < -0.1:
                st.info(f"ℹ️ 可能欠擬合，建議降低正則化強度 (當前α={alpha})")
                st.markdown(f"R²差距：{regularization_gap:.4f}")
            elif non_zero_features < len(model.coef_) * 0.1:
                st.warning(f"⚠️ 正則化過強！僅保留{non_zero_features}個特徵，考慮降低α值")
            else:
                st.success("✅ 正則化效果良好！")
                st.markdown(f"保留了 {non_zero_features}/{len(model.coef_)} 個特徵")
        
        # 額外的正則化建議
        if reg_type == "Lasso" and non_zero_features == 0:
            st.error("🚨 Lasso將所有特徵歸零！")
            st.markdown("**建議操作：**")
            st.markdown("- 🔽 大幅降低α值（嘗試0.001或更小）")
            st.markdown("- 📊 檢查特徵是否已正確標準化")
            st.markdown("- 🔍 考慮使用Ridge回歸替代")
        elif reg_type == "Lasso" and non_zero_features < 3:
            st.warning(f"⚠️ Lasso僅保留{non_zero_features}個特徵，模型可能過於簡化")
            st.markdown("**建議：** 適當降低α值以保留更多有用特徵")
        
        # 預測 vs 真實值圖
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_train, y=y_train_pred,
                mode='markers', name='Train-Data',
                marker=dict(color='blue', size=6, opacity=0.6)
            ))
            fig.add_trace(go.Scatter(
                x=[y_train.min(), y_train.max()],
                y=[y_train.min(), y_train.max()],
                mode='lines', name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                title=f"Train-Data: Predict vs Actual (R² = {train_r2:.4f})",
                xaxis_title="Actual",
                yaxis_title="Predict"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test, y=y_test_pred,
                mode='markers', name='Test-Data',
                marker=dict(color='green', size=6, opacity=0.6)
            ))
            fig.add_trace(go.Scatter(
                x=[y_test.min(), y_test.max()],
                y=[y_test.min(), y_test.max()],
                mode='lines', name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                title=f"Test-Data: Predict vs Actual (R² = {test_r2:.4f})",
                xaxis_title="Actual",
                yaxis_title="Predict"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 時間序列風格的預測圖
        st.markdown("### 📈 預測結果時間序列視圖")
        
        train_size = len(y_train)
        train_indices = np.arange(train_size)
        test_indices = np.arange(train_size, train_size + len(y_test))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=train_indices, y=y_train,
            mode='markers', name='Actual (Train)',
            marker=dict(color='blue', size=4)
        ))
        fig.add_trace(go.Scatter(
            x=test_indices, y=y_test,
            mode='markers', name='Actual (Test)',
            marker=dict(color='blue', size=4)
        ))
        fig.add_trace(go.Scatter(
            x=train_indices, y=y_train_pred,
            mode='markers', name='Predict (Train)',
            marker=dict(color='green', size=4)
        ))
        fig.add_trace(go.Scatter(
            x=test_indices, y=y_test_pred,
            mode='markers', name='Predict (Test)',
            marker=dict(color='red', size=4)
        ))
        
        fig.add_vline(
            x=train_size-0.5, 
            line_dash="dash", 
            line_color="black",
            annotation_text="Train/Test Split"
        )
        
        fig.update_layout(
            title=f"{reg_type} Regression: Predictions vs Actual Values",
            xaxis_title="Index",
            yaxis_title="Value",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 係數分析
        st.markdown("### 📊 正則化係數分析")
        
        coef_df = pd.DataFrame({
            '特徵': selected_features,
            '係數': model.coef_,
            '絕對值': np.abs(model.coef_)
        }).sort_values('絕對值', ascending=False)
        
        # 統計零係數（特徵選擇效果）
        zero_coefs = np.sum(np.abs(model.coef_) < 1e-6)
        st.info(f"**{reg_type} 正則化效果：** {zero_coefs} 個係數接近0（共{len(model.coef_)}個特徵）")
        
        fig = go.Figure(data=[
            go.Bar(x=coef_df['特徵'], y=coef_df['係數'], 
                   marker_color=['red' if x < 0 else 'blue' for x in coef_df['係數']])
        ])
        fig.update_layout(
            title=f"{reg_type} 回歸係數 (α={alpha})",
            xaxis_title="特徵",
            yaxis_title="係數值",
            xaxis_tickangle=45
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"**截距項 (β₀)**: {model.intercept_:.4f}")

elif page == "🌳 決策樹回歸":
    st.markdown('<h1 class="main-header">🌳 決策樹回歸 (Decision Tree Regression)</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🧮 數學原理")
    
    st.markdown("### 🌳 決策樹的基本概念")
    st.markdown("決策樹通過遞歸地分割特徵空間來建立預測模型：")
    
    st.latex(r'''
    \hat{y} = \sum_{j=1}^{J} c_j \cdot I(x \in R_j)
    ''')
    
    st.markdown("其中：")
    st.markdown("- $R_j$ = 第j個葉節點區域")
    st.markdown("- $c_j$ = 第j個葉節點的預測值（平均值）")
    st.markdown("- $I(x \\in R_j)$ = 指示函數，當x在區域$R_j$時為1，否則為0")
    
    st.markdown("### 🎯 分割標準 (MSE)")
    st.markdown("決策樹使用均方誤差作為分割標準：")
    st.latex(r'''
    MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \bar{y})^2
    ''')
    
    st.markdown("### 📊 分割過程")
    st.markdown("在每個節點，尋找最佳分割點：")
    st.latex(r'''
    \min_{j,s} \left[ \min_{c_1} \sum_{x_i \in R_1(j,s)} (y_i - c_1)^2 + \min_{c_2} \sum_{x_i \in R_2(j,s)} (y_i - c_2)^2 \right]
    ''')
    
    # 優缺點並排顯示
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("### ✅ 優點")
        st.markdown("""
        - 📝 **易於理解和解釋**：樹狀結構直觀
        - 🔧 **不需要特徵標準化**：處理原始數據
        - 🌊 **能處理非線性關係**：分段線性逼近
        - 🎯 **自動進行特徵選擇**：忽略不重要特徵
        - 📊 **能處理數值和類別特徵**：混合數據類型
        - 🛡️ **對離群值不敏感**：基於分割點
        """)
    
    with col2:
        st.error("### ❌ 缺點")
        st.markdown("""
        - ⚠️ **容易過擬合**：深度過大時
        - 🎲 **對訓練數據敏感**：小變化大影響
        - 📈 **偏向多值特徵**：選擇偏差
        - 📏 **難以捕捉線性關係**：階梯式預測
        - 🔄 **預測不穩定**：高方差問題
        - ⚖️ **可能產生偏差**：局部最優分割
        """)
    
    st.markdown("## 🎛️ 互動式實驗")
    
    X, y = get_current_data()
    
    # 參數設置
    col1, col2 = st.columns(2)
    
    with col1:
        max_depth = st.slider("最大深度 (max_depth)：", 1, 20, 5)
        min_samples_split = st.slider("最小分割樣本數 (min_samples_split)：", 2, 50, 10)
    
    with col2:
        min_samples_leaf = st.slider("葉節點最小樣本數 (min_samples_leaf)：", 1, 20, 5)
        test_size = st.slider("測試集比例：", 0.1, 0.5, 0.2, 0.05)
    
    # 特徵選擇
    selected_features = st.multiselect(
        "選擇用於建模的特徵：",
        X.columns.tolist(),
        default=X.columns.tolist()[:5]
    )
    
    if len(selected_features) > 0:
        X_selected = X[selected_features]
        
        # 數據分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=test_size, random_state=42
        )
        
        # 決策樹模型
        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
        # 訓練模型
        model.fit(X_train, y_train)
        
        # 預測
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # 評估指標
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # 顯示結果
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Train-Data R²", f"{train_r2:.4f}")
        with col2:
            st.metric("Test-Data R²", f"{test_r2:.4f}")
        with col3:
            st.metric("Train-Data RMSE", f"{train_rmse:.2f}")
        with col4:
            st.metric("Test-Data RMSE", f"{test_rmse:.2f}")
        
        # 過擬合檢測
        overfitting_gap = train_r2 - test_r2
        if overfitting_gap > 0.15:
            st.warning(f"⚠️ 決策樹過擬合嚴重！R²差距：{overfitting_gap:.4f}")
            st.markdown("**建議：** 減少max_depth或增加min_samples_split")
        elif overfitting_gap < -0.05:
            st.info("ℹ️ 模型可能欠擬合，可以增加模型複雜度")
        else:
            st.success("✅ 模型拟合良好！")
        
        # 預測 vs 真實值圖
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_train, y=y_train_pred,
                mode='markers', name='Train-Data',
                marker=dict(color='blue', size=6, opacity=0.6)
            ))
            fig.add_trace(go.Scatter(
                x=[y_train.min(), y_train.max()],
                y=[y_train.min(), y_train.max()],
                mode='lines', name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                title=f"Train-Data: Predict vs Actual (R² = {train_r2:.4f})",
                xaxis_title="Actual",
                yaxis_title="Predict"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test, y=y_test_pred,
                mode='markers', name='Test-Data',
                marker=dict(color='green', size=6, opacity=0.6)
            ))
            fig.add_trace(go.Scatter(
                x=[y_test.min(), y_test.max()],
                y=[y_test.min(), y_test.max()],
                mode='lines', name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                title=f"Test-Data: Predict vs Actual (R² = {test_r2:.4f})",
                xaxis_title="Actual",
                yaxis_title="Predict"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 時間序列風格的預測圖
        st.markdown("### 📈 預測結果時間序列視圖")
        
        train_size = len(y_train)
        train_indices = np.arange(train_size)
        test_indices = np.arange(train_size, train_size + len(y_test))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=train_indices, y=y_train,
            mode='markers', name='Actual (Train)',
            marker=dict(color='blue', size=4)
        ))
        fig.add_trace(go.Scatter(
            x=test_indices, y=y_test,
            mode='markers', name='Actual (Test)',
            marker=dict(color='blue', size=4)
        ))
        fig.add_trace(go.Scatter(
            x=train_indices, y=y_train_pred,
            mode='markers', name='Predict (Train)',
            marker=dict(color='green', size=4)
        ))
        fig.add_trace(go.Scatter(
            x=test_indices, y=y_test_pred,
            mode='markers', name='Predict (Test)',
            marker=dict(color='red', size=4)
        ))
        
        fig.add_vline(
            x=train_size-0.5, 
            line_dash="dash", 
            line_color="black",
            annotation_text="Train/Test Split"
        )
        
        fig.update_layout(
            title="Decision Tree: Predictions vs Actual Values",
            xaxis_title="Index",
            yaxis_title="Value",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 特徵重要性分析
        st.markdown("### 📊 特徵重要性分析")
        
        feature_importance = pd.DataFrame({
            '特徵': selected_features,
            '重要性': model.feature_importances_
        }).sort_values('重要性', ascending=False)
        
        fig = go.Figure(data=[
            go.Bar(x=feature_importance['特徵'], y=feature_importance['重要性'],
                   marker_color='green')
        ])
        fig.update_layout(
            title="決策樹特徵重要性",
            xaxis_title="特徵",
            yaxis_title="重要性",
            xaxis_tickangle=45
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 樹的深度信息
        actual_depth = model.get_depth()
        n_leaves = model.get_n_leaves()
        st.info(f"**決策樹結構：** 實際深度={actual_depth}, 葉節點數={n_leaves}")

elif page == "🌲 隨機森林回歸":
    st.markdown('<h1 class="main-header">🌲 隨機森林回歸 (Random Forest Regression)</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🧮 數學原理")
    
    st.markdown("### 🌲 隨機森林的基本概念")
    st.markdown("隨機森林是多個決策樹的集成，通過平均預測來減少方差：")
    
    st.latex(r'''
    \hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(x)
    ''')
    
    st.markdown("其中：")
    st.latex(r'''
    \begin{align}
    B &= \text{樹的數量} \\
    T_b(x) &= \text{第b棵樹的預測結果} \\
    \end{align}
    ''')
    
    st.markdown("### 🎲 Bagging + 隨機特徵")
    st.markdown("每棵樹使用Bootstrap抽樣的數據，並在每個分割點隨機選擇特徵子集：")
    st.latex(r'''
    \text{每個分割點使用} \approx \sqrt{p} \text{個隨機特徵 (p為總特徵數)}
    ''')
    
    st.markdown("### 📊 袋外誤差 (OOB Error)")
    st.latex(r'''
    OOB = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i^{OOB})^2
    ''')
    
    # 優缺點並排顯示
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("### ✅ 優點")
        st.markdown("""
        - 🛡️ **減少過擬合**：集成多棵樹
        - 📊 **提供特徵重要性**：量化特徵貢獻
        - 💾 **處理大數據集**：高效算法
        - 🎯 **對離群值穩健**：多數決定
        - ⚡ **可並行訓練**：獨立建樹
        - 🔧 **不需要調參太多**：相對魯棒
        - 📈 **提供OOB誤差估計**：內建驗證
        """)
    
    with col2:
        st.error("### ❌ 缺點")
        st.markdown("""
        - 🔍 **模型不可解釋**：黑盒性質
        - 💿 **記憶體需求大**：存儲多棵樹
        - ⏱️ **預測時間較長**：多個模型預測
        - 📉 **可能在噪聲數據上過擬合**：學習噪聲
        - 📊 **偏向類別多的特徵**：選擇偏差
        """)
    
    st.markdown("## 🎛️ 互動式實驗")
    
    X, y = get_current_data()
    
    # 參數設置
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider("樹的數量 (n_estimators)：", 10, 200, 100, 10)
        max_depth = st.slider("最大深度 (max_depth)：", 3, 20, 10)
    
    with col2:
        min_samples_split = st.slider("最小分割樣本數：", 2, 20, 5)
        test_size = st.slider("測試集比例：", 0.1, 0.5, 0.2, 0.05)
    
    # 特徵選擇
    selected_features = st.multiselect(
        "選擇用於建模的特徵：",
        X.columns.tolist(),
        default=X.columns.tolist()
    )
    
    if len(selected_features) > 0:
        X_selected = X[selected_features]
        
        # 數據分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=test_size, random_state=42
        )
        
        # 隨機森林模型
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1  # 使用所有CPU核心
        )
        
        # 訓練模型
        model.fit(X_train, y_train)
        
        # 預測
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # 評估指標
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # OOB分數
        oob_score = model.oob_score_ if hasattr(model, 'oob_score_') else None
        
        # 顯示結果
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Train-Data R²", f"{train_r2:.4f}")
        with col2:
            st.metric("Test-Data R²", f"{test_r2:.4f}")
        with col3:
            st.metric("Train-Data RMSE", f"{train_rmse:.2f}")
        with col4:
            st.metric("Test-Data RMSE", f"{test_rmse:.2f}")
        
        # 模型穩定性分析
        stability_gap = train_r2 - test_r2
        if stability_gap > 0.1:
            st.warning(f"⚠️ 模型可能輕微過擬合，R²差距：{stability_gap:.4f}")
        else:
            st.success("✅ 隨機森林模型穩定性良好！")
        
        # 預測 vs 真實值圖
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_train, y=y_train_pred,
                mode='markers', name='Train-Data',
                marker=dict(color='blue', size=6, opacity=0.6)
            ))
            fig.add_trace(go.Scatter(
                x=[y_train.min(), y_train.max()],
                y=[y_train.min(), y_train.max()],
                mode='lines', name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                title=f"Train-Data: Predict vs Actual (R² = {train_r2:.4f})",
                xaxis_title="Actual",
                yaxis_title="Predict"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test, y=y_test_pred,
                mode='markers', name='Test-Data',
                marker=dict(color='green', size=6, opacity=0.6)
            ))
            fig.add_trace(go.Scatter(
                x=[y_test.min(), y_test.max()],
                y=[y_test.min(), y_test.max()],
                mode='lines', name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                title=f"Test-Data: Predict vs Actual (R² = {test_r2:.4f})",
                xaxis_title="Actual",
                yaxis_title="Predict"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 時間序列風格的預測圖
        st.markdown("### 📈 預測結果時間序列視圖")
        
        train_size = len(y_train)
        train_indices = np.arange(train_size)
        test_indices = np.arange(train_size, train_size + len(y_test))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=train_indices, y=y_train,
            mode='markers', name='Actual (Train)',
            marker=dict(color='blue', size=4)
        ))
        fig.add_trace(go.Scatter(
            x=test_indices, y=y_test,
            mode='markers', name='Actual (Test)',
            marker=dict(color='blue', size=4)
        ))
        fig.add_trace(go.Scatter(
            x=train_indices, y=y_train_pred,
            mode='markers', name='Predict (Train)',
            marker=dict(color='green', size=4)
        ))
        fig.add_trace(go.Scatter(
            x=test_indices, y=y_test_pred,
            mode='markers', name='Predict (Test)',
            marker=dict(color='red', size=4)
        ))
        
        fig.add_vline(
            x=train_size-0.5, 
            line_dash="dash", 
            line_color="black",
            annotation_text="Train/Test Split"
        )
        
        fig.update_layout(
            title="Random Forest: Predictions vs Actual Values",
            xaxis_title="Index",
            yaxis_title="Value",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 特徵重要性分析
        st.markdown("### 📊 特徵重要性分析")
        
        feature_importance = pd.DataFrame({
            '特徵': selected_features,
            '重要性': model.feature_importances_
        }).sort_values('重要性', ascending=False)
        
        fig = go.Figure(data=[
            go.Bar(x=feature_importance['特徵'], y=feature_importance['重要性'],
                   marker_color='forestgreen')
        ])
        fig.update_layout(
            title=f"隨機森林特徵重要性 ({n_estimators}棵樹)",
            xaxis_title="特徵",
            yaxis_title="重要性",
            xaxis_tickangle=45
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"**隨機森林配置：** {n_estimators}棵樹, 最大深度={max_depth}")

elif page == "🚀 梯度提升回歸":
    st.markdown('<h1 class="main-header">🚀 梯度提升回歸 (Gradient Boosting Regression, XGBoost)</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🧮 數學原理")
    
    st.markdown("### 🚀 梯度提升的核心思想")
    st.markdown("梯度提升通過迭代地添加弱學習器（通常是決策樹）來逐步改進模型：")
    
    st.latex(r'''
    F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)
    ''')
    
    st.markdown("其中：")
    st.latex(r'''
    \begin{align}
    F_m(x) &= \text{第m次迭代後的模型} \\
    h_m(x) &= \text{第m個弱學習器} \\
    \gamma_m &= \text{學習率}
    \end{align}
    ''')
    
    st.markdown("### 🎯 損失函數最小化")
    st.markdown("每個新的學習器都試圖修正前一個模型的殘差：")
    st.latex(r'''
    h_m(x) = \arg\min_h \sum_{i=1}^{n} L(y_i, F_{m-1}(x_i) + h(x_i))
    ''')
    
    st.markdown("### 📊 梯度下降視角")
    st.markdown("梯度提升可以看作在函數空間中的梯度下降：")
    st.latex(r'''
    r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}
    ''')
    
    st.markdown("### ⚡ XGBoost 關鍵改進")
    st.markdown("XGBoost 通過以下技術提升性能：")
    st.latex(r'''
    \text{目標函數} = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)
    ''')
    
    st.markdown("正則化項：")
    st.latex(r'''
    \Omega(f) = \gamma T + \frac{1}{2}\lambda \|w\|^2
    ''')
    
    # 優缺點並排顯示
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("### ✅ 優點")
        st.markdown("""
        - 🎯 **強大的預測能力**：通常在各種數據集上表現優秀
        - 🔧 **靈活的損失函數**：支持多種回歸和分類目標
        - 📊 **內建特徵重要性**：提供特徵選擇參考
        - 🛡️ **處理缺失值**：XGBoost能自動處理缺失數據
        - ⚖️ **對異常值較穩健**：樹模型的天然優勢
        - 🚀 **高效實現**：XGBoost針對速度和記憶體優化
        - 🎛️ **豐富的調參選項**：精細控制模型行為
        """)
    
    with col2:
        st.error("### ❌ 缺點")
        st.markdown("""
        - ⚠️ **容易過擬合**：需要仔細調參
        - 🎛️ **參數眾多**：調優複雜度高
        - ⏱️ **訓練時間較長**：迭代構建多個模型
        - 🔍 **模型不可解釋**：複雜的集成黑盒
        - 📈 **對噪聲敏感**：可能學習噪聲模式
        - 💿 **記憶體消耗大**：存儲多個模型
        """)
    
    st.markdown("### 🌟 算法演進對比")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("#### 🌲 傳統GBDT")
        st.markdown("""
        - **Scikit-learn實現**
        - 基礎梯度提升
        - 適合中小數據集
        - 調參相對簡單
        """)
    
    with col2:
        st.warning("#### ⚡ XGBoost")
        st.markdown("""
        - **工程優化版本**
        - 正則化 + 剪枝
        - 並行計算優化
        - 競賽界最愛
        """)
    
    with col3:
        st.success("#### 🚀 現代變種")
        st.markdown("""
        - **LightGBM**: 更快速度
        - **CatBoost**: 更好類別特徵
        - **HistGradientBoosting**: sklearn新版
        """)
    
    st.markdown("## 🎛️ 互動式實驗")
    
    X, y = get_current_data()
    
    # 參數設置
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider("提升器數量 (n_estimators)：", 50, 300, 100, 10)
        learning_rate = st.slider("學習率 (learning_rate)：", 0.01, 0.3, 0.1, 0.01)
        max_depth = st.slider("最大深度 (max_depth)：", 3, 10, 6)
    
    with col2:
        subsample = st.slider("樣本抽樣比例 (subsample)：", 0.5, 1.0, 0.8, 0.1)
        test_size = st.slider("測試集比例：", 0.1, 0.5, 0.2, 0.05)
        random_seed = st.slider("隨機種子：", 1, 100, 42)
    
    # 高級參數（可摺疊）
    with st.expander("🔧 高級參數設置"):
        col1, col2 = st.columns(2)
        with col1:
            min_samples_split = st.slider("最小分割樣本數：", 2, 20, 2)
            min_samples_leaf = st.slider("葉節點最小樣本數：", 1, 10, 1)
        with col2:
            max_features = st.selectbox("最大特徵數：", ["sqrt", "log2", None], index=0)
            validation_fraction = st.slider("驗證集比例 (early stopping)：", 0.1, 0.3, 0.1, 0.05)
    
    # 特徵選擇
    selected_features = st.multiselect(
        "選擇用於建模的特徵：",
        X.columns.tolist(),
        default=X.columns.tolist()
    )
    
    if len(selected_features) > 0 and len(X) > 0:
        X_selected = X[selected_features]
        
        # 檢查數據是否為空
        if len(X_selected) == 0:
            st.error("❌ 所選數據集為空，請檢查數據集。")
            st.stop()
        
        # 數據分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=test_size, random_state=random_seed
        )
        
        # 特徵標準化（可選，梯度提升通常不需要）
        normalize = st.checkbox("是否進行特徵標準化？", value=False, 
                               help="梯度提升通常不需要標準化，但某些情況下可能有幫助")
        
        if normalize:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
        
        # 梯度提升模型
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            validation_fraction=validation_fraction,
            n_iter_no_change=10,  # early stopping
            random_state=random_seed
        )
        
        # 訓練模型
        with st.spinner("🚀 梯度提升模型訓練中..."):
            model.fit(X_train_scaled, y_train)
        
        # 預測
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # 評估指標
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # 顯示結果
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Train-Data R²", f"{train_r2:.4f}")
        with col2:
            st.metric("Test-Data R²", f"{test_r2:.4f}")
        with col3:
            st.metric("Train-Data RMSE", f"{train_rmse:.2f}")
        with col4:
            st.metric("Test-Data RMSE", f"{test_rmse:.2f}")
        
        # 過擬合檢測與模型複雜度分析
        overfitting_gap = train_r2 - test_r2
        if overfitting_gap > 0.15:
            st.warning(f"⚠️ 梯度提升可能過擬合！R²差距：{overfitting_gap:.4f}")
            st.markdown("**建議：** 降低學習率、減少提升器數量或增加正則化")
        elif overfitting_gap < -0.05:
            st.info("ℹ️ 模型可能欠擬合，可以增加模型複雜度")
        else:
            st.success("✅ 梯度提升模型表現良好！")
        
        # 特殊指標
        n_estimators_used = model.n_estimators_
        st.info(f"**實際使用的提升器數量：** {n_estimators_used} (設定：{n_estimators})")
        
        # 預測 vs 真實值圖
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_train, y=y_train_pred,
                mode='markers', name='Train-Data',
                marker=dict(color='blue', size=6, opacity=0.6)
            ))
            fig.add_trace(go.Scatter(
                x=[y_train.min(), y_train.max()],
                y=[y_train.min(), y_train.max()],
                mode='lines', name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                title=f"Train-Data: Predict vs Actual (R² = {train_r2:.4f})",
                xaxis_title="Actual",
                yaxis_title="Predict"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test, y=y_test_pred,
                mode='markers', name='Test-Data',
                marker=dict(color='green', size=6, opacity=0.6)
            ))
            fig.add_trace(go.Scatter(
                x=[y_test.min(), y_test.max()],
                y=[y_test.min(), y_test.max()],
                mode='lines', name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                title=f"Test-Data: Predict vs Actual (R² = {test_r2:.4f})",
                xaxis_title="Actual",
                yaxis_title="Predict"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 時間序列風格的預測圖
        st.markdown("### 📈 預測結果時間序列視圖")
        
        train_size = len(y_train)
        train_indices = np.arange(train_size)
        test_indices = np.arange(train_size, train_size + len(y_test))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=train_indices, y=y_train,
            mode='markers', name='Actual (Train)',
            marker=dict(color='blue', size=4)
        ))
        fig.add_trace(go.Scatter(
            x=test_indices, y=y_test,
            mode='markers', name='Actual (Test)',
            marker=dict(color='blue', size=4)
        ))
        fig.add_trace(go.Scatter(
            x=train_indices, y=y_train_pred,
            mode='markers', name='Predict (Train)',
            marker=dict(color='green', size=4)
        ))
        fig.add_trace(go.Scatter(
            x=test_indices, y=y_test_pred,
            mode='markers', name='Predict (Test)',
            marker=dict(color='red', size=4)
        ))
        
        fig.add_vline(
            x=train_size-0.5, 
            line_dash="dash", 
            line_color="black",
            annotation_text="Train/Test Split"
        )
        
        fig.update_layout(
            title="Gradient Boosting: Predictions vs Actual Values",
            xaxis_title="Index",
            yaxis_title="Value",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 特徵重要性分析
        st.markdown("### 📊 特徵重要性分析")
        
        feature_importance = pd.DataFrame({
            '特徵': selected_features,
            '重要性': model.feature_importances_
        }).sort_values('重要性', ascending=False)
        
        fig = go.Figure(data=[
            go.Bar(x=feature_importance['特徵'], y=feature_importance['重要性'],
                   marker_color='orange')
        ])
        fig.update_layout(
            title=f"梯度提升特徵重要性 ({n_estimators_used}個提升器)",
            xaxis_title="特徵",
            yaxis_title="重要性",
            xaxis_tickangle=45
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 訓練過程視覺化
        st.markdown("### 📈 訓練過程分析")
        
        if hasattr(model, 'train_score_'):
            col1, col2 = st.columns(2)
            
            with col1:
                # 訓練過程損失函數
                iterations = range(1, len(model.train_score_) + 1)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(iterations), y=model.train_score_,
                    mode='lines', name='Training Score',
                    line=dict(color='blue')
                ))
                
                if hasattr(model, 'validation_scores_') and len(model.validation_scores_) > 0:
                    fig.add_trace(go.Scatter(
                        x=list(iterations[:len(model.validation_scores_)]), 
                        y=model.validation_scores_,
                        mode='lines', name='Validation Score',
                        line=dict(color='red')
                    ))
                
                fig.update_layout(
                    title="模型訓練過程",
                    xaxis_title="迭代次數",
                    yaxis_title="損失值",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 學習曲線說明
                st.markdown("#### 📚 學習曲線解讀")
                st.markdown("""
                - **藍線 (Training Score)**: 訓練集上的表現
                - **紅線 (Validation Score)**: 驗證集上的表現
                - **理想情況**: 兩線趨於收斂
                - **過擬合**: 藍線持續下降，紅線上升
                - **欠擬合**: 兩線都在高位平穩
                """)
                
                # 提升建議
                if overfitting_gap > 0.1:
                    st.warning("**調優建議**：降低學習率或增加正則化")
                else:
                    st.success("**模型狀態**：訓練過程健康！")
    else:
        st.warning("⚠️ 請選擇至少一個特徵進行梯度提升回歸實驗。")

elif page == "🎯 支持向量回歸":
    st.markdown('<h1 class="main-header">🎯 支持向量回歸 (Support Vector Regression)</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🧮 數學原理")
    
    st.markdown("### 🎯 SVR的基本思想")
    st.markdown("SVR通過尋找一個函數來擬合數據，允許預測誤差在ε範圍內：")
    
    st.latex(r'''
    f(x) = \sum_{i=1}^{n} (\alpha_i - \alpha_i^*) K(x_i, x) + b
    ''')
    
    st.markdown("### 📊 優化目標")
    st.latex(r'''
    \min_{w,b,\xi,\xi^*} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} (\xi_i + \xi_i^*)
    ''')
    
    st.markdown("約束條件：")
    st.latex(r'''
    \begin{align}
    y_i - w^T \phi(x_i) - b &\leq \varepsilon + \xi_i \\
    w^T \phi(x_i) + b - y_i &\leq \varepsilon + \xi_i^* \\
    \xi_i, \xi_i^* &\geq 0
    \end{align}
    ''')
    
    st.markdown("### 🔧 核函數")
    st.markdown("常用核函數：")
    st.latex(r'''
    \begin{align}
    \text{線性核：} K(x, x') &= x^T x' \\
    \text{多項式核：} K(x, x') &= (x^T x' + c)^d \\
    \text{RBF核：} K(x, x') &= \exp(-\gamma \|x - x'\|^2)
    \end{align}
    ''')
    
    # 優缺點並排顯示
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("### ✅ 優點")
        st.markdown("""
        - 📊 **處理高維數據效果好**：維度詛咒抵抗
        - 💿 **記憶體效率高**：僅存儲支持向量
        - 🔧 **核技巧處理非線性**：映射到高維空間
        - 🛡️ **對離群值相對穩健**：ε-tube機制
        - 🎯 **泛化能力強**：最大化邊界
        """)
    
    with col2:
        st.error("### ❌ 缺點")
        st.markdown("""
        - ⏱️ **大數據集訓練慢**：二次規劃問題
        - 🎛️ **參數調優複雜**：C、ε、核參數
        - 📊 **沒有概率輸出**：點估計
        - ⚖️ **對特徵縮放敏感**：需要標準化
        - 🔍 **核參數選擇困難**：需要經驗
        """)
    
    st.markdown("## 🎛️ 互動式實驗")
    
    X, y = get_current_data()
    
    # 參數設置
    col1, col2 = st.columns(2)
    
    with col1:
        kernel = st.selectbox("核函數類型：", ["rbf", "linear", "poly"])
        C = st.slider("懲罰參數 (C)：", 0.1, 100.0, 1.0, 0.1)
        epsilon = st.slider("ε-tube 寬度：", 0.01, 1.0, 0.1, 0.01)
    
    with col2:
        if kernel == "rbf":
            gamma = st.selectbox("Gamma 參數：", ["scale", "auto", 0.001, 0.01, 0.1, 1.0])
        elif kernel == "poly":
            degree = st.slider("多項式階數：", 2, 5, 3)
        test_size = st.slider("測試集比例：", 0.1, 0.5, 0.2, 0.05)
    
    # 特徵選擇
    selected_features = st.multiselect(
        "選擇用於建模的特徵：",
        X.columns.tolist(),
        default=X.columns.tolist()[:5]
    )
    
    if len(selected_features) > 0:
        X_selected = X[selected_features]
        
        # 數據分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=test_size, random_state=42
        )
        
        # 特徵標準化（SVR必須標準化）
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # SVR模型
        svr_params = {
            'kernel': kernel,
            'C': C,
            'epsilon': epsilon
        }
        
        if kernel == "rbf":
            svr_params['gamma'] = gamma
        elif kernel == "poly":
            svr_params['degree'] = degree
        
        model = SVR(**svr_params)
        
        # 訓練模型
        model.fit(X_train_scaled, y_train)
        
        # 預測
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # 評估指標
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # 顯示結果
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Train-Data R²", f"{train_r2:.4f}")
        with col2:
            st.metric("Test-Data R²", f"{test_r2:.4f}")
        with col3:
            st.metric("Train-Data RMSE", f"{train_rmse:.2f}")
        with col4:
            st.metric("Test-Data RMSE", f"{test_rmse:.2f}")
        
        # SVR特殊指標
        n_support = len(model.support_)
        support_ratio = n_support / len(X_train)
        st.info(f"**支持向量：** {n_support} 個 ({support_ratio:.1%} 的訓練樣本)")
        
        # 預測 vs 真實值圖
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_train, y=y_train_pred,
                mode='markers', name='Train-Data',
                marker=dict(color='blue', size=6, opacity=0.6)
            ))
            fig.add_trace(go.Scatter(
                x=[y_train.min(), y_train.max()],
                y=[y_train.min(), y_train.max()],
                mode='lines', name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                title=f"Train-Data: Predict vs Actual (R² = {train_r2:.4f})",
                xaxis_title="Actual",
                yaxis_title="Predict"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test, y=y_test_pred,
                mode='markers', name='Test-Data',
                marker=dict(color='green', size=6, opacity=0.6)
            ))
            fig.add_trace(go.Scatter(
                x=[y_test.min(), y_test.max()],
                y=[y_test.min(), y_test.max()],
                mode='lines', name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                title=f"Test-Data: Predict vs Actual (R² = {test_r2:.4f})",
                xaxis_title="Actual",
                yaxis_title="Predict"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 時間序列風格的預測圖
        st.markdown("### 📈 預測結果時間序列視圖")
        
        train_size = len(y_train)
        train_indices = np.arange(train_size)
        test_indices = np.arange(train_size, train_size + len(y_test))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=train_indices, y=y_train,
            mode='markers', name='Actual (Train)',
            marker=dict(color='blue', size=4)
        ))
        fig.add_trace(go.Scatter(
            x=test_indices, y=y_test,
            mode='markers', name='Actual (Test)',
            marker=dict(color='blue', size=4)
        ))
        fig.add_trace(go.Scatter(
            x=train_indices, y=y_train_pred,
            mode='markers', name='Predict (Train)',
            marker=dict(color='green', size=4)
        ))
        fig.add_trace(go.Scatter(
            x=test_indices, y=y_test_pred,
            mode='markers', name='Predict (Test)',
            marker=dict(color='red', size=4)
        ))
        
        fig.add_vline(
            x=train_size-0.5, 
            line_dash="dash", 
            line_color="black",
            annotation_text="Train/Test Split"
        )
        
        fig.update_layout(
            title=f"SVR ({kernel} kernel): Predictions vs Actual Values",
            xaxis_title="Index",
            yaxis_title="Value",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "📏 評價指標詳解":
    st.markdown('<h1 class="main-header">📏 評價指標詳解</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🎯 回歸評價指標概述")
    st.info("💡 評價指標幫助我們量化模型預測的準確性，選擇合適的指標對模型評估至關重要。")
    
    # 創建標籤頁式佈局
    tab1, tab2, tab3, tab4 = st.tabs(["🔵 R² 決定係數", "🟡 RMSE 均方根誤差", "🟢 MAE 平均絕對誤差", "🔴 MSE 均方誤差"])
    
    with tab1:
        st.markdown("### 🔵 R² (決定係數)")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("**R²衡量模型解釋變異的比例，是最常用的回歸評價指標。**")
            
            st.latex(r'''
            R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}
            ''')
            
            st.markdown("其中：")
            st.markdown("- $SS_{res}$ = 殘差平方和")
            st.markdown("- $SS_{tot}$ = 總平方和")
            st.markdown("- $\\bar{y}$ = 真實值平均數")
        
        with col2:
            st.success("### 📊 R² 範圍解釋")
            st.markdown("""
            - **R² = 1**: 🎯 完美預測
            - **R² = 0**: 📊 等同於預測平均值
            - **R² < 0**: ❌ 比預測平均值還差
            - **R² ≥ 0.7**: ✅ 通常認為是良好模型
            """)
            
            st.info("### 🎯 使用建議")
            st.markdown("""
            - ✅ 用於模型比較和選擇時
            - ✅ 向業務人員報告結果時
            - ✅ 需要解釋模型效果時
            - ⚠️ 注意避免過擬合問題
            """)
    
    with tab2:
        st.markdown("### 🟡 RMSE (均方根誤差)")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("**RMSE是最常用的回歸損失函數，對大誤差懲罰更重。**")
            
            st.latex(r'''
            RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
            ''')
            
            st.markdown("**特點：**")
            st.markdown("- 📏 單位與目標變數相同")
            st.markdown("- 🎯 對離群值敏感")
            st.markdown("- 📉 值越小表示模型越好")
            st.markdown("- 🔄 與MSE有直接關係")
        
        with col2:
            st.warning("### 📊 RMSE 特性")
            st.markdown("""
            - **優點**: 直觀易懂，單位一致
            - **缺點**: 對離群值非常敏感
            - **適用**: 大誤差代價高的場景
            - **解釋**: 平均預測誤差大小
            """)
            
            st.info("### 🎯 使用建議")
            st.markdown("""
            - ✅ 需要懲罰大誤差的情況下
            - ✅ 進行模型參數調優時
            - ✅ 需要直觀比較誤差大小時
            - ⚠️ 注意離群值的影響
            """)
    
    with tab3:
        st.markdown("### 🟢 MAE (平均絕對誤差)")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("**MAE對所有誤差一視同仁，對離群值相對穩健。**")
            
            st.latex(r'''
            MAE = \frac{1}{n} \sum_{i=1}^{n}|y_i - \hat{y}_i|
            ''')
            
            st.markdown("**特點：**")
            st.markdown("- 🛡️ 對離群值不敏感")
            st.markdown("- 📈 線性損失函數")
            st.markdown("- 📊 容易理解和計算")
            st.markdown("- ⚖️ 所有誤差權重相等")
        
        with col2:
            st.success("### 📊 MAE 特性")
            st.markdown("""
            - **優點**: 穩健性好，不受極值影響
            - **缺點**: 不可微分，優化困難
            - **適用**: 有離群值的數據
            - **解釋**: 預測誤差的中位數
            """)
            
            st.info("### 🎯 使用建議")
            st.markdown("""
            - ✅ 數據中存在離群值時
            - ✅ 所有誤差同等重要時
            - ✅ 需要穩健評估時
            - ⚠️ 不適合用於梯度優化
            """)
    
    with tab4:
        st.markdown("### 🔴 MSE (均方誤差)")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("**MSE是RMSE的平方，是許多算法的基礎損失函數。**")
            
            st.latex(r'''
            MSE = \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
            ''')
            
            st.markdown("**與其他指標的關係：**")
            st.latex(r'''RMSE = \sqrt{MSE}''')
            
            st.markdown("**特點：**")
            st.markdown("- 🔄 可微分，便於優化")
            st.markdown("- 🎯 懲罰大誤差")
            st.markdown("- 📊 數學性質良好")
        
        with col2:
            st.error("### 📊 MSE 特性")
            st.markdown("""
            - **優點**: 數學性質好，易優化
            - **缺點**: 單位不直觀（平方）
            - **適用**: 算法內部優化
            - **解釋**: 誤差平方的平均值
            """)
            
            st.info("### 🎯 使用建議")
            st.markdown("""
            - ✅ 作為算法內部損失函數時
            - ✅ 進行數學推導和分析時
            - ✅ 使用梯度下降優化時
            - ⚠️ 向他人報告時建議轉換為RMSE
            """)
    
    # 實際計算示例
    st.markdown("---")
    st.markdown("## 🧪 互動式指標計算器")
    st.markdown("使用真實數據來比較不同評價指標的表現")

    
    X, y = get_current_data()
    
    # 訓練一個簡單模型來展示指標
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 訓練線性回歸模型
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # 計算所有指標
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    # 顯示指標結果
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    with col2:
        st.metric("RMSE", f"{rmse:.2f}")
    with col3:
        st.metric("MAE", f"{mae:.2f}")
    with col4:
        st.metric("MSE", f"{mse:.2f}")
    
    # 可視化不同指標的表現
    st.markdown("### 📈 指標視覺化比較")
    
    # 創建殘差圖
    residuals = y_test - y_pred
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_pred, y=residuals,
            mode='markers', name='Residuals',
            marker=dict(color='blue', size=6, opacity=0.6)
        ))
        fig.add_hline(y=0, line_dash="dot", line_color="red")
        fig.update_layout(
            title="殘差圖 (Residual Plot)",
            xaxis_title="Predicted Values",
            yaxis_title="Residuals"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=residuals,
            nbinsx=30,
            name='Residuals Distribution',
            marker_color='lightblue'
        ))
        fig.update_layout(
            title="殘差分布",
            xaxis_title="Residuals",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 指標選擇建議
    st.markdown("## 🎯 如何選擇評價指標")
    
    # 創建四個不同場景的指標選擇建議
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("### ✅ 使用 R² 決定係數當...")
        st.markdown("""
        - 📊 **需要解釋模型解釋力**：告訴他人模型能解釋多少變異
        - 🏆 **模型比較與選擇**：快速比較不同模型的整體表現
        - 💼 **向業務人員報告**：百分比形式直觀易懂
        - 📈 **評估線性關係強度**：衡量特徵與目標的線性相關程度
        - 🎯 **設定性能基準**：R²>0.7通常認為是良好模型
        - ⚠️ **注意**: 樣本量小時可能過度樂觀
        """)
        
        st.error("### ✅ 使用 RMSE 均方根誤差當...")
        st.markdown("""
        - 💰 **大誤差代價高**：如房價預測、投資決策
        - 📏 **需要與目標同單位**：便於理解實際誤差大小
        - 🎛️ **模型調參優化**：梯度下降等優化算法友好
        - 📊 **正態分布假設下**：誤差符合正態分布時最適用
        - 🔍 **離群值敏感場景**：希望重點關注和懲罰大誤差
        - ⚠️ **避免**: 數據中有很多離群值時
        """)
    
    with col2:
        st.warning("### ✅ 使用 MAE 平均絕對誤差當...")
        st.markdown("""
        - 🛡️ **數據有離群值**：不希望極端值影響評估
        - ⚖️ **所有誤差同等重要**：大小誤差一視同仁
        - 📊 **中位數比平均值重要**：關注典型誤差而非極端情況
        - 🏥 **醫療診斷場景**：每個病人的誤差都同樣重要
        - 🎯 **穩健評估需求**：希望評估結果不受極值干擾
        - ⚠️ **避免**: 需要可微分優化的算法訓練時
        """)
        
        st.info("### ✅ 使用 MSE 均方誤差當...")
        st.markdown("""
        - 🧮 **算法內部損失函數**：線性回歸等算法的標準損失
        - 📐 **數學推導與分析**：數學性質良好，便於理論分析
        - 🔄 **梯度下降優化**：可微分特性適合優化算法
        - 🏗️ **模型構建階段**：訓練過程中的內部評估
        - 📚 **學術研究比較**：與其他研究保持一致的評估標準
        - ⚠️ **報告時轉換**: 向他人匯報時建議轉換為RMSE更直觀
        """)
    
    # 新增實際場景應用建議
    st.markdown("---")
    st.markdown("### 🏢 不同行業場景的指標選擇建議")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🏠 房地產預測")
        st.markdown("""
        - **主要指標**: RMSE
        - **原因**: 房價誤差以萬元為單位有實際意義
        - **輔助指標**: R²（解釋模型效果）
        - **避免**: MSE（平方單位不直觀）
        """)
        
        st.markdown("#### 💊 藥物劑量預測")
        st.markdown("""
        - **主要指標**: MAE
        - **原因**: 每位患者的劑量誤差同等重要
        - **輔助指標**: RMSE（檢查是否有異常大誤差）
        - **避免**: 只看R²（可能忽略個體差異）
        """)
    
    with col2:
        st.markdown("#### 📈 股票價格預測")
        st.markdown("""
        - **主要指標**: RMSE
        - **原因**: 大幅波動的懲罰權重要高
        - **輔助指標**: MAE（了解典型誤差）
        - **特殊考慮**: 可能需要MAPE（百分比誤差）
        """)
        
        st.markdown("#### 🌡️ 溫度預測")
        st.markdown("""
        - **主要指標**: MAE
        - **原因**: 溫度預測通常比較穩定，很少有極端離群值
        - **輔助指標**: R²（評估季節性規律的捕捉）
        - **注意**: 單位直觀易懂
        """)
    
    with col3:
        st.markdown("#### 🏭 工業品質控制")
        st.markdown("""
        - **主要指標**: RMSE + MAE
        - **原因**: 既要避免大偏差又要關注整體穩定性
        - **特殊需求**: 可能需要設定誤差容忍區間
        - **決策邏輯**: 雙重標準保證品質
        """)
        
        st.markdown("#### 🚗 燃油效率預測")
        st.markdown("""
        - **主要指標**: R²
        - **原因**: 消費者更關心模型能否解釋燃油差異
        - **輔助指標**: MAE（實際里程誤差）
        - **應用**: 幫助購車決策制定
        """)

elif page == "🔄 交叉驗證與穩定性":
    st.markdown('<h1 class="main-header">🔄 交叉驗證與穩定性</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🎯 什麼是交叉驗證？")
    st.markdown("交叉驗證是評估模型泛化能力的重要技術，通過多次訓練和驗證來獲得更可靠的性能估計。")
    
    st.markdown("### 📊 K-fold交叉驗證")
    st.latex(r'''
    CV(k) = \frac{1}{k} \sum_{i=1}^{k} L(f^{-i}, D_i)
    ''')
    
    st.markdown("其中：")
    st.markdown("- $k$ = 折數（通常為5或10）")
    st.markdown("- $f^{-i}$ = 在除第i折外的數據上訓練的模型")
    st.markdown("- $D_i$ = 第i折的驗證數據")
    st.markdown("- $L$ = 損失函數")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("### 📋 交叉驗證優點")
        st.markdown("""
        - 🎯 **更可靠的評估**：減少隨機性影響
        - 📊 **充分利用數據**：每個樣本都參與驗證
        - 🔄 **穩定性分析**：了解模型在不同數據上的表現
        - ⚖️ **避免偏見**：減少數據分割的偏差
        """)
    
    with col2:
        st.warning("### ⚠️ 注意事項")
        st.markdown("""
        - ⏱️ **計算成本高**：需要訓練k次模型
        - 📈 **時間序列數據**：需要特殊處理
        - 🔗 **數據洩漏**：避免未來信息洩漏
        - 🎛️ **參數調優**：嵌套交叉驗證
        """)
    
    st.markdown("## 🧪 互動式實驗")
    
    X, y = get_current_data()
    
    # 參數設置
    col1, col2 = st.columns(2)
    
    with col1:
        cv_folds = st.slider("K-fold折數：", 3, 10, 5)
        model_type = st.selectbox("選擇模型：", [
            "Linear Regression", "Ridge", "Lasso", "Random Forest", "SVR"
        ])
    
    with col2:
        random_seed = st.slider("隨機種子：", 1, 100, 42)
        selected_features = st.multiselect(
            "選擇特徵：",
            X.columns.tolist(),
            default=X.columns.tolist()[:5]
        )
    
    if len(selected_features) > 0 and len(X) > 0:
        X_selected = X[selected_features]
        
        # 檢查數據是否為空
        if len(X_selected) == 0:
            st.error("❌ 所選數據集為空，請檢查數據集。")
            st.stop()
        
        # 標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        # 選擇模型
        if model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Ridge":
            model = Ridge(alpha=1.0)
        elif model_type == "Lasso":
            model = Lasso(alpha=1.0, max_iter=2000)
        elif model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=50, random_state=random_seed)
        else:  # SVR
            model = SVR(kernel='rbf', C=1.0)
        
        # 執行交叉驗證
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
        cv_scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='r2')
        cv_rmse_scores = -cross_val_score(model, X_scaled, y, cv=kfold, 
                                        scoring='neg_root_mean_squared_error')
        
        # 顯示結果
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("平均 R²", f"{cv_scores.mean():.4f}")
        with col2:
            st.metric("R² 標準差", f"{cv_scores.std():.4f}")
        with col3:
            st.metric("平均 RMSE", f"{cv_rmse_scores.mean():.2f}")
        with col4:
            st.metric("RMSE 標準差", f"{cv_rmse_scores.std():.2f}")
        
        # 穩定性分析
        stability_coefficient = cv_scores.std() / abs(cv_scores.mean())
        if stability_coefficient < 0.1:
            st.success(f"✅ 模型非常穩定！變異係數：{stability_coefficient:.3f}")
        elif stability_coefficient < 0.2:
            st.info(f"ℹ️ 模型較為穩定。變異係數：{stability_coefficient:.3f}")
        else:
            st.warning(f"⚠️ 模型穩定性較差，變異係數：{stability_coefficient:.3f}")
        
        # 視覺化結果
        st.markdown("### 📊 交叉驗證結果視覺化")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # R²分布圖
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=cv_scores,
                name=f'{model_type} R²',
                boxpoints='all',
                marker_color='blue'
            ))
            fig.update_layout(
                title=f"{cv_folds}-Fold 交叉驗證 R² 分布",
                yaxis_title="R² Score"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # RMSE分布圖
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=cv_rmse_scores,
                name=f'{model_type} RMSE',
                boxpoints='all',
                marker_color='red'
            ))
            fig.update_layout(
                title=f"{cv_folds}-Fold 交叉驗證 RMSE 分布",
                yaxis_title="RMSE"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 折次比較
        st.markdown("### 📈 各折次詳細結果")
        
        results_df = pd.DataFrame({
            '折次': [f'Fold {i+1}' for i in range(cv_folds)],
            'R²': cv_scores,
            'RMSE': cv_rmse_scores
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results_df['折次'], y=results_df['R²'],
            mode='lines+markers', name='R²',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=results_df['折次'], y=results_df['RMSE'],
            mode='lines+markers', name='RMSE',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig.update_layout(
                title="各折次 R² 變化",
                xaxis_title="折次",
                yaxis_title="R² Score"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig2.update_layout(
                title="各折次 RMSE 變化",
                xaxis_title="折次",
                yaxis_title="RMSE"
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # 詳細結果表格
        st.dataframe(results_df, use_container_width=True)
    else:
        st.warning("⚠️ 請選擇至少一個特徵進行交叉驗證實驗。")

elif page == "🏆 模型綜合比較":
    st.markdown('<h1 class="main-header">🏆 模型綜合比較</h1>', unsafe_allow_html=True)
    
    st.markdown("## 🎯 模型比較實驗")
    st.markdown("在相同數據集上比較所有回歸模型的性能，幫助您選擇最適合的模型。")
    
    X, y = get_current_data()
    
    # 參數設置
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("測試集比例：", 0.1, 0.5, 0.2, 0.05)
        cv_folds = st.slider("交叉驗證折數：", 3, 10, 5)
    
    with col2:
        selected_features = st.multiselect(
            "選擇特徵：",
            X.columns.tolist(),
            default=X.columns.tolist()
        )
        random_seed = st.slider("隨機種子：", 1, 100, 42)
    
    if len(selected_features) > 0 and len(X) > 0:
        X_selected = X[selected_features]
        
        # 檢查數據是否為空
        if len(X_selected) == 0:
            st.error("❌ 所選數據集為空，請檢查數據集。")
            st.stop()
        
        # 數據分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=test_size, random_state=random_seed
        )
        
        # 標準化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 定義所有模型
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0),
            'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=random_seed),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_seed),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_seed),
            'SVR': SVR(kernel='rbf', C=1.0)
        }
        
        # 執行模型比較
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (name, model) in enumerate(models.items()):
            status_text.text(f'正在訓練 {name}...')
            
            # 訓練模型
            if name in ['SVR', 'Ridge', 'Lasso', 'ElasticNet']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                # 交叉驗證
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='r2')
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                # 交叉驗證
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
            
            # 計算指標
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            results.append({
                '模型': name,
                'Test R²': r2,
                'Test RMSE': rmse,
                'Test MAE': mae,
                'CV R² (mean)': cv_scores.mean(),
                'CV R² (std)': cv_scores.std(),
                '穩定性': 'Good' if cv_scores.std() < 0.1 else 'Fair' if cv_scores.std() < 0.2 else 'Poor'
            })
            
            progress_bar.progress((i + 1) / len(models))
        
        status_text.text('模型比較完成！')
        
        # 結果DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Test R²', ascending=False)
        
        st.markdown("### 📊 模型性能比較表")
        
        # 突出顯示最佳模型
        def highlight_best(s):
            if s.name in ['Test R²', 'CV R² (mean)']:
                max_val = s.max()
                return ['background-color: lightgreen' if v == max_val else '' for v in s]
            elif s.name in ['Test RMSE', 'Test MAE', 'CV R² (std)']:
                min_val = s.min()
                return ['background-color: lightgreen' if v == min_val else '' for v in s]
            else:
                return ['' for _ in s]
        
        styled_df = results_df.style.apply(highlight_best, axis=0)
        st.dataframe(styled_df, use_container_width=True)
        
        # 性能視覺化
        st.markdown("### 📈 模型性能視覺化")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # R²比較
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=results_df['模型'],
                y=results_df['Test R²'],
                name='Test R²',
                marker_color='blue'
            ))
            fig.add_trace(go.Bar(
                x=results_df['模型'],
                y=results_df['CV R² (mean)'],
                name='CV R² (mean)',
                marker_color='lightblue'
            ))
            fig.update_layout(
                title="R² 分數比較",
                xaxis_title="模型",
                yaxis_title="R² Score",
                xaxis_tickangle=45
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # RMSE比較
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=results_df['模型'],
                y=results_df['Test RMSE'],
                marker_color='red'
            ))
            fig.update_layout(
                title="RMSE 比較 (越低越好)",
                xaxis_title="模型",
                yaxis_title="RMSE",
                xaxis_tickangle=45
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 新增MAE和MSE比較
        col1, col2 = st.columns(2)
        
        with col1:
            # MAE比較
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=results_df['模型'],
                y=results_df['Test MAE'],
                marker_color='orange'
            ))
            fig.update_layout(
                title="MAE 比較 (越低越好)",
                xaxis_title="模型",
                yaxis_title="MAE",
                xaxis_tickangle=45
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # MSE比較
            mse_values = results_df['Test RMSE'] ** 2  # MSE = RMSE^2
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=results_df['模型'],
                y=mse_values,
                marker_color='purple'
            ))
            fig.update_layout(
                title="MSE 比較 (越低越好)",
                xaxis_title="模型",
                yaxis_title="MSE",
                xaxis_tickangle=45
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 雷達圖比較前3名模型
        st.markdown("### 🎯 前三名模型雷達圖比較")
        
        top3_models = results_df.head(3)
        
        # 標準化指標 (0-1範圍)
        max_r2 = results_df['Test R²'].max()
        min_rmse = results_df['Test RMSE'].min()
        max_rmse = results_df['Test RMSE'].max()
        min_mae = results_df['Test MAE'].min()
        max_mae = results_df['Test MAE'].max()
        max_std = results_df['CV R² (std)'].max()
        
        fig = go.Figure()
        
        for _, row in top3_models.iterrows():
            # 標準化指標
            r2_norm = row['Test R²'] / max_r2
            rmse_norm = 1 - (row['Test RMSE'] - min_rmse) / (max_rmse - min_rmse)  # 反轉，越低越好
            mae_norm = 1 - (row['Test MAE'] - min_mae) / (max_mae - min_mae)  # 反轉，越低越好
            stability_norm = 1 - row['CV R² (std)'] / max_std  # 反轉，越低越好
            
            fig.add_trace(go.Scatterpolar(
                r=[r2_norm, rmse_norm, mae_norm, stability_norm, r2_norm],
                theta=['R²', 'RMSE', 'MAE', 'Stability', 'R²'],
                fill='toself',
                name=row['模型']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="前三名模型綜合性能比較"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 模型選擇建議
        st.markdown("### 🎯 模型選擇建議")
        
        best_model = results_df.iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"### 🏆 最佳模型：{best_model['模型']}")
            st.markdown(f"""
            - **Test R²**: {best_model['Test R²']:.4f}
            - **Test RMSE**: {best_model['Test RMSE']:.2f}
            - **交叉驗證穩定性**: {best_model['穩定性']}
            """)
        
        with col2:
            st.info("### 💡 選擇準則")
            st.markdown("""
            - **準確性**: R²越高越好
            - **穩定性**: 交叉驗證標準差越小越好
            - **簡單性**: 簡單模型更易解釋
            - **效率**: 考慮訓練和預測時間
            """)
    else:
        st.warning("⚠️ 請選擇至少一個特徵進行模型比較實驗。")

else:
    st.markdown(f"# {page}")
    st.info("此頁面正在開發中，敬請期待！")

# 應用結束