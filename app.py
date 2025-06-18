import streamlit as st
from streamlit_ketcher import st_ketcher
import pandas as pd
import os
import re
import glob
import pickle
import json
import pubchempy as pcp
import joblib
import random
from Bio import Entrez, Medline
from anthropic import Anthropic  # 如果使用Claude
import google.generativeai as palm  # 如果使用PaLM
import string
import requests
import time
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
import networkx as nx
import pandas as pd
from rdkit import Chem
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import learning_curve
from datetime import datetime
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import shap
from Bio import Entrez
from openai import OpenAI
from io import StringIO
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import seaborn as sns
import matplotlib.pyplot as plt
# Set page configuration
st.set_page_config(page_title="2025CADD课程实践", page_icon="🔬")
import os


from utils import *

# --- Streamlit UI ---
sidebar_option = st.sidebar.selectbox("选择功能", ["首页", "数据展示", "模型训练", "模型预测", "查看已有项目", "知识获取","药物结构可视化"])

# 首页
if sidebar_option == "首页":
    # Set header
    st.markdown("""
        <h1 style="text-align: center; color: #4CAF50;">2025CADD课程实践</h1>
        <p style="text-align: center; font-size: 18px; color: #555;">欢迎来到唐诗骐的药物成药性预测平台！选择您感兴趣的功能开始使用。</p>
    """, unsafe_allow_html=True)
    # Add some styling
    st.markdown("""
        <style>
            .card {
                background-color: #f9f9f9;
                border: 2px solid #d1d1d1;
                border-radius: 10px;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                padding: 20px;
                margin-top: 20px;
                text-align: center;
                font-size: 16px;
            }
            .card:hover {
                background-color: #e8f4f8;
                cursor: pointer;
            }
            .card-title {
                font-size: 20px;
                font-weight: bold;
                color: #4CAF50;
            }
            .card-description {
                color: #666;
                font-size: 14px;
                margin-top: 10px;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Add columns for a cleaner layout
    col1, col2, col3 = st.columns(3)
    
    # Define the clickable cards (functionality links)
    with col1:
        st.markdown("""
            <div class="card" onclick="window.location.href='#'">
                <div class="card-title">数据展示</div>
                <div class="card-description">查看数据集概况并生成相关的统计图表。</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="card" onclick="window.location.href='#'">
                <div class="card-title">模型训练</div>
                <div class="card-description">训练机器学习模型并评估性能(AUC曲线等)。</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="card" onclick="window.location.href='#'">
                <div class="card-title">模型预测</div>
                <div class="card-description">基于训练好的模型在独立测试集检查效果。</div>
            </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="card" onclick="window.location.href='#'">
                <div class="card-title">查看已有项目</div>
                <div class="card-description">查看您之前创建的项目和模型评估结果。</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="card" onclick="window.location.href='#'">
                <div class="card-title">知识获取</div>
                <div class="card-description">获取文献中的药物成药性信息，支持文献摘要提取。</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="card" onclick="window.location.href='#'">
                <div class="card-title">药物结构可视化</div>
                <div class="card-description">可视化感兴趣的药物结构，支持交互式编辑与下载。</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
        <footer style="text-align: center; margin-top: 50px;">
            <p style="font-size: 14px; color: #888;">© 2025 计算机辅助药物设计课程实践平台 | 由TJCADD团队开发</p>
        </footer>
    """, unsafe_allow_html=True)

# 功能1: 展示数据
elif sidebar_option == "数据展示":
    st.title("数据展示")
    csv_files = glob.glob("./data/*.csv")
    dataset_choice = st.sidebar.selectbox("选择数据集", [os.path.basename(file) for file in csv_files])
    selected_file = csv_files[[os.path.basename(file) for file in csv_files].index(dataset_choice)]
    data = pd.read_csv(selected_file)
    display_data_summary(data)

# 主流程
elif sidebar_option == "模型训练":
    st.title("模型训练")
    
    # Step 1: 创建项目
    st.header("第1步：创建项目")
    project_name = st.text_input("输入项目名称", "my_project")
    if st.button("创建项目"):
        project_dir = create_project_directory(project_name)
        st.session_state['project_dir'] = project_dir
        st.success(f"项目目录创建成功：{project_dir}")

    # Step 2: 数据预处理
    st.header("第2步：数据预处理")
    if 'project_dir' in st.session_state:
        # 数据集选择
        csv_files = glob.glob("./data/*.csv")
        dataset_choice = st.selectbox("选择数据集", [os.path.basename(file) for file in csv_files])
        selected_file = csv_files[[os.path.basename(file) for file in csv_files].index(dataset_choice)]
        
        if st.button("加载数据"):
            data = load_dataset(selected_file)
            if data is not None:
                st.session_state['data'] = data
                st.write("数据预览：")
                st.write(data.head())
                st.success("数据加载完成！")

        if 'data' in st.session_state:
            data = st.session_state['data']
            
            # 特征类型选择
            feature_type = st.radio(
                "选择特征类型",
                ["分子指纹(SMILES)", "分子指纹(分子式)", "自选特征列"]
            )
            
            # 选择标签列
            label_column = st.selectbox("选择标签列", data.columns.tolist())
            
            if feature_type == "分子指纹(SMILES)":
                smiles_column = st.selectbox(
                    "选择SMILES列",
                    [col for col in data.columns if 'smiles' in col.lower()]
                )
                
                if st.button("生成SMILES指纹"):
                    output_file = process_smiles_fingerprint(
                        data, smiles_column, label_column, st.session_state['project_dir']
                    )
                    if output_file:
                        st.session_state['processed_data_path'] = output_file
                        st.success(f"SMILES指纹数据已保存到：{output_file}")
                feature_columns = smiles_column
            elif feature_type == "分子指纹(分子式)":
                formula_column = st.selectbox(
                    "选择分子式列",
                    [col for col in data.columns if 'formula' in col.lower()]
                )
                
                if st.button("生成分子式指纹"):
                    output_file = process_formula_fingerprint(
                        data, formula_column, label_column, st.session_state['project_dir']
                    )
                    if output_file:
                        st.session_state['processed_data_path'] = output_file
                        st.success(f"分子式指纹数据已保存到：{output_file}")
                feature_columns = formula_column
            else:  # 自选特征列
                feature_columns = st.multiselect(
                    "选择特征列",
                    [col for col in data.columns if col != label_column]
                )
                
                if st.button("保存特征数据"):
                    output_file = process_selected_features(
                        data, feature_columns, label_column, st.session_state['project_dir']
                    )
                    if output_file:
                        st.session_state['processed_data_path'] = output_file
                        st.success(f"特征数据已保存到：{output_file}")
                feature_columns = feature_columns
    # Step 3: 模型训练
    st.header("第3步：模型训练")
    if 'processed_data_path' in st.session_state:
        # 加载训练数据
        X, y = load_training_data(st.session_state['processed_data_path'])
        
        if X is not None and y is not None:
            # 模型选择
            model_type = st.selectbox(
                "选择模型类型",
                ["随机森林", "XGBoost", "LightGBM", "支持向量机", "神经网络"]
            )
            
            # 获取模型参数
            model_params = get_model_params(model_type)
            
            # 训练设置
            test_size = st.slider("测试集比例", 0.1, 0.4, 0.2)
            random_state = st.number_input("随机种子", value=42)
            
            if st.button("开始训练"):
                with st.spinner("模型训练中..."):
                    # 数据集分割
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    
                    # 创建模型
                    model = create_model(model_type, model_params)
                    
                    if model is not None:
                        # 训练模型
                        model.fit(X_train, y_train)
                        
                        # 获取特征名称
                        feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
                        
                        # 使用综合可视化函数
                        visualize_results(model, X_test, y_test, feature_names,st.session_state['project_dir'] )
                        
                        # 保存模型和结果
                        if save_model_and_results(
                            model, 
                            {
                                'model_type': model_type,
                                'model_params': model_params,
                                'test_size': test_size,
                                'random_state': random_state,
                                'feature_type': feature_type,
                                'feature_columns': feature_columns   # 如果是自选特征，则保存特征列名
                            },
                            st.session_state['project_dir']
                        ):
                            st.success("模型和训练结果已保存！")

    else:
        st.warning("请先完成数据预处理步骤")
                    
                    
# # 功能2: 模型训练
# elif sidebar_option == "模型训练":
#     st.title("模型训练")
#     csv_files = glob.glob("./data/*.csv")
#     dataset_choice = st.sidebar.selectbox("选择数据集", [os.path.basename(file) for file in csv_files])
#     selected_file = csv_files[[os.path.basename(file) for file in csv_files].index(dataset_choice)]
#     data = pd.read_csv(selected_file)
#     label_column = st.sidebar.selectbox("选择标签列", data.columns.tolist())

#     rf_params = {
#         'n_estimators': st.sidebar.slider("随机森林 n_estimators", 50, 500, 100),
#         'max_depth': st.sidebar.slider("随机森林 max_depth", 1, 30, 3),
#         'max_features': st.sidebar.slider("随机森林 max_features", 0.1, 1.0, 0.2)
#     }

#     if st.sidebar.button("开始训练模型"):
#         project_dir = create_project_directory()
#         fp_file = save_input_data_with_fingerprint(data, project_dir, label_column)
#         model, acc, roc_auc = train_and_save_model(fp_file, project_dir, rf_params)
#         st.write(f"训练完成，模型准确率(Accuracy): {acc:.4f}; 模型AUC: {roc_auc:.4f}")
#         st.success(f"模型已保存到：{os.path.join(project_dir, 'model.pkl')}")

elif sidebar_option == "模型预测":
    st.title("模型预测")
    
    # 1. 加载项目和模型
    projects = glob.glob('./projects/*')
    if not projects:
        st.error("没有找到已训练的项目")
    else:
        project_names = [os.path.basename(project) for project in projects]
        project_name = st.selectbox("选择预测项目", project_names)
        selected_project_dir = os.path.join("./projects", project_name)
        
        # 查找带时间戳的模型和配置文件
        model_path = glob.glob(os.path.join(selected_project_dir, "model_*.pkl"))[0]
        config_path = glob.glob(os.path.join(selected_project_dir, "training_config_*.json"))[0]
        
        if os.path.exists(model_path) and os.path.exists(config_path):
            model = load_saved_model(model_path)
            with open(config_path, 'r') as f:
                training_config = json.load(f)
            
            st.info(f"模型类型: {training_config['model_type']}")
            st.info(f"特征类型: {training_config['feature_type']}")

            # 2. 数据输入选择
            input_method = st.radio("选择数据输入方式", ["从data目录选择", "上传新文件"])
            
            test_data = None
            if input_method == "从data目录选择":
                csv_files = glob.glob("./data/*.csv")
                if csv_files:
                    dataset_choice = st.selectbox("选择数据集", 
                                                [os.path.basename(file) for file in csv_files])
                    file_path = csv_files[[os.path.basename(file) for file in csv_files].index(dataset_choice)]
                    test_data = pd.read_csv(file_path)
            else:
                uploaded_file = st.file_uploader("上传CSV文件", type=['csv'])
                if uploaded_file:
                    test_data = pd.read_csv(uploaded_file)

            if test_data is not None:
                st.write("数据预览:", test_data.head())
                
                # 3. 特征处理
                X_test = None
                predictions = None
                prob = None
                
                if training_config['feature_type'] == "分子指纹(SMILES)":
                    smiles_cols = [col for col in test_data.columns if 'smiles' in col.lower()]
                    if smiles_cols:
                        smiles_col = st.selectbox("选择SMILES列", smiles_cols)
                        if st.button("开始预测"):
                            with st.spinner("正在处理SMILES并进行预测..."):
                                X_test = np.vstack(test_data[smiles_col].apply(mol_to_fp))
                                st.session_state['X_test'] = X_test
                
                elif training_config['feature_type'] == "分子指纹(分子式)":
                    formula_cols = [col for col in test_data.columns if 'formula' in col.lower()]
                    if formula_cols:
                        formula_col = st.selectbox("选择分子式列", formula_cols)
                        if st.button("开始预测"):
                            with st.spinner("正在处理分子式并进行预测..."):
                                X_test = np.vstack(test_data[formula_col].apply(formula_to_fp))
                                st.session_state['X_test'] = X_test
                
                else:  # 自选特征列
                    required_features = training_config.get('feature_columns', [])
                    if all(col in test_data.columns for col in required_features):
                        if st.button("开始预测"):
                            with st.spinner("正在进行预测..."):
                                X_test = test_data[required_features].values
                                st.session_state['X_test'] = X_test
                    else:
                        st.error("测试数据缺少必要的特征列")
                
                # 4. 预测和结果展示
                if 'X_test' in st.session_state:
                    X_test = st.session_state['X_test']
                    predictions = model.predict(X_test)
                    prob = model.predict_proba(X_test)[:, 1]
                    
                    # 保存到session state
                    st.session_state['predictions'] = predictions
                    st.session_state['prob'] = prob
                    
                    # 保存预测结果
                    results_df = test_data.copy()
                    results_df['预测标签'] = predictions
                    results_df['预测概率'] = prob
                    
                    # 结果展示
                    st.header("预测结果")
                    st.write(results_df)
                    
                    # 结果可视化
                    fig, ax = plt.subplots()
                    sns.histplot(data=prob, bins=30)
                    plt.title("Probability Distribution")
                    st.pyplot(fig)
                    
                    # 下载结果
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "下载预测结果",
                        csv,
                        f"predictions_{project_name}.csv",
                        "text/csv"
                    )
                    
                    # 保存到项目目录
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(selected_project_dir, f'predictions_{timestamp}.csv')
                    results_df.to_csv(save_path, index=False)
                    st.success(f"预测结果已保存: {save_path}")

                    # 5. 模型评估部分
                    st.header("模型评估")
                    label_col = st.selectbox("选择标签列进行评估", test_data.columns, key='label_select')
                    
                    if st.button("开始评估", key='eval_button'):
                        try:
                            y_true = test_data[label_col].values
                            current_predictions = st.session_state['predictions']
                            current_prob = st.session_state['prob']
                            
                            # 计算评估指标
                            metrics = {
                                'Accuracy': accuracy_score(y_true, current_predictions),
                                'Precision': precision_score(y_true, current_predictions),
                                'Recall': recall_score(y_true, current_predictions),
                                'F1 Score': f1_score(y_true, current_predictions),
                                'ROC AUC': roc_auc_score(y_true, current_prob)
                            }
                            
                            # 显示评估指标
                            col1, col2 = st.columns(2)
                            with col1:
                                metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
                                st.table(metrics_df.style.format({'Value': '{:.4f}'}))
                                
                            with col2:
                                cm = confusion_matrix(y_true, current_predictions)
                                fig_cm = plt.figure(figsize=(8, 6))
                                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                                plt.xlabel('Predicted')
                                plt.ylabel('True')
                                plt.title('Confusion Matrix')
                                st.pyplot(fig_cm)
                                plt.close()
                            
                            # ROC曲线
                            fig_roc = plt.figure(figsize=(8, 6))
                            fpr, tpr, _ = roc_curve(y_true, current_prob)
                            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                                    label=f'ROC curve (AUC = {metrics["ROC AUC"]:.2f})')
                            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                            plt.xlim([0.0, 1.0])
                            plt.ylim([0.0, 1.05])
                            plt.xlabel('False Positive Rate')
                            plt.ylabel('True Positive Rate')
                            plt.title('Receiver Operating Characteristic')
                            plt.legend(loc="lower right")
                            st.pyplot(fig_roc)
                            plt.close()
                            
                            # 保存评估结果
                            eval_results = {
                                'metrics': metrics,
                                'confusion_matrix': cm.tolist(),
                                'roc_curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
                            }
                            eval_path = os.path.join(selected_project_dir, f'evaluation_{timestamp}.json')
                            with open(eval_path, 'w') as f:
                                json.dump(eval_results, f, indent=4)
                            st.success(f"评估结果已保存: {eval_path}")
                            
                        except Exception as e:
                            st.error(f"评估过程中出现错误: {str(e)}")
                            st.write("请确保选择了正确的标签列，并且已经完成模型预测")
                            
        else:
            st.error("未找到模型文件或配置文件")
            
            
            
            
            

# 功能4: 查看已有项目
elif sidebar_option == "查看已有项目":
    st.title("查看已有项目")
    projects = glob.glob('./projects/*')
    if not projects:
        st.write("没有找到项目")
    else:
        project_names = [os.path.basename(project) for project in projects]
        project_name = st.selectbox("选择一个项目查看", project_names)
        selected_project_dir = os.path.join("./projects", project_name)

        # 显示项目信息
        st.write("### 项目信息")
        st.write(f"项目名称: {project_name}")

        # 显示特征数据
        if os.path.exists(os.path.join(selected_project_dir, "selected_features.csv")):
            data = pd.read_csv(os.path.join(selected_project_dir, "selected_features.csv"))
            st.write("### 特征数据预览")
            st.dataframe(data.head())

        # 创建两列布局显示评估图表
        col1, col2 = st.columns(2)

        with col1:
            # ROC曲线
            if os.path.exists(os.path.join(selected_project_dir, "roc_curve.png")):
                st.write("### ROC曲线")
                st.image(os.path.join(selected_project_dir, "roc_curve.png"))

            # 混淆矩阵
            if os.path.exists(os.path.join(selected_project_dir, "confusion_matrix.png")):
                st.write("### 混淆矩阵")
                st.image(os.path.join(selected_project_dir, "confusion_matrix.png"))

        with col2:
            # 特征重要性图
            if os.path.exists(os.path.join(selected_project_dir, "feature_importance.png")):
                st.write("### 特征重要性")
                st.image(os.path.join(selected_project_dir, "feature_importance.png"))

            # 学习曲线
            if os.path.exists(os.path.join(selected_project_dir, "learning_curve.png")):
                st.write("### 学习曲线")
                st.image(os.path.join(selected_project_dir, "learning_curve.png"))

        # 显示评估指标
        if os.path.exists(os.path.join(selected_project_dir, "evaluation_metrics.json")):
            st.write("### 模型评估指标")
            with open(os.path.join(selected_project_dir, "evaluation_metrics.json"), 'r') as f:
                metrics = json.load(f)
            
            # 创建四列显示主要指标
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("准确率", f"{metrics['Value']['Accuracy']:.4f}")
            with col2:
                st.metric("精确率", f"{metrics['Value']['Precision']:.4f}")
            with col3:
                st.metric("召回率", f"{metrics['Value']['Recall']:.4f}")
            with col4:
                st.metric("F1分数", f"{metrics['Value']['F1 Score']:.4f}")

        # 添加下载按钮
        st.write("### 下载项目文件")
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists(os.path.join(selected_project_dir, "selected_features.csv")):
                with open(os.path.join(selected_project_dir, "selected_features.csv"), 'rb') as f:
                    st.download_button(
                        label="下载特征数据",
                        data=f,
                        file_name="selected_features.csv",
                        mime="text/csv"
                    )
        
        with col2:
            if os.path.exists(os.path.join(selected_project_dir, "model.pkl")):
                with open(os.path.join(selected_project_dir, "model.pkl"), 'rb') as f:
                    st.download_button(
                        label="下载模型文件",
                        data=f,
                        file_name="model.pkl",
                        mime="application/octet-stream"
                    )


# 功能5: 知识获取
elif sidebar_option == "知识获取":
    st.title("化合物成药性文献分析")
    
    # API配置
    with st.sidebar:
        model_option = st.selectbox(
            "选择分析模型",
            ["GPT (OpenAI)", "星火 (讯飞)", "Claude (Anthropic)", "PaLM (Google)"]
        )
        
        # 根据选择的模型显示对应的API配置
        if model_option == "GPT (OpenAI)":
            api_key = st.text_input("OpenAI API Key", type="password")
            client = OpenAI(api_key=api_key) if api_key else None
            analyze_func = analyze_druggability_gpt
            
        elif model_option == "星火 (讯飞)":
            api_key = st.text_input("星火 API Bearer Token", type="password",
                help="格式: Bearer XXXXXXXX")
            client = api_key
            analyze_func = analyze_druggability_spark
            
        elif model_option == "Claude (Anthropic)":
            api_key = st.text_input("Anthropic API Key", type="password")
            client = Anthropic(api_key=api_key) if api_key else None
            analyze_func = analyze_druggability_claude
            
        else:  # PaLM
            api_key = st.text_input("Google PaLM API Key", type="password")
            if api_key:
                import google.generativeai as palm
                palm.configure(api_key=api_key)
                client = palm
            else:
                client = None
            analyze_func = analyze_druggability_palm
    
    if not api_key:
        st.warning("请配置API密钥")
        st.stop()
    
    # 搜索界面
    st.header("文献检索")
    compound_name = st.text_input("输入化合物名称/靶点/相关关键词")
    search_type = st.selectbox("搜索重点", 
                             ["drug", "mechanism", "clinical"],
                             format_func=lambda x: {
                                 "drug": "药物开发",
                                 "mechanism": "作用机制",
                                 "clinical": "临床研究"
                             }[x])
    
    if st.button("开始分析"):
        if compound_name:
            with st.spinner("正在检索相关文献..."):
                # 设置邮箱
                Entrez.email = "your_email@example.com"
                
                # 搜索文献
                pmids = search_literature(compound_name, search_type)
                
                if not pmids:
                    st.warning("未找到相关文献")
                    st.stop()
                
                # 获取摘要
                abstracts = []
                for pmid in pmids:
                    abstract = fetch_abstract(pmid)
                    if abstract:
                        abstracts.append(abstract)
                        
                if not abstracts:
                    st.warning("未能获取文献摘要")
                    st.stop()
                
                # 显示找到的文献
                st.subheader("相关文献")
                for abs in abstracts:
                    with st.expander(abs['title']):
                        st.write(f"期刊: {abs['journal']}")
                        st.write(f"年份: {abs['year']}")
                        st.write(f"摘要: {abs['abstract']}")
                
                # 使用选定的模型进行分析
                st.subheader("成药性分析")
                
                with st.spinner(f"{model_option}正在分析..."):
                    analysis = analyze_func(abstracts, client)
                    with st.expander("分析结果", expanded=True):
                        st.write(analysis)
                
                # 保存分析结果
                analysis_results = {
                    "compound": compound_name,
                    "search_type": search_type,
                    "abstracts": abstracts,
                    "analysis": {
                        "model": model_option,
                        "result": analysis
                    }
                }
                
                # 导出功能
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    "下载分析报告",
                    data=json.dumps(analysis_results, indent=2, ensure_ascii=False),
                    file_name=f"druggability_analysis_{timestamp}.json",
                    mime="application/json"
                )
                
        else:
            st.error("请输入化合物名称或相关关键词")
            
elif sidebar_option == "药物结构可视化":
    st.title("药物结构可视化")
    
    # 初始化session state
    if 'edited_mol' not in st.session_state:
        st.session_state.edited_mol = None
    
    input_source = st.radio("选择输入来源", ["SMILES输入", "PubChem搜索"])
    
    initial_smiles = None
    
    if input_source == "SMILES输入":
        initial_smiles = st.text_input("输入分子SMILES", 
            r"C1C=CC(C)=C(CC2C=C(CCC)C=C2)C=1")
    
    elif input_source == "PubChem搜索":
        compound_name = st.text_input("输入化合物名称(英文):")
        if compound_name:
            try:
                compounds = pcp.get_compounds(compound_name, 'name')
                if compounds:
                    compound = compounds[0]
                    initial_smiles = compound.canonical_smiles
                    st.success(f"已找到化合物: {compound_name}")
                else:
                    st.warning("未找到该化合物")
            except Exception as e:
                st.error(f"搜索出错: {str(e)}")
    
    if initial_smiles:
        try:
            mol = Chem.MolFromSmiles(initial_smiles)
            if mol:
                mol_block = Chem.MolToMolBlock(mol)
                
                st.subheader("结构编辑器")
                edited_mol = st_ketcher(mol_block)
                
                # 保存编辑后的结构到session state
                if edited_mol:
                    st.session_state.edited_mol = edited_mol
                
                # 使用session state中的结构进行后续处理
                if st.session_state.edited_mol:
                    try:
                        edited_rdkit_mol = Chem.MolFromMolBlock(st.session_state.edited_mol)
                        if edited_rdkit_mol:
                            new_smiles = Chem.MolToSmiles(edited_rdkit_mol)
                            
                            # 显示编辑后的SMILES
                            st.subheader("编辑后的结构")
                            st.code(new_smiles, language="plaintext")
                            
                            # 生成不同格式的分子表示
                            mol_block = Chem.MolToMolBlock(edited_rdkit_mol)
                            pdb_block = Chem.MolToPDBBlock(edited_rdkit_mol)
                            inchi = Chem.MolToInchi(edited_rdkit_mol)
                            inchikey = Chem.MolToInchiKey(edited_rdkit_mol)
                            
                            # 创建下载按钮
                            st.subheader("下载结构")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.download_button(
                                    label="下载SMILES",
                                    data=new_smiles,
                                    file_name="molecule.smi",
                                    mime="chemical/x-daylight-smiles"
                                )
                                
                                st.download_button(
                                    label="下载MOL文件",
                                    data=mol_block,
                                    file_name="molecule.mol",
                                    mime="chemical/x-mdl-molfile"
                                )
                            
                            with col2:
                                st.download_button(
                                    label="下载PDB文件",
                                    data=pdb_block,
                                    file_name="molecule.pdb",
                                    mime="chemical/x-pdb"
                                )
                                
                                st.download_button(
                                    label="下载InChI",
                                    data=inchi,
                                    file_name="molecule.inchi",
                                    mime="chemical/x-inchi"
                                )
                            
                            # 显示分子性质
                            with st.expander("查看分子性质"):
                                prop_col1, prop_col2 = st.columns(2)
                                with prop_col1:
                                    st.write("基本性质:")
                                    st.write(f"分子量: {Descriptors.ExactMolWt(edited_rdkit_mol):.2f}")
                                    st.write(f"LogP: {Descriptors.MolLogP(edited_rdkit_mol):.2f}")
                                    st.write(f"TPSA: {Descriptors.TPSA(edited_rdkit_mol):.2f}")
                                with prop_col2:
                                    st.write("结构特征:")
                                    st.write(f"原子数: {edited_rdkit_mol.GetNumAtoms()}")
                                    st.write(f"环数: {Descriptors.RingCount(edited_rdkit_mol)}")
                                    st.write(f"芳香环数: {Descriptors.NumAromaticRings(edited_rdkit_mol)}")
                            
                            # 显示其他标识符
                            with st.expander("其他分子标识符"):
                                st.write("InChI:", inchi)
                                st.write("InChIKey:", inchikey)
                                
                    except Exception as e:
                        st.error(f"处理编辑后的结构时出错: {str(e)}")
        except Exception as e:
            st.error(f"处理初始SMILES时出错: {str(e)}")

    # 添加使用说明
    st.sidebar.markdown("""
    ### 使用说明
    
    1. 选择输入来源:
       - SMILES直接输入
       - PubChem名称搜索
    
    2. 使用结构编辑器:
       - 可以直接编辑分子结构
       - 点击"Apply"保存修改
       - 支持2D结构调整
    
    3. 导出选项:
       - SMILES格式 (.smi)
       - MOL格式 (.mol)
       - PDB格式 (.pdb)
       - InChI格式 (.inchi)
    
    ### 编辑器快捷键
    - 选择: V
    - 原子: A
    - 键: B
    - 橡皮擦: E
    - 撤销: Ctrl+Z
    - 重做: Ctrl+Y
    """)