import streamlit as st
from streamlit_ketcher import st_ketcher
import pandas as pd
import os
import re
import glob
from Bio import Entrez, Medline
from anthropic import Anthropic  # 如果使用Claude
import google.generativeai as palm  # 如果使用PaLM
import pickle
import json
import pubchempy as pcp
import joblib
import random
import string
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
import networkx as nx
import time
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


# --- Helper Functions ---
# Display basic data summary
def display_data_summary(data):
    st.subheader("数据集概况")
    
    # 显示基本信息
    with st.expander("查看数据基本信息"):
        buffer = StringIO()
        data.info(buf=buffer)
        st.text(buffer.getvalue())
        st.write("描述性统计：", data.describe())

    # 获取数值型列
    numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
    
    # 列选择
    selected_columns = st.multiselect(
        "选择要查看分布的列：",
        numeric_columns,
        default=numeric_columns[0] if numeric_columns else None
    )

    # 绘制所选列的分布图
    if selected_columns:
        for col in selected_columns:
            st.write(f"{col} 的分布：")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data[col], kde=True, ax=ax)
            ax.set_title(f"{col}")
            st.pyplot(fig)
            plt.close()  # 清理内存

def create_project_directory(project_name):
    """创建项目目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_dir = os.path.join("projects", f"{project_name}_{timestamp}")
    os.makedirs(project_dir, exist_ok=True)
    return project_dir

def load_dataset(file_path):
    """加载数据集"""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        st.error(f"加载数据集出错: {str(e)}")
        return None

def process_smiles_fingerprint(data, smiles_column, label_column, project_dir):
    """处理SMILES指纹"""
    try:
        fingerprints = data[smiles_column].apply(mol_to_fp)
        fingerprint_df = pd.DataFrame(fingerprints.tolist())
        fingerprint_df['label'] = data[label_column]
        
        output_file = os.path.join(project_dir, "fingerprint_smiles.csv")
        fingerprint_df.to_csv(output_file, index=False)
        
        return output_file
    except Exception as e:
        st.error(f"生成SMILES指纹时出错: {str(e)}")
        return None

def process_formula_fingerprint(data, formula_column, label_column, project_dir):
    """处理分子式指纹"""
    try:
        fingerprints = data[formula_column].apply(formula_to_fp)
        fingerprint_df = pd.DataFrame(fingerprints.tolist())
        fingerprint_df['label'] = data[label_column]
        
        output_file = os.path.join(project_dir, "fingerprint_formula.csv")
        fingerprint_df.to_csv(output_file, index=False)
        
        return output_file
    except Exception as e:
        st.error(f"生成分子式指纹时出错: {str(e)}")
        return None

def process_selected_features(data, feature_columns, label_column, project_dir):
    """处理选定的特征列"""
    try:
        feature_df = data[feature_columns + [label_column]]
        feature_df = feature_df.rename(columns={label_column: 'label'})
        output_file = os.path.join(project_dir, "selected_features.csv")
        feature_df.to_csv(output_file, index=False)
        
        return output_file
    except Exception as e:
        st.error(f"保存特征数据时出错: {str(e)}")
        return None

def save_preprocessing_config(config, project_dir):
    """保存预处理配置"""
    try:
        config_path = os.path.join(project_dir, "preprocessing_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        return config_path
    except Exception as e:
        st.error(f"保存配置文件时出错: {str(e)}")
        return None


def load_training_data(data_path):
    """
    加载预处理后的训练数据
    
    参数:
        data_path (str): 数据文件路径
        
    返回:
        tuple: (特征矩阵X, 标签向量y)
    """
    try:
        # 读取数据
        data = pd.read_csv(data_path)
        
        # 分离特征和标签
        if 'label' in data.columns:
            y = data['label'].values
            X = data.drop('label', axis=1).values
        else:
            st.error("数据文件中未找到'label'列")
            return None, None
        
        # 检查数据
        if X.shape[0] == 0 or y.shape[0] == 0:
            st.error("数据为空")
            return None, None
            
        # 数据类型转换
        X = X.astype(np.float32)
        
        # 检查是否有无效值
        if np.isnan(X).any() or np.isinf(X).any():
            st.warning("数据中包含无效值，将进行处理")
            X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        # 显示数据信息
        st.write("数据加载完成:")
        st.write(f"特征数量: {X.shape[1]}")
        st.write(f"样本数量: {X.shape[0]}")
        
        # 检查标签分布
        unique_labels, counts = np.unique(y, return_counts=True)
        st.write("标签分布:")
        for label, count in zip(unique_labels, counts):
            st.write(f"类别 {label}: {count} 个样本")
        
        return X, y
        
    except Exception as e:
        st.error(f"加载训练数据时出错: {str(e)}")
        return None, None

def get_model_params(model_type):
    """获取模型参数设置"""
    params = {}
    if model_type == "随机森林":
        params = {
            'n_estimators': st.slider("树的数量", 50, 500, 100),
            'max_depth': st.slider("最大深度", 1, 30, 10),
            'min_samples_split': st.slider("最小分裂样本数", 2, 20, 2),
            'random_state': 42
        }
    elif model_type == "XGBoost":
        params = {
            'n_estimators': st.slider("树的数量", 50, 500, 100),
            'max_depth': st.slider("最大深度", 1, 15, 6),
            'learning_rate': st.slider("学习率", 0.01, 0.3, 0.1),
            'random_state': 42
        }
    elif model_type == "LightGBM":
        params = {
            'n_estimators': st.slider("树的数量", 50, 500, 100),
            'max_depth': st.slider("最大深度", -1, 15, -1),
            'learning_rate': st.slider("学习率", 0.01, 0.3, 0.1),
            'random_state': 42
        }
    elif model_type == "支持向量机":
        params = {
            'C': st.slider("正则化参数", 0.1, 10.0, 1.0),
            'kernel': st.selectbox("核函数", ['rbf', 'linear', 'poly']),
            'probability': True,
            'random_state': 42
        }
    elif model_type == "神经网络":
        params = {
            'hidden_layer_sizes': tuple([st.slider("隐藏层神经元数量", 10, 100, 50)]),
            'max_iter': st.slider("最大迭代次数", 100, 1000, 200),
            'learning_rate_init': st.slider("初始学习率", 0.001, 0.1, 0.001),
            'random_state': 42
        }
    return params

def create_model(model_type, params):
    """根据选择的模型类型创建模型实例"""
    if model_type == "随机森林":
        return RandomForestClassifier(**params)
    elif model_type == "XGBoost":
        return XGBClassifier(**params)
    elif model_type == "LightGBM":
        return LGBMClassifier(**params)
    elif model_type == "支持向量机":
        return SVC(**params)
    elif model_type == "神经网络":
        return MLPClassifier(**params)

# Generate fingerprint for a molecule
def mol_to_fp(smiles):
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            fp = fpgen.GetFingerprint(mol)
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr
        else:
            st.warning(f"无法解析SMILES: {smiles}")
            return [None] * 2048
    else:
        return [0] * 2048
def formula_to_smiles(formula):
    """
    使用PubChemPy将分子式转换为SMILES
    
    参数:
    formula: 分子式或化合物名称
    search_type: 搜索类型,'formula'或'name'
    
    返回:
    SMILES字符串或None
    """
    try:
        # 添加延时避免过快请求
        time.sleep(0.1)
        
        # 搜索化合物
        compounds = pcp.get_compounds(formula, search_type='formula')
        
        if compounds:
            # 返回第一个匹配结果的SMILES
            return compounds[0].canonical_smiles
        else:
            print(f"未找到化合物: {formula}")
            return None
            
    except Exception as e:
        print(f"查询出错 ({formula}): {str(e)}")
        return None
    
    
def formula_to_fp(formula):
    smiles = formula_to_smiles(formula)
    fp = mol_to_fp(smiles)
    return fp




def visualize_results(model, X_test, y_test, feature_names=None, project_dir=None):
    """
    综合可视化模型评估结果并保存到项目文件夹
    
    参数:
        model: 训练好的模型
        X_test: 测试集特征
        y_test: 测试集标签
        feature_names: 特征名称列表
        project_dir: 项目文件夹路径
    """
    # 获取预测结果
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 创建两列布局
    col1, col2 = st.columns(2)

    with col1:
        # ROC曲线
        st.write("### ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        fig_roc = plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        st.pyplot(fig_roc)
        if project_dir:
            plt.savefig(os.path.join(project_dir, 'roc_curve.png'))
        plt.close()

        # 混淆矩阵
        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        st.pyplot(fig_cm)
        if project_dir:
            plt.savefig(os.path.join(project_dir, 'confusion_matrix.png'))
        plt.close()

    with col2:
        # 评估指标
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc
        }
        
        st.write("### Performance Metrics")
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
        st.table(metrics_df.style.format({'Value': '{:.4f}'}))

        # 保存评估指标
        if project_dir:
            metrics_df.to_json(os.path.join(project_dir, 'evaluation_metrics.json'))

        # 特征重要性（如果模型支持）
        if hasattr(model, 'feature_importances_') and feature_names is not None:
            st.write("### Feature Importance")
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            fig_imp = plt.figure(figsize=(8, 6))
            plt.title('Feature Importances')
            plt.bar(range(min(20, len(indices))), 
                   importances[indices[:20]])
            plt.xticks(range(min(20, len(indices))), 
                      [feature_names[i] for i in indices[:20]], 
                      rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig_imp)
            if project_dir:
                plt.savefig(os.path.join(project_dir, 'feature_importance.png'))
            plt.close()

            # 保存特征重要性数据
            if project_dir:
                feature_importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                })
                feature_importance_df.to_csv(os.path.join(project_dir, 'feature_importance.csv'), 
                                          index=False)

    # 学习曲线
    st.write("### Learning Curve")
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_test, y_test, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10))
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    fig_lc = plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, test_mean, label='Cross-validation score')
    plt.fill_between(train_sizes, train_mean - train_std,
                    train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std,
                    test_mean + test_std, alpha=0.1)
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True)
    st.pyplot(fig_lc)
    if project_dir:
        plt.savefig(os.path.join(project_dir, 'learning_curve.png'))
    plt.close()

    # 保存预测结果
    if project_dir:
        predictions_df = pd.DataFrame({
            'True_Label': y_test,
            'Predicted_Label': y_pred,
            'Prediction_Probability': y_pred_proba
        })
        predictions_df.to_csv(os.path.join(project_dir, 'predictions.csv'), index=False)

    return metrics
def save_model_and_results(model, training_info, project_dir):
    """
    保存模型和训练结果
    
    参数:
        model: 训练好的模型
        training_info: 训练相关信息字典
        project_dir: 项目目录
    """
    try:
        # 创建时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存模型（使用pickle）
        model_path = os.path.join(project_dir, f'model_{timestamp}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # 保存模型元信息
        model_info = {
            'model_type': training_info['model_type'],
            'model_params': training_info['model_params'],
            'training_timestamp': timestamp,
            'feature_type': training_info['feature_type'],
            'test_size': training_info['test_size'],
            'random_state': training_info['random_state'],
            'feature_columns': training_info['feature_columns'] 
        }
        
        # 保存训练配置和结果
        config_path = os.path.join(project_dir, f'training_config_{timestamp}.json')
        with open(config_path, 'w') as f:
            json.dump(model_info, f, indent=4)
            
        # 创建模型说明文件
        readme_path = os.path.join(project_dir, f'README_{timestamp}.md')
        with open(readme_path, 'w') as f:
            f.write(f"# Model Training Summary\n\n")
            f.write(f"## Training Information\n")
            f.write(f"- Model Type: {model_info['model_type']}\n")
            f.write(f"- Training Time: {timestamp}\n")
            f.write(f"- Feature Type: {model_info['feature_type']}\n\n")
            f.write(f"## Model Parameters\n")
            for param, value in model_info['model_params'].items():
                f.write(f"- {param}: {value}\n")
            
        st.success(f"模型和相关信息已保存到：{project_dir}")
        st.write("保存的文件：")
        st.write(f"- 模型文件：model_{timestamp}.pkl")
        st.write(f"- 配置文件：training_config_{timestamp}.json")
        st.write(f"- 说明文件：README_{timestamp}.md")
        
        return True
        
    except Exception as e:
        st.error(f"保存模型和结果时出错: {str(e)}")
        return False

def load_saved_model(model_path):
    """
    加载保存的模型
    
    参数:
        model_path: 模型文件路径
    返回:
        加载的模型
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"加载模型时出错: {str(e)}")
        return None

def search_literature(keyword, search_type="drug"):
    """增强版文献搜索"""
    search_terms = {
        "drug": f'("{keyword}"[Title/Abstract]) AND ("drug development"[Title/Abstract] OR "drug discovery"[Title/Abstract] OR "clinical trial"[Title/Abstract])',
        "mechanism": f'("{keyword}"[Title/Abstract]) AND ("mechanism of action"[Title/Abstract] OR "pathway"[Title/Abstract])',
        "clinical": f'("{keyword}"[Title/Abstract]) AND ("clinical"[Title/Abstract] OR "phase"[Title/Abstract])'
    }
    
    handle = Entrez.esearch(db="pubmed", term=search_terms[search_type], retmax=10)
    record = Entrez.read(handle)
    return record["IdList"]

def fetch_abstract(pmid):
    """获取文章摘要"""
    try:
        handle = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
        record = Medline.read(handle)
        return {
            'title': record.get('TI', ''),
            'abstract': record.get('AB', ''),
            'journal': record.get('JT', ''),
            'year': record.get('YEAR', '')
        }
    except:
        return None

def analyze_druggability_gpt(abstracts, client, model="gpt-4"):
    """使用GPT分析成药性"""
    context = "\n\n".join([f"标题：{a['title']}\n摘要：{a['abstract']}" for a in abstracts])
    
    query = f"""作为一个药物开发专家，请基于以下文献分析该化合物的成药性。

文献内容：
{context}

请从以下方面进行分析：
1. 目前的开发阶段
2. 作用机制的明确程度
3. 已知的药效学特征
4. 已知的安全性问题
5. 临床应用潜力
6. 可能存在的开发风险

请以结构化的方式呈现分析结果。如果信息不足，请明确指出。"""

    response = client.responses.create(
        model=model,
        input=query
    )
    return response.output_text

def analyze_druggability_claude(abstracts, client):
    """使用Claude分析成药性"""
    context = "\n\n".join([f"标题：{a['title']}\n摘要：{a['abstract']}" for a in abstracts])
    
    prompt = f"""As a pharmaceutical expert, please analyze the druggability of the compound based on the following literature.

Literature:
{context}

Please analyze from these aspects:
1. Current development stage
2. Clarity of mechanism of action
3. Known pharmacodynamic characteristics
4. Known safety issues
5. Clinical application potential
6. Potential development risks

Please provide structured analysis. If information is insufficient, please explicitly state so."""

    response = client.messages.create(
        model="claude-2",
        content=prompt
    )
    return response.content

def analyze_druggability_palm(abstracts, palm):
    """使用PaLM分析成药性"""
    context = "\n\n".join([f"标题：{a['title']}\n摘要：{a['abstract']}" for a in abstracts])
    
    prompt = f"""As a pharmaceutical expert, analyze the druggability of the compound from the following literature.

Literature:
{context}

Analyze:
1. Development stage
2. Mechanism clarity
3. Pharmacodynamics
4. Safety profile
5. Clinical potential
6. Development risks

Provide structured analysis. Note if information is insufficient."""

    response = palm.generate_text(
        prompt=prompt,
        temperature=0.3
    )
    return response.result

def analyze_druggability_spark(abstracts, api_key):
    """使用讯飞星火分析成药性"""
    import requests
    import json
    
    url = "https://spark-api-open.xf-yun.com/v1/chat/completions"
    
    # 构建消息内容
    context = "\n\n".join([f"标题：{a['title']}\n摘要：{a['abstract']}" for a in abstracts])
    
    data = {
        "max_tokens": 4096,
        "top_k": 4,
        "temperature": 0.5,
        "messages": [
            {
                "role": "system",
                "content": "你是一个专业的药物开发专家,请基于提供的文献进行专业分析。"
            },
            {
                "role": "user",
                "content": f"""请基于以下文献分析该化合物的成药性。

文献内容：
{context}

请从以下方面进行分析：
1. 目前的开发阶段
2. 作用机制的明确程度
3. 已知的药效学特征
4. 已知的安全性问题
5. 临床应用潜力
6. 可能存在的开发风险

请以结构化的方式呈现分析结果。如果信息不足，请明确指出。"""
            }
        ],
        "model": "4.0Ultra",
        "stream": True
    }
    
    header = {
        "Authorization": api_key,
        "Content-Type": "application/json"
    }

    try:
        # 创建进度显示
        progress_placeholder = st.empty()
        response_placeholder = st.empty()
        full_response = ""
        
        # 发送请求并处理流式响应
        response = requests.post(url, headers=header, json=data, stream=True)
        response.encoding = "utf-8"
        
        for line in response.iter_lines(decode_unicode="utf-8"):
            if line:
                # 跳过心跳信息
                if line == "data: [DONE]" or not line.startswith("data: "):
                    continue
                    
                try:
                    # 解析返回数据
                    json_data = json.loads(line[6:])  # 去掉 "data: " 前缀
                    if 'choices' in json_data and len(json_data['choices']) > 0:
                        delta = json_data['choices'][0].get('delta', {})
                        if 'content' in delta:
                            content = delta['content']
                            full_response += content
                            # 实时更新显示
                            response_placeholder.markdown(full_response)
                            
                except json.JSONDecodeError:
                    continue
                
        return full_response
        
    except requests.exceptions.RequestException as e:
        error_msg = f"API请求错误: {str(e)}"
        st.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"分析过程出错: {str(e)}"
        st.error(error_msg)
        return error_msg
    finally:
        # 清理进度显示
        if 'progress_placeholder' in locals():
            progress_placeholder.empty()

