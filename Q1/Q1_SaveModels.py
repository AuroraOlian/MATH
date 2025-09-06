import pandas as pd
import numpy as np
import xgboost as xgb
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class MaternalDataPreprocessor:
    """母体数据预处理类"""

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def convert_gestational_week(self, week_str):
        """将 'Xw+Y' 格式的孕周字符串转换为数值（以周为单位）"""
        if pd.isna(week_str):
            return np.nan
        if isinstance(week_str, (int, float)):
            return float(week_str)
        if isinstance(week_str, str):
            # 匹配 "11w+6" 或 "11w" 或 "11" 格式
            pattern = r'(\d+)w?(?:\+(\d+))?'
            match = re.match(pattern, week_str.strip())
            if match:
                weeks = int(match.group(1))
                days = int(match.group(2)) if match.group(2) else 0
                return weeks + days / 7.0
        return np.nan

    def clean_categorical_data(self, series, column_name):
        """清理分类数据"""
        if series.dtype == 'object':
            # 常见的分类映射
            mapping_dict = {
                '胎儿是否健康': {'是': 1, '健康': 1, '正常': 1, '否': 0, '异常': 0, '不健康': 0},
                'IVF妊娠': {'是': 1, 'IVF妊娠': 1, 'IVF': 1, '否': 0, '自然受孕': 0, '自然': 0},
                '染色体的非整倍体': {'正常': 0, '异常': 1, '是': 1, '否': 0}
            }

            if column_name in mapping_dict:
                return series.map(mapping_dict[column_name])
            else:
                # 对于其他分类变量，使用LabelEncoder
                if column_name not in self.label_encoders:
                    self.label_encoders[column_name] = LabelEncoder()
                    # 处理缺失值
                    non_null_values = series.dropna()
                    if len(non_null_values) > 0:
                        self.label_encoders[column_name].fit(non_null_values)
                        encoded = series.copy()
                        mask = series.notna()
                        encoded[mask] = self.label_encoders[column_name].transform(series[mask])
                        return pd.to_numeric(encoded, errors='coerce')
                return pd.to_numeric(series, errors='coerce')
        return pd.to_numeric(series, errors='coerce')

    def preprocess_data(self, df):
        """主要的数据预处理函数"""
        df_processed = df.copy()

        # 1. 处理检测孕周
        print("处理检测孕周...")
        df_processed['检测孕周_数值'] = df_processed['检测孕周'].apply(self.convert_gestational_week)

        # 2. 处理所有列的数据类型
        print("清理和转换数据类型...")
        for col in df_processed.columns:
            if col not in ['序号', '孕妇代码', '末次月经', '检测日期']:  # 跳过ID和日期列
                print(f"  处理列: {col}")
                print(f"    原始数据类型: {df_processed[col].dtype}")
                print(f"    唯一值示例: {df_processed[col].unique()[:5]}")

                try:
                    df_processed[col] = self.clean_categorical_data(df_processed[col], col)
                    print(f"    转换后数据类型: {df_processed[col].dtype}")
                except Exception as e:
                    print(f"    警告：处理列 '{col}' 时出错: {e}")
                    # 如果出错，直接尝试数值转换
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

        return df_processed

def explore_data(df, target_variable):
    """数据探索性分析（不包含绘图）"""
    print("=== 数据探索性分析 ===")
    print(f"数据集形状: {df.shape}")
    print(f"\n缺失值统计:")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        '缺失数量': missing_data,
        '缺失百分比': missing_percent
    }).sort_values('缺失百分比', ascending=False)
    print(missing_df[missing_df['缺失数量'] > 0])

    # 目标变量统计
    print(f"\n目标变量 '{target_variable}' 统计:")
    print(df[target_variable].describe())
    
    # 返回用于后续绘图的数据
    return {
        'missing_df': missing_df,
        'target_stats': df[target_variable].describe(),
        'target_data': df[target_variable].dropna()
    }

def train_and_evaluate_model(X_train, X_val, X_test, y_train, y_val, y_test, use_grid_search=False):
    """训练和评估模型 - 修正版本"""
    
    print(f"数据划分: 训练集{X_train.shape}, 验证集{X_val.shape}, 测试集{X_test.shape}")
    
    if use_grid_search:
        print("正在进行超参数调优...")
        param_grid = {
            'n_estimators': [500, 1000, 1500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }

        xgb_regressor = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
            eval_metric='rmse'
        )

        # 使用训练集+验证集进行网格搜索
        X_train_val = pd.concat([X_train, X_val])
        y_train_val = pd.concat([y_train, y_val])
        
        grid_search = GridSearchCV(
            xgb_regressor,
            param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=2
        )

        import time
        start_time = time.time()
        print("开始网格搜索...")
        grid_search.fit(X_train_val, y_train_val)
        end_time = time.time()
        
        best_model = grid_search.best_estimator_
        print(f"网格搜索耗时: {end_time - start_time:.2f}秒")
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳CV得分: {grid_search.best_score_:.4f}")
        
    else:
        print("使用固定参数训练...")
        best_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=2000,  
            learning_rate=0.03,  
            max_depth=6,         
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='rmse'
        )

        import time
        start_time = time.time()
        print("开始训练...")
        
        try:
            from xgboost.callback import EarlyStopping
            
            best_model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],  # 使用验证集进行早停
                eval_names=['train', 'validation'],
                callbacks=[EarlyStopping(rounds=200, save_best=True)],
                verbose=50
            )
            
            end_time = time.time()
            print(f"训练完成！耗时: {end_time - start_time:.2f}秒")
            print(f"实际训练轮数: {best_model.best_iteration}")
            
            # 验证集性能（训练过程中的监控）
            val_pred = best_model.predict(X_val)
            val_r2 = r2_score(y_val, val_pred)
            print(f"验证集R² (训练过程监控): {val_r2:.4f}")
            
        except (ImportError, TypeError):
            print("使用兼容模式，训练全部轮数...")
            best_model.fit(X_train, y_train)
            end_time = time.time()
            print(f"训练完成！耗时: {end_time - start_time:.2f}秒")
            print(f"训练了全部 {best_model.n_estimators} 轮")

    return best_model

def evaluate_model_performance(model, X_train, X_val, X_test, y_train, y_val, y_test):
    """详细的模型性能评估"""
    print("\n=== 详细模型性能评估 ===")
    
    # 分别预测三个数据集
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # 计算各数据集的评估指标
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    
    val_r2 = r2_score(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # 打印结果
    print("训练集性能:")
    print(f"  R²: {train_r2:.4f}")
    print(f"  MAE: {train_mae:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")
    
    print("\n验证集性能:")
    print(f"  R²: {val_r2:.4f}")
    print(f"  MAE: {val_mae:.4f}")
    print(f"  RMSE: {val_rmse:.4f}")
    
    print("\n测试集性能 (最终评估):")
    print(f"  R²: {test_r2:.4f}")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    
    # 过拟合检查
    train_val_diff = train_r2 - val_r2
    val_test_diff = val_r2 - test_r2
    
    print(f"\n性能差异分析:")
    print(f"  训练集 - 验证集 R²差异: {train_val_diff:.4f}")
    print(f"  验证集 - 测试集 R²差异: {val_test_diff:.4f}")
    
    if train_val_diff > 0.1:
        print("  ⚠️  训练集和验证集差异较大，可能存在过拟合")
    elif abs(val_test_diff) > 0.05:
        print("  ⚠️  验证集和测试集差异较大，模型泛化性可能不稳定")
    else:
        print("  ✅ 模型在各数据集上表现一致，泛化性良好")
    
    return {
        'train': {'r2': train_r2, 'mae': train_mae, 'rmse': train_rmse, 'predictions': y_train_pred},
        'validation': {'r2': val_r2, 'mae': val_mae, 'rmse': val_rmse, 'predictions': y_val_pred},
        'test': {'r2': test_r2, 'mae': test_mae, 'rmse': test_rmse, 'predictions': y_test_pred},
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'residuals_test': y_test - y_test_pred
    }

def plot_target_distribution(target_data, target_variable):
    """绘制目标变量分布"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(target_data, bins=30, alpha=0.7, color='skyblue')
    plt.title(f'{target_variable} 分布')
    plt.xlabel(target_variable)
    plt.ylabel('频次')

    plt.subplot(1, 2, 2)
    plt.boxplot(target_data)
    plt.title(f'{target_variable} 箱线图')
    plt.ylabel(target_variable)

    plt.tight_layout()
    plt.show()

def plot_model_performance(results):
    """绘制模型性能图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    datasets = ['train', 'validation', 'test']
    colors = ['blue', 'orange', 'green']
    
    # 第一行：实际值 vs 预测值
    for i, (dataset, color) in enumerate(zip(datasets, colors)):
        ax = axes[0, i]
        y_true = results[f'y_{dataset}'] if dataset != 'validation' else results['y_val']
        y_pred = results[dataset]['predictions']
        r2 = results[dataset]['r2']
        
        ax.scatter(y_true, y_pred, alpha=0.6, color=color)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax.set_xlabel('实际值')
        ax.set_ylabel('预测值')
        ax.set_title(f'{dataset.title()} Set: 实际值 vs 预测值 (R² = {r2:.3f})')
    
    # 第二行：残差分析
    for i, (dataset, color) in enumerate(zip(datasets, colors)):
        ax = axes[1, i]
        y_true = results[f'y_{dataset}'] if dataset != 'validation' else results['y_val']
        y_pred = results[dataset]['predictions']
        residuals = y_true - y_pred
        
        ax.scatter(y_pred, residuals, alpha=0.6, color=color)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('预测值')
        ax.set_ylabel('残差')
        ax.set_title(f'{dataset.title()} Set: 残差图')
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names, top_n=15):
    """绘制特征重要性"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(top_n)

    bars = plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('特征重要性')
    plt.title(f'Top {top_n} 特征重要性')
    plt.gca().invert_yaxis()

    # 添加数值标签
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{importance:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.show()

    return importance_df

def save_model_and_results(model, results, feature_importance, exploration_data, preprocessor, feature_names, save_dir="model_output"):
    """保存模型和训练结果"""
    import os
    import pickle
    import joblib
    import json
    from datetime import datetime

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_dir = f"{save_dir}_{timestamp}"
    
    print(f"\n=== 保存模型和结果到 '{save_dir}' 目录 ===")
    
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建目录: {save_dir}")
    
    
    
    try:
        # 1. 保存训练好的模型
        model_path = os.path.join(save_dir, f"xgb_model_{timestamp}.pkl")
        joblib.dump(model, model_path)
        print(f"✅ 模型已保存: {model_path}")
        
        # 2. 保存数据预处理器
        preprocessor_path = os.path.join(save_dir, f"preprocessor_{timestamp}.pkl")
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(preprocessor, f)
        print(f"✅ 预处理器已保存: {preprocessor_path}")
        
        # 3. 保存特征名称
        feature_names_path = os.path.join(save_dir, f"feature_names_{timestamp}.pkl")
        with open(feature_names_path, 'wb') as f:
            pickle.dump(list(feature_names), f)
        print(f"✅ 特征名称已保存: {feature_names_path}")
        
        # 4. 保存模型评估结果
        results_to_save = {}
        for dataset in ['train', 'validation', 'test']:
            results_to_save[dataset] = {
                'r2': float(results[dataset]['r2']),
                'mae': float(results[dataset]['mae']),
                'rmse': float(results[dataset]['rmse'])
            }
        
        results_path = os.path.join(save_dir, f"model_results_{timestamp}.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, ensure_ascii=False, indent=2)
        print(f"✅ 模型评估结果已保存: {results_path}")
        
        # 5. 保存特征重要性
        feature_importance_path = os.path.join(save_dir, f"feature_importance_{timestamp}.csv")
        feature_importance.to_csv(feature_importance_path, index=False, encoding='utf-8-sig')
        print(f"✅ 特征重要性已保存: {feature_importance_path}")
        
        # 6. 保存预测结果 - 分别保存每个数据集的结果
        try:
            # 分别保存训练集预测结果
            train_predictions_df = pd.DataFrame({
                'actual': results['y_train'].values,
                'predicted': results['train']['predictions']
            })
            train_pred_path = os.path.join(save_dir, f"train_predictions_{timestamp}.csv")
            train_predictions_df.to_csv(train_pred_path, index=False, encoding='utf-8-sig')
            
            # 分别保存验证集预测结果
            val_predictions_df = pd.DataFrame({
                'actual': results['y_val'].values,
                'predicted': results['validation']['predictions']
            })
            val_pred_path = os.path.join(save_dir, f"val_predictions_{timestamp}.csv")
            val_predictions_df.to_csv(val_pred_path, index=False, encoding='utf-8-sig')
            
            # 分别保存测试集预测结果
            test_predictions_df = pd.DataFrame({
                'actual': results['y_test'].values,
                'predicted': results['test']['predictions']
            })
            test_pred_path = os.path.join(save_dir, f"test_predictions_{timestamp}.csv")
            test_predictions_df.to_csv(test_pred_path, index=False, encoding='utf-8-sig')
            
            print(f"✅ 训练集预测结果已保存: {train_pred_path}")
            print(f"✅ 验证集预测结果已保存: {val_pred_path}")
            print(f"✅ 测试集预测结果已保存: {test_pred_path}")
            
        except Exception as pred_error:
            print(f"❌ 保存预测结果时出错: {pred_error}")
            # 如果上面的方法仍有问题，使用更安全的方法
            print("尝试使用备用保存方法...")
            
            # 备用方法：逐个检查并保存
            for dataset_name, y_actual, y_pred in [
                ('train', results['y_train'], results['train']['predictions']),
                ('val', results['y_val'], results['validation']['predictions']),
                ('test', results['y_test'], results['test']['predictions'])
            ]:
                try:
                    # 确保长度一致
                    min_len = min(len(y_actual), len(y_pred))
                    df = pd.DataFrame({
                        'actual': y_actual.values[:min_len] if hasattr(y_actual, 'values') else y_actual[:min_len],
                        'predicted': y_pred[:min_len]
                    })
                    path = os.path.join(save_dir, f"{dataset_name}_predictions_{timestamp}.csv")
                    df.to_csv(path, index=False, encoding='utf-8-sig')
                    print(f"✅ {dataset_name}集预测结果已保存: {path}")
                except Exception as e:
                    print(f"❌ 保存{dataset_name}集预测结果失败: {e}")
        
        # 7. 保存模型配置和训练信息
        model_info = {
            'timestamp': timestamp,
            'model_type': 'XGBoost Regressor',
            'model_params': model.get_params(),
            'feature_count': len(feature_names),
            'train_samples': len(results['y_train']),
            'validation_samples': len(results['y_val']),
            'test_samples': len(results['y_test']),
            'final_test_r2': float(results['test']['r2']),
            'final_test_rmse': float(results['test']['rmse']),
            'best_features_top5': list(feature_importance.head(5)['feature'].values)
        }
        
        model_info_path = os.path.join(save_dir, f"model_info_{timestamp}.json")
        with open(model_info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        print(f"✅ 模型信息已保存: {model_info_path}")
        
        # 8. 创建加载模型的示例代码
        load_example = f"""
# 加载已保存的模型和相关文件示例代码
import joblib
import pickle
import pandas as pd
import json

# 加载模型
model = joblib.load('{model_path}')

# 加载预处理器
with open('{preprocessor_path}', 'rb') as f:
    preprocessor = pickle.load(f)

# 加载特征名称
with open('{feature_names_path}', 'rb') as f:
    feature_names = pickle.load(f)

# 加载模型结果
with open('{results_path}', 'r', encoding='utf-8') as f:
    results = json.load(f)

# 加载特征重要性
feature_importance = pd.read_csv('{feature_importance_path}', encoding='utf-8-sig')

# 加载预测结果
train_predictions = pd.read_csv('{os.path.join(save_dir, f"train_predictions_{timestamp}.csv")}', encoding='utf-8-sig')
val_predictions = pd.read_csv('{os.path.join(save_dir, f"val_predictions_{timestamp}.csv")}', encoding='utf-8-sig')
test_predictions = pd.read_csv('{os.path.join(save_dir, f"test_predictions_{timestamp}.csv")}', encoding='utf-8-sig')

# 使用模型进行预测 (示例)
# new_data = preprocessor.preprocess_data(your_new_dataframe)
# predictions = model.predict(new_data[feature_names])

print("模型加载完成！")
print(f"测试集R²: {{results['test']['r2']}}")
print(f"最重要的5个特征: {{feature_importance.head(5)['feature'].tolist()}}")
"""
        
        load_example_path = os.path.join(save_dir, f"load_model_example_{timestamp}.py")
        with open(load_example_path, 'w', encoding='utf-8') as f:
            f.write(load_example)
        print(f"✅ 模型加载示例代码已保存: {load_example_path}")
        
        print(f"\n🎉 所有文件已成功保存到 '{save_dir}' 目录！")
        print(f"📁 共保存了 10 个文件，时间戳: {timestamp}")
        
        return {
            'save_directory': save_dir,
            'timestamp': timestamp,
            'saved_files': {
                'model': model_path,
                'preprocessor': preprocessor_path,
                'feature_names': feature_names_path,
                'results': results_path,
                'feature_importance': feature_importance_path,
                'train_predictions': train_pred_path,
                'val_predictions': val_pred_path,
                'test_predictions': test_pred_path,
                'model_info': model_info_path,
                'load_example': load_example_path
            }
        }
        
    except Exception as e:
        print(f"❌ 保存过程中出现错误: {e}")
        return None
    
def main(enable_plotting=False, use_grid_search=False):
    """主函数 - 修正版本"""
    print("=== Y染色体浓度预测模型 (修正版) ===")

    # 1. 数据加载
    print("\n1. 加载数据...")
    try:
        df = pd.read_excel("maternal_data.xlsx")
        print(f"成功加载数据，形状: {df.shape}")
    except FileNotFoundError:
        print("错误：请确保 'maternal_data.xlsx' 文件与此脚本在同一个目录下。")
        return
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return

    # 2. 数据预处理
    print("\n2. 数据预处理...")
    preprocessor = MaternalDataPreprocessor()
    df_processed = preprocessor.preprocess_data(df)

    # 3. 数据探索（不绘图）
    TARGET_VARIABLE = 'Y染色体浓度'
    exploration_data = explore_data(df_processed, TARGET_VARIABLE)

    # 4. 特征选择
    print("\n3. 特征选择...")
    COLUMNS_TO_DROP = [
        '序号', '孕妇代码', '末次月经', '检测日期',
        '检测孕周',  # 已转换为数值版本
    ]

    # 移除不存在的列
    columns_to_drop_filtered = [col for col in COLUMNS_TO_DROP if col in df_processed.columns]

    # 筛选特征
    available_columns = [col for col in df_processed.columns if col not in columns_to_drop_filtered + [TARGET_VARIABLE]]
    features = df_processed[available_columns].select_dtypes(include=[np.number])
    target = df_processed[TARGET_VARIABLE]

    print(f"可用特征: {list(features.columns)}")
    print(f"特征数量: {features.shape[1]}")

    # 5. 处理缺失值
    print("\n4. 处理缺失值...")
    print("特征缺失值情况:")
    missing_features = features.isnull().sum()[features.isnull().sum() > 0]
    if len(missing_features) > 0:
        print(missing_features)
    else:
        print("无缺失值")

    # 对数值特征使用中位数填充
    features = features.fillna(features.median())
    target = target.fillna(target.median())

    # 6. 数据划分 - 修正：划分为训练集/验证集/测试集
    print("\n5. 划分数据集...")
    # 首先划分出测试集 (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    # 然后从剩余数据中划分训练集和验证集 (64% 训练, 16% 验证)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42
    )

    print(f"训练集样本数: {len(X_train)} ({len(X_train)/len(features)*100:.1f}%)")
    print(f"验证集样本数: {len(X_val)} ({len(X_val)/len(features)*100:.1f}%)")
    print(f"测试集样本数: {len(X_test)} ({len(X_test)/len(features)*100:.1f}%)")

    # 7. 模型训练
    print("\n6. 训练模型...")
    model = train_and_evaluate_model(X_train, X_val, X_test, y_train, y_val, y_test, use_grid_search=use_grid_search)

    # 8. 详细的模型评估
    print("\n7. 模型评估...")
    model_results = evaluate_model_performance(model, X_train, X_val, X_test, y_train, y_val, y_test)

    # 9. 特征重要性分析
    print("\n8. 特征重要性分析...")
    importance_df = pd.DataFrame({
        'feature': features.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n特征重要性排名:")
    print(importance_df.head(10))

    # 10. 结果总结
    print("\n=== 最终结果总结 ===")
    print(f"最终测试集R²: {model_results['test']['r2']:.4f}")
    print(f"最终测试集RMSE: {model_results['test']['rmse']:.4f}")
    print(f"最重要的5个特征: {list(importance_df.head(5)['feature'])}")

    # 11. 绘图部分（可选）
    if enable_plotting:
        print("\n9. 生成图表...")
        
        # 绘制目标变量分布
        print("绘制目标变量分布...")
        plot_target_distribution(exploration_data['target_data'], TARGET_VARIABLE)
        
        # 绘制模型性能
        print("绘制模型性能...")
        plot_model_performance(model_results)
        
        # 绘制特征重要性
        print("绘制特征重要性...")
        plot_feature_importance(model, features.columns)
        
        print("所有图表已生成完成！")
    else:
        print("\n注意：绘图功能已禁用。如需查看图表，请设置 enable_plotting=True")

    # 12. 保存模型和结果（新增部分）
    print("\n10. 保存模型和训练数据...")
    save_info = save_model_and_results(
        model=model,
        results=model_results,
        feature_importance=importance_df,
        exploration_data=exploration_data,
        preprocessor=preprocessor,
        feature_names=features.columns
    )
    
    if save_info:
        print(f"✅ 模型和数据已成功保存！")
        print(f"📂 保存位置: {save_info['save_directory']}")
        print(f"🕒 时间戳: {save_info['timestamp']}")
    else:
        print("❌ 保存失败，请检查错误信息")

    return {
        'model': model,
        'results': model_results,
        'feature_importance': importance_df,
        'exploration_data': exploration_data,
        'save_info': save_info  # 新增保存信息
    }

if __name__ == "__main__":
    # 运行选项
    results = main(
        enable_plotting=True,      # 设置为True启用绘图
        use_grid_search=True       # 设置为True启用超参数调优
    )