import os
import time
from usefulTools import  *
from dataApi.dividend import *
from dataApi.getData import *
from BarraFactor.barra_factor import *
from dataApi.indName import *
from dataApi.stockList import *
from dataApi.tradeDate import *
import pandas as pd
import numpy as np
from sklearn import linear_model,model_selection
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor,AdaBoostRegressor,GradientBoostingRegressor,VotingRegressor
from SectorRotation.FactorTest import *
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import lightgbm as lgb
import xgboost as xgb

# 注1：因子只处理了去极值和标准化，并没有进行行业中性化处理：测试下来发现两者差距不大，可以先不进行中性化处理
# 注2：由于时间问题，策略和实盘存在一定差异：在因子的筛选期，使用的是月初开盘——下个月月初的开盘，这个是策略可行性是可行的；
#     但是策略实际测试，使用的是月末收盘——下个月末收盘；主要是由于策略端不能自动化的问题，而导致会出现在人工调仓的时候，会出现隔夜收益的损失。
# 可以考虑用rank加权，而不是因子原始值加权

def factor_model(df_x, df_y, pre_x,model_type='linear'):
    if model_type == 'linear':
        model = linear_model.LinearRegression(fit_intercept=False)
        model.fit(df_x.fillna(0), df_y)
    elif model_type == 'Ridge':
        model = linear_model.RidgeCV(alphas=(10,1,0.1,0.01,0.005,0.001),cv = 5, fit_intercept=False)
        model.fit(df_x.fillna(0), df_y)
    elif model_type == 'Lars':
        model = linear_model.LassoLarsCV(cv = 5, fit_intercept=False,max_iter = 500)
        model.fit(df_x.fillna(0), df_y)
    elif model_type == 'Bayes':
        model = linear_model.BayesianRidge(alpha_1=0.1, lambda_1=0.1,fit_intercept=False, tol = 1e-5)
        model.fit(df_x.fillna(0), df_y)
        '''
        param_grid = {'alpha_1':[10,1,0.1,0.005,0.001],'alpha_2':[10,1,0.1,0.005,0.001],
                      'lambda_1':[10,1,0.1,0.005,0.001],'lambda_2':[10,1,0.1,0.005,0.001]}
        search = model_selection.GridSearchCV(linear_model.BayesianRidge(fit_intercept=False, tol = 1e-5),
                                     param_grid,
                                     scoring='r2', n_jobs=1,
                                     refit = True, cv =3,
                                     error_score =np.nan, return_train_score ='warn')
        search.fit(df_x.fillna(0), df_y)
        model = search.best_estimator_
        '''

    elif model_type == 'logistic':
        df_y1 = df_y.copy()
        df_y1.index.names = ['date','ind']
        df_y1[(df_y1.groupby('date').rank(pct=True) >= 0.8)] = 1
        #df_y[df_y.groupby('date').rank(pct=True) <= 0.2] = 1
        df_y1[df_y1 < 1] = 0
        #model = linear_model.LogisticRegression(C = 1,fit_intercept = False,class_weight={0:0.8,2:0.2},penalty='l1',solver='liblinear')
        #model = linear_model.LogisticRegressionCV(Cs=[10,1,0.1,0.05,0.01,0.005],cv = 5, fit_intercept=False, class_weight={0:0.8,1:0.2})
        model = linear_model.LogisticRegression(C = 1,fit_intercept=False, class_weight={0:0.8,1:0.2})
        model.fit(df_x.fillna(0), df_y1.astype(int))
        #model.predict_proba(pre_x)
        return pd.Series(model.predict_proba(pre_x)[:,1],index=pre_x.index)

    elif model_type == 'RandomForest':

        model = RandomForestRegressor(n_estimators=100,
                                      # 数值型参数，默认值为100，此参数指定了弱分类器的个数。设置的值越大，精确度越好，但是当 n_estimators 大于特定值之后，带来的提升效果非常有限。
                                      criterion='mse',  # 其中，参数criterion 是字符串类型，默认值为 ‘mse’，是衡量回归效果的指标。可选的还有‘mae’ 。
                                      max_features='sqrt', # if “auto”, “sqrt”,“log2”, None, max_features=n_features.
                                      max_depth=5,
                                      # 数值型，默认值None。这是与剪枝相关的参数，设置为None时，树的节点会一直分裂，直到：（1）每个叶子都是“纯”的；（2）或者叶子中包含于min_sanples_split个样本。推荐从 max_depth = 3 尝试增加，观察是否应该继续加大深度。
                                      #min_samples_split=2,
                                      # 数值型，默认值2，指定每个内部节点(非叶子节点)包含的最少的样本数。与min_samples_leaf这个参数类似，可以是整数也可以是浮点数。
                                      #min_samples_leaf=1,
                                      # 数值型，默认值1，指定每个叶子结点包含的最少的样本数。参数的取值除了整数之外，还可以是浮点数，此时（min_samples_leaf * n_samples）向下取整后的整数是每个节点的最小样本数。此参数设置的过小会导致过拟合，反之就会欠拟合。
                                      #min_weight_fraction_leaf=0.0,  # (default=0) 叶子节点所需要的最小权值
                                      # 可以为整数、浮点、字符或者None，默认值为None。此参数用于限制分枝时考虑的特征个数，超过限制个数的特征都会被舍弃。
                                      #max_leaf_nodes=None,
                                      # 数值型参数，默认值为None，即不限制最大叶子节点数。这个参数通过限制树的最大叶子数量来防止过拟合，如果设置了一个正整数，则会在建立的最大叶节点内的树中选择最优的决策树。
                                      bootstrap=True,  # 是否有放回的采样。
                                      oob_score=True,  # oob（out of band，带外）数据，即：在某次决策树训练中没有被bootstrap选中的数据
                                      random_state=0,  # # 随机种子
                                      )
        model.fit(df_x.fillna(0), df_y)
        '''
        param_grid = {'n_estimators': [10,50,100,200], 'max_depth': [5,10,15,20],
                      'max_features': ['auto','sqrt','log2'],
                      'min_samples_split':[2,3,4,5,6,8,10],'criterion':['squared_error','absolute_error']}
        search = model_selection.RandomizedSearchCV(RandomForestRegressor(bootstrap=True,random_state=0),
                                              param_grid,
                                              scoring='r2', n_jobs=10,
                                              refit=True,cv=3,
                                              error_score=np.nan, return_train_score=False)
        search.fit(df_x.fillna(0), df_y)
        model = search.best_estimator_
        '''
    elif model_type == 'GBRT':

        model = GradientBoostingRegressor(loss = 'huber',  #ls,quantile,huber,
                                          n_estimators = 100,
                                          learning_rate = 0.01,
                                          max_depth = 5,
                                          subsample = 0.8,
                                          max_features = 'auto', # if “auto”, “sqrt”,“log2”, None, max_features=n_features.
                                          criterion = 'friedman_mse',# "mse"、"friedman_mse" 和 "mae"
                                          random_state=0)
        model.fit(df_x.fillna(0), df_y)
        '''
        param_grid = {'n_estimators': [10, 50, 100, 200], 'max_depth': [5, 10, 15, 20],
                      'max_features': [None,'auto', 'sqrt', 'log2'],
                      'subsample':[0.7,0.8,0.9,1],'learning_rate':[0.5,0.1,0.05,0.01,0.005,0.001],
                      'loss': ['ls','huber','quantile'], 'criterion': ['squared_error', 'friedman_mse','absolute_error']}
        search = model_selection.RandomizedSearchCV(GradientBoostingRegressor(random_state=0),
                                              param_grid,
                                              scoring='explained_variance', n_jobs=10,
                                              refit=True,cv=3,
                                              error_score=np.nan, return_train_score=False)
        search.fit(df_x.fillna(0), df_y)
        model = search.best_estimator_
        '''
    elif model_type == 'LGB':
        # boosting_type指定弱学习器的类型，默认值为 ‘gbdt’
        # ‘gbdt’，使用梯度提升树     ‘rf’，使用随机森林    ‘dart’，
        # ‘goss’，使用单边梯度抽样算法，速度很快，但是可能欠拟合。
        model = lgb.LGBMRegressor(boosting_type='gbdt',  # gbdt,df,dart
                                  #objective='regression_l1',  # objective：指定目标可选参数如下：“regression”，
                                  # 使用L2正则项的回归模型（默认值）。“regression_l1”，使用L1正则项的回归模型。“mae”，平均绝对百分比误差。“binary”，二分类。“multiclass”，多分类。num_class用于设置多分类问题的类别个数。
                                  learning_rate=0.01,
                                  # LightGBM 不完全信任每个弱学习器学到的残差值，为此需要给每个弱学习器拟合的残差值都乘上取值范围在(0, 1] 的 eta，设置较小的 eta 就可以多学习几个弱学习器来弥补不足的残差。推荐的候选值为：[0.01, 0.015, 0.025, 0.05, 0.1]
                                  colsample_bytree=0.6,  # 构建弱学习器时，对特征随机采样的比例，默认值为1。推荐的候选值为：[0.6, 0.7, 0.8, 0.9, 1]
                                  subsample=0.8,  # 默认值1，指定采样出 subsample * n_samples 个样本用于训练弱学习器。
                                  # 注意这里的子采样和随机森林不一样，随机森林使用的是放回抽样，而这里是不放回抽样。
                                  # 取值在(0, 1)之间，设置为1表示使用所有数据训练弱学习器。如果取值小于1，则只有一部分样本会去做GBDT的决策树拟合。选择小于1的比例可以减少方差，即防止过拟合，但是会增加样本拟合的偏差，因此取值不能太低。注意： bagging_freq 设置为非0值时才生效。
                                  # 推荐的候选值为：[0.6, 0.7, 0.8, 0.9, 1]
                                  subsample_freq=0,  # 默认值0，表示禁用样本采样。
                                  #num_leaves=2 ** 11 - 1,  # 指定叶子的个数。
                                  #max_bin=100,
                                  n_estimators=100,  # 初始树的数量
                                  n_jobs=10,  # 选择几个核
                                  #boost_from_average=False,
                                  random_state=0,
                                  metric='root_mean_squared_error'
                                  # 用于指定评估指标，可以传递各种评估方法组成的list   ‘mae’，用于回归任务，效果与 ‘mean_absolute_error’， ‘l1’ 相同。
                                  # ‘mse’，用于回归任务，效果与 ‘mean_squared_error’， ‘l2’ 相同。
                                  # ‘rmse’，用于回归任务，效果与 ‘root_mean_squared_error’， ‘l2_root’ 相同。
                                  # ‘auc’，用于二分类任务。
                                  # ‘binary’，用于二分类任务。
                                  # ‘binary_logloss’，用于二分类任务。
                                  # ‘binary_error’，用于二分类任务。
                                  # ‘multiclass’，用于多分类。
                                  # ‘multi_logloss’， 用于多分类。
                                  # ‘multi_error’， 用于多分类。
                                  )
        model.fit(df_x.fillna(0), df_y, eval_metric='auc')  # 训练clf
        '''
        param_grid = {'boosting_type':['gbdt','rf','dart'],
                      'learning_rate': [0.5,0.1, 0.05, 0.01, 0.005, 0.001],
                      'colsample_bytree':[0.6,0.7,0.8,0.9,1],
                      'subsample':[0.6,0.7,0.8,0.9,1],
                      'n_estimators': [10, 50, 100, 200],
                      'max_depth': [5, 10, 15, 20],}
        search = model_selection.RandomizedSearchCV(lgb.LGBMRegressor(subsample_freq=0, random_state=0,metric='root_mean_squared_error'),
                                              param_grid,
                                              scoring='explained_variance', n_jobs=10,
                                              refit=True,cv=3,
                                              error_score=np.nan, return_train_score=False)
        search.fit(df_x.fillna(0), df_y)
        model = search.best_estimator_
        '''


    elif model_type == 'Bagging':
        base_model = GradientBoostingRegressor(loss = 'huber',  #ls,quantile,huber,
                                          n_estimators = 100,
                                          learning_rate = 0.01,
                                          max_depth = 5,
                                          subsample = 0.8,
                                          max_features = 'auto', # if “auto”, “sqrt”,“log2”, None, max_features=n_features.
                                          criterion = 'friedman_mse',# "mse"、"friedman_mse" 和 "mae"
                                          random_state=0)
        model = BaggingRegressor(base_estimator=base_model,
                                 n_estimators=100, max_samples=0.8, max_features=0.8, oob_score=True,random_state=0)
        model.fit(df_x.fillna(0), df_y)
        model.predict(pre_x)
    elif model_type == 'AdaBoost':
        base_model = lgb.LGBMRegressor(boosting_type='gbdt',  # gbdt,df,dart
                                  #objective='regression_l1',  # objective：指定目标可选参数如下：“regression”，
                                  # 使用L2正则项的回归模型（默认值）。“regression_l1”，使用L1正则项的回归模型。“mae”，平均绝对百分比误差。“binary”，二分类。“multiclass”，多分类。num_class用于设置多分类问题的类别个数。
                                  learning_rate=0.01,
                                  # LightGBM 不完全信任每个弱学习器学到的残差值，为此需要给每个弱学习器拟合的残差值都乘上取值范围在(0, 1] 的 eta，设置较小的 eta 就可以多学习几个弱学习器来弥补不足的残差。推荐的候选值为：[0.01, 0.015, 0.025, 0.05, 0.1]
                                  colsample_bytree=0.6,  # 构建弱学习器时，对特征随机采样的比例，默认值为1。推荐的候选值为：[0.6, 0.7, 0.8, 0.9, 1]
                                  subsample=0.8,  # 默认值1，指定采样出 subsample * n_samples 个样本用于训练弱学习器。
                                  # 注意这里的子采样和随机森林不一样，随机森林使用的是放回抽样，而这里是不放回抽样。
                                  # 取值在(0, 1)之间，设置为1表示使用所有数据训练弱学习器。如果取值小于1，则只有一部分样本会去做GBDT的决策树拟合。选择小于1的比例可以减少方差，即防止过拟合，但是会增加样本拟合的偏差，因此取值不能太低。注意： bagging_freq 设置为非0值时才生效。
                                  # 推荐的候选值为：[0.6, 0.7, 0.8, 0.9, 1]
                                  subsample_freq=0,  # 默认值0，表示禁用样本采样。
                                  #num_leaves=2 ** 11 - 1,  # 指定叶子的个数。
                                  #max_bin=100,
                                  n_estimators=100,  # 初始树的数量
                                  n_jobs=10,  # 选择几个核
                                  #boost_from_average=False,
                                  random_state=0,
                                  metric='root_mean_squared_error'
                                  # 用于指定评估指标，可以传递各种评估方法组成的list   ‘mae’，用于回归任务，效果与 ‘mean_absolute_error’， ‘l1’ 相同。
                                  # ‘mse’，用于回归任务，效果与 ‘mean_squared_error’， ‘l2’ 相同。
                                  # ‘rmse’，用于回归任务，效果与 ‘root_mean_squared_error’， ‘l2_root’ 相同。
                                  # ‘auc’，用于二分类任务。
                                  # ‘binary’，用于二分类任务。
                                  # ‘binary_logloss’，用于二分类任务。
                                  # ‘binary_error’，用于二分类任务。
                                  # ‘multiclass’，用于多分类。
                                  # ‘multi_logloss’， 用于多分类。
                                  # ‘multi_error’， 用于多分类。
                                  )
        model = AdaBoostRegressor(base_estimator=base_model ,
                                  loss = 'square', # ‘linear’, 平方‘square’和指数 ‘exponential’三种选择
                                  n_estimators=100,
                                  learning_rate = 0.1,
                                  random_state= 0
                                  )
        model.fit(df_x.fillna(0), df_y)
        #scores = cross_val_score(model, df_x.fillna(0), df_y)
    elif model_type == 'voting':
        reg1 = lgb.LGBMRegressor(boosting_type='gbdt',  # gbdt,df,dart
                                  #objective='regression_l1',  # objective：指定目标可选参数如下：“regression”，
                                  # 使用L2正则项的回归模型（默认值）。“regression_l1”，使用L1正则项的回归模型。“mae”，平均绝对百分比误差。“binary”，二分类。“multiclass”，多分类。num_class用于设置多分类问题的类别个数。
                                  learning_rate=0.01,
                                  # LightGBM 不完全信任每个弱学习器学到的残差值，为此需要给每个弱学习器拟合的残差值都乘上取值范围在(0, 1] 的 eta，设置较小的 eta 就可以多学习几个弱学习器来弥补不足的残差。推荐的候选值为：[0.01, 0.015, 0.025, 0.05, 0.1]
                                  colsample_bytree=0.6,  # 构建弱学习器时，对特征随机采样的比例，默认值为1。推荐的候选值为：[0.6, 0.7, 0.8, 0.9, 1]
                                  subsample=0.8,  # 默认值1，指定采样出 subsample * n_samples 个样本用于训练弱学习器。
                                  # 注意这里的子采样和随机森林不一样，随机森林使用的是放回抽样，而这里是不放回抽样。
                                  # 取值在(0, 1)之间，设置为1表示使用所有数据训练弱学习器。如果取值小于1，则只有一部分样本会去做GBDT的决策树拟合。选择小于1的比例可以减少方差，即防止过拟合，但是会增加样本拟合的偏差，因此取值不能太低。注意： bagging_freq 设置为非0值时才生效。
                                  # 推荐的候选值为：[0.6, 0.7, 0.8, 0.9, 1]
                                  subsample_freq=0,  # 默认值0，表示禁用样本采样。
                                  #num_leaves=2 ** 11 - 1,  # 指定叶子的个数。
                                  #max_bin=100,
                                  n_estimators=100,  # 初始树的数量
                                  n_jobs=10,  # 选择几个核
                                  #boost_from_average=False,
                                  random_state=0,
                                  metric='root_mean_squared_error'
                                  # 用于指定评估指标，可以传递各种评估方法组成的list   ‘mae’，用于回归任务，效果与 ‘mean_absolute_error’， ‘l1’ 相同。
                                  # ‘mse’，用于回归任务，效果与 ‘mean_squared_error’， ‘l2’ 相同。
                                  # ‘rmse’，用于回归任务，效果与 ‘root_mean_squared_error’， ‘l2_root’ 相同。
                                  # ‘auc’，用于二分类任务。
                                  # ‘binary’，用于二分类任务。
                                  # ‘binary_logloss’，用于二分类任务。
                                  # ‘binary_error’，用于二分类任务。
                                  # ‘multiclass’，用于多分类。
                                  # ‘multi_logloss’， 用于多分类。
                                  # ‘multi_error’， 用于多分类。
                                  )
        reg2 = GradientBoostingRegressor(loss = 'huber',  #ls,quantile,huber,
                                          n_estimators = 100,
                                          learning_rate = 0.01,
                                          max_depth = 5,
                                          subsample = 0.8,
                                          max_features = 'auto', # if “auto”, “sqrt”,“log2”, None, max_features=n_features.
                                          criterion = 'friedman_mse',# "mse"、"friedman_mse" 和 "mae"
                                          random_state=0)
        reg3 = RandomForestRegressor(n_estimators=100,
                                      # 数值型参数，默认值为100，此参数指定了弱分类器的个数。设置的值越大，精确度越好，但是当 n_estimators 大于特定值之后，带来的提升效果非常有限。
                                      criterion='mse',  # 其中，参数criterion 是字符串类型，默认值为 ‘mse’，是衡量回归效果的指标。可选的还有‘mae’ 。
                                      max_features='sqrt', # if “auto”, “sqrt”,“log2”, None, max_features=n_features.
                                      max_depth=5,
                                      # 数值型，默认值None。这是与剪枝相关的参数，设置为None时，树的节点会一直分裂，直到：（1）每个叶子都是“纯”的；（2）或者叶子中包含于min_sanples_split个样本。推荐从 max_depth = 3 尝试增加，观察是否应该继续加大深度。
                                      #min_samples_split=2,
                                      # 数值型，默认值2，指定每个内部节点(非叶子节点)包含的最少的样本数。与min_samples_leaf这个参数类似，可以是整数也可以是浮点数。
                                      #min_samples_leaf=1,
                                      # 数值型，默认值1，指定每个叶子结点包含的最少的样本数。参数的取值除了整数之外，还可以是浮点数，此时（min_samples_leaf * n_samples）向下取整后的整数是每个节点的最小样本数。此参数设置的过小会导致过拟合，反之就会欠拟合。
                                      #min_weight_fraction_leaf=0.0,  # (default=0) 叶子节点所需要的最小权值
                                      # 可以为整数、浮点、字符或者None，默认值为None。此参数用于限制分枝时考虑的特征个数，超过限制个数的特征都会被舍弃。
                                      #max_leaf_nodes=None,
                                      # 数值型参数，默认值为None，即不限制最大叶子节点数。这个参数通过限制树的最大叶子数量来防止过拟合，如果设置了一个正整数，则会在建立的最大叶节点内的树中选择最优的决策树。
                                      bootstrap=True,  # 是否有放回的采样。
                                      oob_score=True,  # oob（out of band，带外）数据，即：在某次决策树训练中没有被bootstrap选中的数据
                                      random_state=0,  # # 随机种子
                                      )
        #reg4 = linear_model.BayesianRidge(alpha_1=0.1, lambda_1=0.1,fit_intercept=False, tol = 1e-5)

        model = VotingRegressor(estimators=[('LGB', reg1), ('GBRT', reg2), ('RF', reg3)])
        model.fit(df_x.fillna(0), df_y)

    elif model_type == 'step_regression':
        useful_feature = []
        error_R2 = -np.inf
        test_feature = df_x.columns.to_list()
        model = linear_model.LinearRegression(fit_intercept=False)
        while len(test_feature) > 0:
            test_result = pd.DataFrame(index=test_feature, columns=['error2'])
            for i in test_feature:
                step_feature = list(set(useful_feature).union([i]))
                model.fit(df_x[step_feature].fillna(0), df_y)
                test_result.loc[i, 'error2'] = model.score(df_x[step_feature].fillna(0), df_y)
            # 开始选取R2最大的项
            choice_code = test_result.sort_values(by='error2').index[0]
            if error_R2 < test_result['error2'].max():
                error_R2 = test_result['error2'].max()
                useful_feature.append(choice_code)
                test_feature.remove(choice_code)
            else:
                break

        model = linear_model.LinearRegression(fit_intercept=False)
        model.fit(df_x[useful_feature].fillna(0), df_y)
        return pd.Series(model.predict(pre_x[useful_feature]), index=pre_x.index)
    else:
        ValueError('print right model_type')

    '''
    elif model_type == 'Lasso':
        model = linear_model.LassoCV(alphas = [0.001,0.0001,0.00005],cv=5, fit_intercept=False)
        model.fit(df_x.fillna(0), df_y)
    elif model_type == 'XGB':
        model = xgb.XGBRegressor(learning_rate=0.01,
                                 n_estimators=100,  # 树的个数--1000棵树建立xgboost
                                 max_depth=10,  # 树的深度
                                 min_child_weight=1,  # 叶子节点最小权重
                                 #gamma=0.01,  # 惩罚项中叶子结点个数前的参数
                                 subsample=0.8,  # 随机选择80%样本建立决策树
                                 #colsample_btree=0.8,  # 随机选择80%特征建立决策树
                                 objective='reg:squarederror',  # 指定损失函数
                                 # scale_pos_weight=5,  # 解决样本个数不平衡的问题
                                 # seed=1000,
                                 nthread=20,  # CPU线程数
                                 random_state = 0
                                 )
        model.fit(df_x.fillna(0), df_y, eval_metric='auc')  # 训练clf
        model.predict(pre_x)  # 输出预测结果
    elif model_type == 'ExtraTrees':
        model = ExtraTreesRegressor(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
        model.fit(df_x.fillna(0), df_y)
        # model.feature_importances_
    elif model_type == 'MLP':
        model = MLPRegressor(random_state=1, max_iter=500)
        model.fit(df_x.fillna(0), df_y)
    '''
    # model.coef_
    # model.intercept_
    # model.get_params()
    return pd.Series(model.predict(pre_x),index=pre_x.index)
####################################### 第二步：全部因子值加工 #######################################################
class StrategyTest(object):
    def __init__(self,day=20,ind='SW1',test_start_date = 20150101,test_end_date = 20201231,fee=0.001,read_path = 'E:/FactorTest/useful_factor/'):
        self.read_path = read_path
        self.ind = ind
        self.day = day
        self.fee = fee
        # 获取因子和因子值
        test_date_list = get_date_range(get_pre_trade_date(test_start_date,100), test_end_date)
        start_date, end_date = test_date_list[0], test_date_list[-1]
        if day == 5:
            period_date_list = get_date_range(start_date, end_date, period='W')
        elif day == 10:
            period_date_list = get_date_range(start_date, end_date, period='W')
            period_date_list = period_date_list[::2]
        elif day == 20:
            period_date_list = get_date_range(start_date, end_date, period='M')
        self.period_date_list = period_date_list

        trade_date_list = [get_pre_trade_date(x, offset=-1) for x in period_date_list] # 当天收盘有结果，第二日开盘交易，所以交易日是下一天

        self.test_date_list = test_date_list
        self.period_date_list = period_date_list
        self.trade_date_list = trade_date_list

        # 对标的收益率
        code_list = get_real_ind(ind[:-1],int(ind[-1]))
        self.ind_list = code_list
        ind_open = get_daily_1factor('open', date_list=get_date_range(get_pre_trade_date(test_start_date,offset=100), get_pre_trade_date(test_end_date,offset=-30)), code_list=code_list,type=ind[:-1])
        ind_close = get_daily_1factor('close', date_list=get_date_range(get_pre_trade_date(test_start_date,offset=100),get_pre_trade_date(test_end_date, offset=-30)),code_list=code_list, type=ind[:-1])

        bench_open = get_daily_1factor('open',date_list=get_date_range(get_pre_trade_date(test_start_date,offset=100), get_pre_trade_date(test_end_date,offset=-30)),
                                       code_list=['HS300','ZZ500','wind_A'],type='bench')
        self.ind_useful = get_useful_ind(ind, test_date_list)
        self.ind_open = ind_open
        self.ind_close = ind_close
        self.bench_open = bench_open

        ind_trade_profit = ind_open.loc[trade_date_list].pct_change()
        ind_trade_profit.index = pd.Series(ind_trade_profit.index).apply(lambda x: get_pre_trade_date(x, offset=1))
        ind_trade_profit = ind_trade_profit[self.ind_useful]


        bench_trade_profit = bench_open.loc[trade_date_list].pct_change()
        bench_trade_profit.index = pd.Series(bench_trade_profit.index).apply(lambda x: get_pre_trade_date(x, offset=1))

        self.ind_trade_profit = ind_trade_profit
        self.bench_trade_profit = bench_trade_profit

        # 指数收益率
        index_open = get_daily_1factor('open', date_list=get_date_range(get_pre_trade_date(test_start_date,offset=100), get_pre_trade_date(test_end_date,offset=-30)), code_list=['wind_A','HS300','ZZ500'], type='bench')
        index_trade_profit = index_open.loc[trade_date_list].pct_change()
        index_trade_profit.index = pd.Series(index_trade_profit.index).apply(lambda x: get_pre_trade_date(x, offset=1))

        self.index_open = index_open
        self.index_trade_profit = index_trade_profit

        #ind_pct = self.ind_open.loc[[get_pre_trade_date(x,-1) for x in self.period_date_list]].pct_change(fill_method=None)[self.ind_useful].dropna(how='all')
        #ind_pct.index = [get_pre_trade_date(x) for x in ind_pct.index]
        ind_pct = self.ind_close.loc[[x for x in self.period_date_list]].pct_change(fill_method=None)[self.ind_useful].dropna(how='all')
        self.ind_pct = ind_pct

        ind_excess = (ind_pct.T - ind_pct.mean(axis=1)).T
        self.ind_excess = ind_excess

    ############################################# 因子读取部分 ########################################################
    # 1、获取所有当期因子值因子值，并将其标准化，形成一个dict
    def get_factor_dict(self, start_date, end_date, period, read_path,mv_meutral = False):
        period_date_list = get_date_range(get_pre_trade_date(start_date,25,period='M'), end_date, period=period)
        factor_dict = {}
        factor_list = [x[:-4] for x in os.listdir(read_path)]
        for factor in factor_list:
            factor_data = pd.read_pickle(read_path + factor + '.pkl')
            factor_data = factor_data.reindex(period_date_list).dropna(how='all')
            factor_data = deal_data(factor_data)
            if mv_meutral == True:
                factor_data = get_mv_neutral(factor_data, type=self.ind)
                factor_data = deal_data(factor_data)
            factor_dict[factor] = factor_data

        return factor_dict
    # 2、获取因子值每一期的IC
    def get_factor_IC(self, factor_dict, start_date, end_date, ind, period):
        period_date_list = get_date_range(start_date, get_pre_trade_date(end_date, offset=-30), period=period)
        ind_open = get_daily_1factor('open',
                                     date_list=get_date_range(start_date, get_pre_trade_date(end_date, offset=-30)),
                                     type=ind[:-1]).reindex(period_date_list).dropna(how='all')

        ind_trade_profit = ind_open.pct_change(fill_method=None)

        ic = pd.DataFrame(index=ind_trade_profit.index, columns=factor_dict.keys())
        for factor in factor_dict:
            factor_data = factor_dict[factor]
            ic[factor] = factor_data.shift(1).corrwith(ind_trade_profit, axis=1)

        return ic
        # 获取未来收益率
    # 3、获取因子之间的相关系数
    def get_factor_corr(self, factor_dict, start_date, end_date, period):
        period_date_list = get_date_range(get_pre_trade_date(start_date,25,period='M'), get_pre_trade_date(end_date, offset=-30), period=period)
        corr_dict = {}
        for date in period_date_list:
            factor_list = pd.concat([factor_dict[factor].loc[date].rename(factor) if date in factor_dict[
                factor].index else pd.Series().rename(factor) for factor in factor_dict], axis=1)
            if len(factor_list) > 0:
                corr_dict[date] = factor_list.corr()

        return corr_dict
    # 4、将按照因子值保存的dict，转化成按照date读取的dict
    def trans_factor_to_datedf(self, factor_dict, start_date, end_date, period):
        period_date_list = get_date_range(get_pre_trade_date(start_date,25,period='M'), end_date, period=period)
        factor_date_df = pd.DataFrame()
        for date in period_date_list:
            factor_list = pd.concat([factor_dict[factor].loc[date].rename(factor) if date in factor_dict[
                factor].index else pd.Series().rename(factor) for factor in factor_dict], axis=1).dropna(how='all')
            if len(factor_list) > 0:
                factor_list['date'] = date
                factor_list = factor_list.reset_index().set_index(['date','index'])
                factor_date_df = pd.concat([factor_date_df,factor_list])

        return factor_date_df.dropna(how='all')

    ############################################# 因子测试部分 ########################################################
    # 1、进行因子处理
    def deal_factor(self, factor):
        # 数据处理：标准化，中性化
        test_factor = deal_data(factor)
        #test_factor = get_mv_neutral(test_factor, type=self.ind)

        return test_factor
    # 2、获取分组的组合
    def get_group_factor(self, factor, direction, group=5):
        choice_num = (~np.isnan(factor)).sum(axis=1) // group
        # 第一步：获取分组
        group_num = pd.DataFrame(index=choice_num.index, columns=range(1,group+1))
        group_num[1], group_num[group] = choice_num, choice_num # 首尾两组必须是一致的
        # 其余的部分往中间组填充
        average_num = ((~np.isnan(factor)).sum(axis=1) - 2 * choice_num) // (group - 2)
        mod_num = ((~np.isnan(factor)).sum(axis=1) - 2 * choice_num) %  (group - 2)
        for i in trans_list_from_middle(list(range(1,group+1))[1:-1]):
            group_num[i] = average_num + mod_num.apply(lambda x: min(1, x))
            mod_num = mod_num.apply(lambda x: max(x - 1, 0))
        group_num = group_num.cumsum(axis=1)

        factor_rank = factor.rank(axis=1, ascending=direction).T
        group_dict = dict()
        for i in range(1,group+1):
            if i == 1 :
                group_dict[i] = (factor_rank <= group_num[i]).T
            else:
                group_dict[i] = ((factor_rank > group_num[i-1]).T & (factor_rank <= group_num[i]).T)

        return group_dict
    # 3、开始进行因子测试
    def single_factor_test(self,test_factor,fee):
        net_value = pd.Series(index=get_date_range(test_factor.index[0],get_pre_trade_date(test_factor.index[-1],offset=-1)))  # 计算净值
        turn = pd.Series(index=test_factor.index)
        base_money = 1 * (1 - fee)
        for i in range(0, len(test_factor) - 1):
            signal_date, next_signal_date = test_factor.index[i], test_factor.index[i + 1]
            buy_date, sell_date = get_pre_trade_date(signal_date, -1), get_pre_trade_date(next_signal_date, -1)
            month_factor = test_factor.loc[signal_date:next_signal_date].iloc[:-1].reindex(getData.get_date_range(signal_date,sell_date)).ffill()

            money_weight = base_money / month_factor.iloc[:-1].sum(axis=1).max()  # 把钱分成几份

            pct_daily = (self.ind_open.loc[buy_date:sell_date] / self.ind_open.loc[buy_date])[month_factor] * money_weight
            daily_net_value = pct_daily.ffill().sum(axis=1)

            net_value.loc[pct_daily.index] = daily_net_value

            base_money = daily_net_value.iloc[-1]  # 最后收盘时的总现金

            # 接下来考虑换手率对于现金的影响
            end_ind_weight = pct_daily.iloc[-1].fillna(0)
            next_ind_weight = test_factor.loc[next_signal_date] / test_factor.loc[next_signal_date].sum() * base_money

            change_rate = (next_ind_weight - end_ind_weight)[(next_ind_weight - end_ind_weight) > 0].sum() / base_money
            turn.loc[next_signal_date] = change_rate

            base_money = base_money * (1 - change_rate * fee)

        net_value = net_value.iloc[1:]

        return net_value, turn
    # 4、画图
    def draw_picture(self, df,sav_path=None):
        df_list = df.copy()
        df_list.index = df_list.index.astype(str)
        fig = plt.subplots(figsize=(40, 15))
        ax1 = plt.subplot(2, 1, 1)

        ax1.plot(df_list.index, df_list[['top','bottom','bench']].values)
        ax1.legend(loc='best', labels=['top','bottom','bench'])
        ax1.set_xticks([])

        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(df_list.index, df_list[['excess', 'ls']].values)
        ax2.legend(loc='best', labels=['excess', 'ls'])

        xticks = list(range(0, len(df_list.index), 20))  # 这里设置的是x轴点的位置（40设置的就是间隔了）
        xlabels = [df_list.index[x] for x in xticks]  # 这里设置X轴上的点对应在数据集中的值（这里用的数据为totalSeed）
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(xlabels, rotation=0, fontsize=20)
        for tl in ax2.get_xticklabels():
            tl.set_rotation(90)

        if sav_path != None:
            plt.savefig(sav_path + 'sentiment.jpg')
        plt.show()
    # 进行综合策略测试
    def concequence_result(self,new_factor,group=5,fee=0.001):
        # 行业轮动因子值的初步处理
        need_del_list = ['801230.SI', '852311', '801231']
        period_date_list = sorted(list(set(new_factor.index).intersection(self.period_date_list)))
        factor = new_factor[self.ind_useful][new_factor.columns.difference(need_del_list)]
        test_factor = self.deal_factor(factor.loc[period_date_list])  # 获取调整后的因子值，进行中性化，和市值中性化
        # 结果1：计算ic,rank_ic
        ic = test_factor.shift(1).corrwith(self.ind_trade_profit, axis=1).sort_index()
        rank_ic = test_factor.shift(1).rank(pct=True, axis=1).corrwith(self.ind_trade_profit.rank(pct=True, axis=1),axis=1).sort_index()
        # 结果2：获取每一期的多头组合，空头组合；基准收益率，多空收益率
        group_dict = self.get_group_factor(factor=test_factor, direction=False, group=group)
        top_ind, bottom_ind = group_dict[1], group_dict[group] # 多头组合，空头组合
        bench_mark = self.ind_useful[new_factor.columns.difference(need_del_list)].loc[factor.index]
        top_net_value, top_turn = self.single_factor_test(top_ind, fee)
        top_turn.index = top_turn.index.astype(int)
        bottom_net_value, bottom_turn = self.single_factor_test(bottom_ind, fee)
        benchmark_net_value, benchmark_turn = self.single_factor_test(bench_mark, fee)

        excess_net_value = top_net_value / benchmark_net_value # 超额净值
        ls_net_value = top_net_value / bottom_net_value # 多空净值

        top_pct = top_net_value.loc[[get_pre_trade_date(x,-1) for x in period_date_list]].pct_change(1) # 头部组合超额收益
        excess_pct = excess_net_value.loc[[get_pre_trade_date(x,-1) for x in period_date_list]].pct_change(1) # 单期超额收益
        ls_pct = ls_net_value.loc[[get_pre_trade_date(x,-1) for x in period_date_list]].pct_change(1) # 单期多空收益

        # 结果3：分年度情况
        year_list = sorted(list(set([x // 10000 for x in period_date_list[1:]])))
        test_result = pd.DataFrame(index=['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_wlratio','excess_maxdown',
                                          'ls_return', 'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown',
                                          'top_return', 'top_sharpe', 'top_turn', 'top_winrate', 'top_wlratio','top_maxdown',
                                          'ic', 'rank_ic', 'ICIR', 'rank_ICIR',
                                          ], columns=['all'] + year_list)
        for year in test_result.columns:
            year_date = top_net_value.index.to_list() if year == 'all' else \
                top_net_value.loc[get_pre_trade_date(year * 10000 + 101):get_pre_trade_date(year * 10000 + 1231)].index.to_list()  # 日期列表
            period_year_list = top_pct.index.to_list() if year == 'all' else \
                top_pct.loc[get_pre_trade_date(year * 10000 + 101,-2):get_pre_trade_date((year+1) * 10000 + 101,-1)].index.to_list()
            period_date = ic .index.to_list() if year == 'all' else \
                ic.loc[(year * 10000 + 101):(year * 10000 + 1231)].index.to_list()  # 日期列表


            test_result.loc[['ic','rank_ic'], year] = ic.loc[period_date].mean(),rank_ic.loc[period_date].mean()
            test_result.loc['ICIR', year] = ic.loc[period_date].mean() / ic.loc[period_date].std() * np.sqrt(240 / self.day)
            test_result.loc['rank_ICIR', year] = rank_ic.loc[period_date].mean() / rank_ic.loc[period_date].std() * np.sqrt(240 / self.day)

            # 计算组合收益率
            for name in ['top', 'excess', 'ls']:
                net_value = top_net_value.loc[year_date].copy() if name == 'top' else excess_net_value.loc[year_date].copy() if name == 'excess' else ls_net_value.loc[year_date].copy()
                net_value = net_value / net_value.iloc[0]

                period_pct = top_pct.loc[period_year_list].copy() if name == 'top' else excess_pct.loc[period_year_list].copy() if name == 'excess' else ls_pct.loc[period_year_list].copy()

                test_result.loc[name + '_return', year] = net_value.iloc[-1] ** (252 / len(year_date)) - 1
                test_result.loc[name + '_sharpe', year] = period_pct.mean() / period_pct.std() * np.sqrt(240 / self.day)
                if name == 'top':
                    test_result.loc[name + '_turn', year] = top_turn.mean() if year == 'all' else top_turn.loc[get_pre_trade_date(year * 10000 + 101):get_pre_trade_date(year * 10000 + 1231)].mean()

                test_result.loc[name + '_winrate', year] = (period_pct > 0).sum() / len(period_pct)
                test_result.loc[name + '_wlratio', year] = -period_pct[period_pct > 0].mean() / period_pct[period_pct < 0].mean()
                test_result.loc[name + '_maxdown', year] = ((net_value - net_value.cummax()) / net_value.cummax()).min()

        test_result = test_result.astype(float).round(4)

        all_net_value = pd.concat([top_net_value.rename('top'),bottom_net_value.rename('bottom'),benchmark_net_value.rename('bench'), \
                                   excess_net_value.rename('excess'),ls_net_value.rename('ls')],axis=1)
        all_pct = pd.concat([top_pct.rename('top_pct'),excess_pct.rename('excess_pct'),ls_pct.rename('ls_pct')],axis=1)
        self.draw_picture(all_net_value, sav_path = 'C:/Users/86181/Desktop/')

        return top_ind, bottom_ind, test_result, all_net_value, all_pct

    ############################################# 因子合成部分 ########################################################
    # 获取单期的因子值，选取结果



    # 因子处理1：当期使用的因子，必须是1年期以上的因子，且在过去12个月和其他因子的相关系数，必须＜0.6；（确保是上个月的因子值的相关系数，当期的因子值不计算相关系数）
    def deal_corr(self, start_date, end_date, corr_rate=0.6, corr_period=12, period='M'):
        factor_dict = self.get_factor_dict(start_date, end_date, period, self.read_path)  # 因子读取，进行标准化和中性化
        corr_dict = self.get_factor_corr(factor_dict, start_date, end_date, period)  # 获取因子之间的相关系数
        factor_date_df = self.trans_factor_to_datedf(factor_dict, start_date, end_date, period)  # 按照日期进行因子值保存
        ic = self.get_factor_IC(factor_dict, factor_date_df.index[0][0], factor_date_df.index[-1][0], self.ind, period)  # 获取因子ic
        ic_rol12 = ic.rolling(12, min_periods = 6).mean()

        date_list = get_date_range(get_pre_trade_date(factor_date_df.index[0][0]), factor_date_df.index[-1][0],period='M')
        choice_factor = pd.DataFrame(False, index=date_list, columns=factor_dict)
        for date in date_list:
            today_use_factor = []
            period_date_list = date_list[max(0, date_list.index(date) - corr_period):date_list.index(date)]
            # 第一步：使用有12期以上因子数据的因子
            for factor in factor_dict:
                if len(factor_dict[factor].loc[:date]) >= 12:
                    today_use_factor.append(factor)
            # 第二步：进行因子相关系数的筛选
            if len(today_use_factor) > 0:
                corr_date = pd.DataFrame(0, index=factor_date_df.columns, columns=factor_date_df.columns).loc[today_use_factor, today_use_factor]
                for d in period_date_list:
                    if d in corr_dict.keys():
                        corr_date = corr_date + corr_dict[d].loc[today_use_factor, today_use_factor]
                corr_date = corr_date / len(period_date_list)
                for i in corr_date.index:
                    flag = 0
                    for j in corr_date.columns:
                        if i != j:
                            if corr_date.loc[i, j] > corr_rate:
                                # 如果超过相关系数阈值，则看过去12个月的平均IC
                                if abs(ic_rol12.loc[date, i]) < abs(ic_rol12.loc[date, j]):
                                    flag = 1
                                    break
                    if flag == 0:
                        choice_factor.loc[date, i] = True

        return choice_factor
    # 最没用的方法—————直接等权
    def strategy_sameweight(self,start_date,end_date,period):
        factor_dict = self.get_factor_dict(start_date, end_date, period, self.read_path)  # 因子读取，进行标准化和中性化
        factor_date_df = self.trans_factor_to_datedf(factor_dict, start_date, end_date, period)  # 按照日期进行因子值保存
        # 要求该因子至少已经运行1年
        test_period = get_date_range(get_pre_trade_date(start_date),end_date,period='M')
        new_factor = pd.DataFrame(index=test_period,columns=self.ind_useful.columns)
        for date in test_period:
            useful_factor = []
            for factor in factor_dict:
                if len(factor_dict[factor].loc[:date]) >= 12:
                    useful_factor.append(factor)
            if len(useful_factor) >0:
                new_factor.loc[date] = factor_date_df.loc[date][useful_factor].mean(axis=1)
        new_factor = new_factor.dropna(how='all').astype(float)

        top_ind, bottom_ind, test_result, all_net_value, all_pct = self.concequence_result(new_factor, group=5,fee=self.fee)

        return top_ind, bottom_ind, test_result, all_net_value, all_pct, new_factor
    # 调整方法1——剔除相关系数后，等权相加（每一期根据前12期的相关系数，将相关系数＞0.6的因子剔除）
    def strategy1_del_corr(self,start_date,end_date,period,corr_rate = 0.6,corr_period = 12):
        factor_dict = self.get_factor_dict(start_date, end_date, period, self.read_path)  # 因子读取，进行标准化和中性化
        factor_date_df = self.trans_factor_to_datedf(factor_dict, start_date, end_date, period)  # 按照日期进行因子值保存
        useful_factor = self.deal_corr(start_date, end_date,corr_rate,corr_period)

        test_period = get_date_range(get_pre_trade_date(start_date), end_date, period='M')
        new_factor1 = pd.DataFrame(index=test_period,columns=self.ind_useful.columns)
        # 先进行筛选，过去N个月个股的相关系数
        for date in test_period:
            use_factor = useful_factor.loc[date][useful_factor.loc[date]==True].index.to_list()
            new_factor1.loc[date] = factor_date_df.loc[date][use_factor].mean(axis=1)

        new_factor1 = new_factor1.dropna(how='all').astype(float)

        top_ind, bottom_ind, test_result, all_net_value, all_pct = self.concequence_result(new_factor1, group=5,fee=self.fee)

        return top_ind, bottom_ind, test_result, all_net_value, all_pct, new_factor1
    # 可用方法2——你未必能提前知道因子表现的好坏，所以你每一期都要进行筛选
    def strategy2_ic_select(self,start_date,end_date,ind,period,corr_rate = 0.6,corr_period = 12,ic_period = 12):
        factor_dict = self.get_factor_dict(start_date, end_date, period, self.read_path)  # 因子读取，进行标准化和中性化
        factor_date_df = self.trans_factor_to_datedf(factor_dict, start_date, end_date, period)  # 按照日期进行因子值保存
        useful_factor = self.deal_corr(start_date, end_date,corr_rate,corr_period) # 因子相关系数筛选完毕之后的因子
        # 因子IC的筛选
        ic = self.get_factor_IC(factor_dict, factor_date_df.index[0][0], factor_date_df.index[-1][0], self.ind, period)  # 获取因子ic
        ic_rol = ic.rolling(ic_period).mean()
        ic_rol = (ic_rol.T / ic_rol.sum(axis=1)).T

        test_period = get_date_range(get_pre_trade_date(start_date), end_date, period='M')
        new_factor2 = pd.DataFrame(index=test_period,columns=self.ind_useful.columns)
        for date in test_period:
            use_factor = useful_factor.loc[date][useful_factor.loc[date]==True].index.to_list()
            if len(use_factor) > 0:
                # 使用过去12个月IC绝对值最大的因子
                use_factor = abs(ic_rol.loc[date,use_factor]).sort_values().iloc[-15:].index.to_list()
                new_factor2.loc[date] = factor_date_df.loc[date][use_factor].mean(axis=1)

        new_factor2 = new_factor2.dropna(how='all').astype(float)
        top_ind, bottom_ind, test_result, all_net_value, all_pct = self.concequence_result(new_factor2, group=5,fee=self.fee)

        return top_ind, bottom_ind, test_result, all_net_value, all_pct, new_factor2

    # 奇怪方法：当因子相关系数较高时，直接将2个因子都删除，而不是选用其中一个
    def strange_strategy(self,start_date,end_date,period,corr_rate = 0.6, corr_period = 6):
        factor_dict = self.get_factor_dict(start_date, end_date, period, self.read_path)  # 因子读取，进行标准化和中性化
        factor_date_df = self.trans_factor_to_datedf(factor_dict, start_date, end_date, period)  # 按照日期进行因子值保存
        corr_dict = self.get_factor_corr(factor_dict, start_date, end_date, period)  # 获取因子之间的相关系数

        date_list = get_date_range(get_pre_trade_date(start_date), end_date, period='M')
        strange_factor = pd.DataFrame(index=date_list, columns=self.ind_useful.columns)
        # 先进行筛选，过去N个月个股的相关系数
        for date in date_list:
            today_use_factor = []
            period_date_list = date_list[max(0, date_list.index(date) - corr_period):date_list.index(date)]
            # 第一步：使用有12期以上因子数据的因子
            for factor in factor_dict:
                if len(factor_dict[factor].loc[:date]) >= 12:
                    today_use_factor.append(factor)
            # 第二步：进行因子相关系数的筛选
            choice_factor = []
            if len(today_use_factor) > 0:
                corr_date = pd.DataFrame(0, index=factor_date_df.columns, columns=factor_date_df.columns).loc[today_use_factor, today_use_factor]
                for d in period_date_list:
                    corr_date = corr_date + corr_dict[d].loc[today_use_factor, today_use_factor]
                corr_date = corr_date / len(period_date_list)
                for i in corr_date.index:
                    flag = 0
                    for j in corr_date.columns:
                        if i != j:
                            if corr_date.loc[i, j] > corr_rate:
                                flag = 1
                                break
                    if flag == 0:
                        choice_factor.append(i)

            strange_factor.loc[date] = factor_date_df.loc[date][choice_factor].mean(axis=1)

        strange_factor = strange_factor.dropna(how='all').astype(float)

        top_ind, bottom_ind, test_result, all_net_value, all_pct = self.concequence_result(strange_factor, group=5,fee=self.fee)

        return top_ind, bottom_ind, test_result, all_net_value, all_pct, strange_factor
    # 可用方法3——FC Model法：单因子出结果，然后所有因子结果进行汇总
    def strategy3_FCModel(self,start_date,end_date,period,corr_rate = 0.6,corr_period = 12):
        factor_dict = self.get_factor_dict(start_date, end_date, period, self.read_path)  # 因子读取，进行标准化和中性化
        useful_factor = self.deal_corr(start_date, end_date,corr_rate,corr_period)  # 因子相关系数筛选完毕之后的因子

        ind_pct = self.ind_excess.copy()
        # 对于每一期，先使用单因子进行回归，得到预测值
        test_period = get_date_range(get_pre_trade_date(start_date), end_date, period='M')
        new_factor3 = pd.DataFrame(index=test_period,columns=self.ind_useful.columns)
        for date in tqdm(test_period):
            use_factor = useful_factor.loc[date][useful_factor.loc[date] == True].index.to_list()
            pre_factor = pd.DataFrame(index = ind_pct.columns,columns=use_factor)
            for factor in (use_factor):
                # 针对单因子进行回归
                factor_data = factor_dict[factor].loc[:date].shift(1).iloc[-12:].dropna(how='all').loc[start_date:end_date]
                para = pd.DataFrame(index = factor_data.index,columns=['const','x'])
                for test_date in factor_data.index:
                    x = factor_data.loc[test_date].dropna()
                    y = ind_pct.loc[test_date].dropna()
                    same_ind = set(x.index).intersection(y.index)
                    x,y = x[same_ind], y[same_ind]
                    #x = sm.add_constant(x)
                    ols = sm.OLS(y,x).fit()
                    #ols.summary()
                    para.loc[test_date] = ols.params.values
                para = para.mean()
                pre_factor[factor] = para['x'] * factor_dict[factor].loc[date] + para['const']

            new_factor3.loc[date] = pre_factor[(pre_factor.rank(pct=True,axis=1) >=0.1) & (pre_factor.rank(pct=True,axis=1) <=0.9)].mean(axis=1)

        new_factor3 = new_factor3.dropna(how='all').astype(float)
        top_ind, bottom_ind, test_result, all_net_value, all_pct = self.concequence_result(new_factor3, group=5,fee=self.fee)

        return top_ind, bottom_ind, test_result, all_net_value, all_pct, new_factor3

    # 可用方法4：线性回归法
    def strategy4_sklearn(self,start_date,end_date,period,train_days = 12,pre_days = 1, model_type = 'linear',corr_rate = 0.6,corr_period = 12,if_test = True):
        factor_dict = self.get_factor_dict(start_date, end_date, period, self.read_path)  # 因子读取，进行标准化和中性化
        factor_date_df = self.trans_factor_to_datedf(factor_dict, start_date, end_date, period)  # 按照日期进行因子值保存
        useful_factor = self.deal_corr(start_date, end_date,corr_rate,corr_period)  # 因子相关系数筛选完毕之后的因子
        ind_pct = self.ind_excess.copy()
        #ind_pct = self.ind_pct.copy()
        ind_pct = ind_pct.shift(-1) * 100
        # 每一期，利用过去12个月进行回归
        test_period = get_date_range(get_pre_trade_date(start_date), end_date, period='M')
        new_factor = pd.DataFrame(index=test_period,columns=self.ind_useful.columns)

        for date in tqdm(test_period[::pre_days]):
            use_factor = useful_factor.loc[date][useful_factor.loc[date] == True].index.to_list()
            if len(use_factor) >0:
                test_date_list = useful_factor.loc[:date].iloc[-train_days-1:-1].index.to_list()
                df_x = factor_date_df.loc[test_date_list][use_factor].dropna(how='all')
                df_y = ind_pct.loc[test_date_list].stack()
                df_x,df_y = df_x.loc[set(df_x.index).intersection(df_y.index)].sort_index(), df_y.loc[set(df_x.index).intersection(df_y.index)].sort_index()
                # 开始进行模型训练
                pre_x = factor_date_df.loc[date:get_pre_trade_date(date,offset=-pre_days+1,period='M')][use_factor].fillna(0)

                #pre_x = factor_date_df.loc[date][use_factor].fillna(0)
                pre_y = factor_model(df_x,df_y,pre_x,model_type)

                #new_factor4.loc[date] = pre_y
                new_factor.loc[date:get_pre_trade_date(date,offset=-pre_days+1,period='M'),self.ind_useful.columns] = \
                    pre_y.unstack().loc[date:get_pre_trade_date(date,offset=-pre_days+1,period='M')]

        new_factor = new_factor.dropna(how='all').astype(float)
        #new_factor4=new_factor4.replace(0,np.nan).dropna(how='all').astype(float)
        if if_test == False:
            return new_factor
        top_ind, bottom_ind, test_result, all_net_value, all_pct = self.concequence_result(new_factor, group=5,fee=self.fee)

        return top_ind, bottom_ind, test_result, all_net_value, all_pct, new_factor

start_date,end_date,ind = 20140101, 20221031, 'SW1'
period = 'M'
self = StrategyTest(test_start_date = start_date,test_end_date=end_date,fee=0.001)
test_start_date = 20151231

def normal_result():
    # 第一个是假设状态：就是直接等权求和（只要求因子至少具有12个月以上的数据）
    first_top_ind, first_bottom_ind, first_test_result, first_all_net_value, first_all_pct, first_new_factor = \
        self.strategy_sameweight(test_start_date,end_date,period)
    # 第二个是假设状态的改良版：根据相关系数剔除一些，保留IC最大的，等权求和
    top_ind1, bottom_ind1, test_result1, all_net_value1, all_pct1, new_factor1 = \
        self.strategy1_del_corr(test_start_date,end_date,period,corr_rate = 0.6,corr_period = 12)
    # 第三个是模拟真实状态：每次选取IC比较好或者满足条件的因子，等权求和
    top_ind2, bottom_ind2, test_result2, all_net_value2, all_pct2, new_factor2 = \
        self.strategy2_ic_select(test_start_date,end_date,ind,period,corr_rate = 0.6,corr_period = 6,ic_period = 6)

    # 从第四个开始，就是开始测试哪些模型相对比较好用，进行模型层面的筛选和使用
    # 1、奇怪的方法
    strange_top_ind, strange_bottom_ind, strange_test_result, strange_all_net_value, strange_all_pct, strange_new_factor = \
        self.strange_strategy(test_start_date,end_date,period,corr_rate = 0.6, corr_period = 12)
    # 2、FC model
    FC_top_ind, FC_bottom_ind, FC_test_result, FC_all_net_value, FC_all_pct, FC_new_factor = \
        self.strategy3_FCModel(test_start_date,end_date,period,corr_rate = 0.6,corr_period = 6)

################################ 线性模型的寻优寻优 ########################################
# 1、线性模型寻优
def linear_find_best():
    corr_list = [0.5,0.55,0.6,0.65,0.7,0.75,0.8]
    corr_period_list = [6,12,24]
    train_days_list = [3,6,12,24]
    pre_days_list = [1,3,4,6]
    save_path = 'E:/FactorTest/model/linear/'
    result_dict = {}
    for corr_rate in corr_list:
        for corr_period in corr_period_list:
            for train_days in train_days_list:
                for pre_days in pre_days_list:
                    name = 'corr' + str(corr_rate) + 'period' + str(corr_period) + 'train_days' + str(train_days) + 'pre_days' + str(pre_days)
                    print(name)
                    linear_top_ind, linear_bottom_ind, linear_test_result, linear_all_net_value, linear_all_pct, linear_factor = \
                        self.strategy4_sklearn(test_start_date,end_date,period,model_type = 'linear',
                        train_days = train_days, pre_days = pre_days, corr_rate = corr_rate , corr_period = corr_period)

                    flag = 0
                    for dict_name in result_dict:
                        other_result = result_dict[dict_name]
                        if (linear_test_result.loc[['excess_return','excess_sharpe','excess_winrate','excess_maxdown','excess_wlratio','ls_return','ls_sharpe','ls_winrate','ls_wlratio','ls_maxdown'],'all'] < \
                            other_result.loc[['excess_return','excess_sharpe','excess_winrate','excess_maxdown','excess_wlratio','ls_return','ls_sharpe','ls_winrate','ls_wlratio','ls_maxdown'],'all']).sum() >= 8:
                            print(pd.concat([linear_test_result['all'],other_result['all']],axis=1))
                            print(name + ' 结果不行')
                            flag = 1
                            break
                        elif linear_test_result.loc['excess_return','all'] < other_result.loc['excess_return','all'] / 2:
                            print(pd.concat([linear_test_result['all'],other_result['all']],axis=1))
                            print(name + ' 超额收益不行')
                            flag = 1
                            break
                        elif linear_test_result.loc['excess_maxdown', 'all'] < -0.15:
                            print(linear_test_result)
                            print(name + ' 回撤不行')
                            flag = 1
                            break
                        elif -linear_test_result.loc['excess_return', 'all'] / linear_test_result.loc['excess_maxdown', 'all'] < 0.6:
                            print(linear_test_result)
                            print(-linear_test_result.loc['excess_return', 'all'] / linear_test_result.loc['excess_maxdown', 'all'])
                            print(name + ' 收益回撤比不行')
                            flag = 1
                            break

                    if flag == 0:
                        linear_test_result.to_pickle(save_path + name +'.pkl')
                        result_dict[name] = linear_test_result

    # 内部还要再清洗一遍；就是因为过早放入，而导致没有被剔除的样本剔除
    useful_result_dict = {}
    factor_list = [x[:-4] for x in os.listdir(save_path)]
    for factor_name in factor_list:
        flag = 0
        factor_result = pd.read_pickle(save_path + factor_name + '.pkl')

        for dict_name in list(useful_result_dict.keys()):
            other_result = useful_result_dict[dict_name]
            if (factor_result.loc[['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio', 'ls_return',
                     'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all'] < \
                other_result.loc[['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio', 'ls_return',
                     'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all']).sum() >= 8:
                print(pd.concat([factor_result['all'], other_result['all']], axis=1))
                print(factor_name + ' 结果不行')
                flag = 1
                break

            elif (factor_result.loc[['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio', 'ls_return',
                     'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all'] > \
                other_result.loc[
                    ['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio', 'ls_return',
                     'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all']).sum() >= 8:
                print(pd.concat([factor_result['all'], other_result['all']], axis=1))
                print(dict_name + ' 结果不行')
                useful_result_dict.pop(dict_name, None)
                #os.remove(save_path + dict_name + '.pkl')

        if flag == 0:
            useful_result_dict[factor_name] = factor_result
# 2、Ridge模型寻优
def ridge_find_best():
    corr_list = [0.5,0.55,0.6,0.65,0.7]
    corr_period_list = [6,12,24]
    train_days_list = [6,12,24]
    pre_days_list = [1,3,4,6]
    save_path = 'E:/FactorTest/model/Ridge/'
    ridge_dict = {}
    for corr_rate in corr_list:
        for corr_period in corr_period_list:
            for train_days in train_days_list:
                for pre_days in pre_days_list:
                    name = 'corr' + str(corr_rate) + 'period' + str(corr_period) + 'train_days' + str(train_days) + 'pre_days' + str(pre_days)
                    print(name)
                    Ridge_top_ind, Ridge_bottom_ind, Ridge_test_result, Ridge_all_net_value, Ridge_all_pct, Ridge_factor = \
                        self.strategy4_sklearn(test_start_date, end_date, period, model_type='Ridge',train_days=train_days, pre_days=pre_days, corr_rate=corr_rate,corr_period=corr_period)
                    flag = 0
                    if Ridge_test_result.loc['excess_maxdown', 'all'] < -0.15:
                        print(Ridge_test_result)
                        print(name + ' 回撤不行')
                        flag = 1
                        continue
                    elif -Ridge_test_result.loc['excess_return', 'all'] / Ridge_test_result.loc['excess_maxdown', 'all'] < 1:
                        print(Ridge_test_result)
                        print(-Ridge_test_result.loc['excess_return', 'all'] / Ridge_test_result.loc['excess_maxdown', 'all'])
                        print(name + ' 收益回撤比不行')
                        flag = 1
                        continue

                    for dict_name in ridge_dict:
                        other_result = ridge_dict[dict_name]
                        if (Ridge_test_result.loc[['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio',
                                 'ls_return', 'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all'] < \
                            other_result.loc[['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio',
                                 'ls_return', 'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all']).sum() >= 8:
                            print(pd.concat([Ridge_test_result['all'], other_result['all']], axis=1))
                            print(name + ' 结果不行')
                            flag = 1
                            break
                        if Ridge_test_result.loc['excess_return', 'all'] < other_result.loc['excess_return', 'all'] / 2:
                            print(pd.concat([Ridge_test_result['all'], other_result['all']], axis=1))
                            print(name + ' 超额收益不行')
                            flag = 1
                            continue

                    if flag == 0:
                        Ridge_test_result.to_pickle(save_path + name + '.pkl')
                        ridge_dict[name] = Ridge_test_result

    # 内部还要再清洗一遍；就是因为过早放入，而导致没有被剔除的样本剔除
    useful_result_dict = {}
    factor_list = [x[:-4] for x in os.listdir(save_path)]
    for factor_name in factor_list:
        flag = 0
        factor_result = pd.read_pickle(save_path + factor_name + '.pkl')

        for dict_name in list(useful_result_dict.keys()):
            other_result = useful_result_dict[dict_name]
            if (factor_result.loc[['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio', 'ls_return',
                     'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all'] < \
                other_result.loc[['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio', 'ls_return',
                     'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all']).sum() >= 8:
                print(pd.concat([factor_result['all'], other_result['all']], axis=1))
                print(factor_name + ' 结果不行')
                flag = 1
                break
            elif (factor_result.loc[['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio', 'ls_return',
                     'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all'] > \
                other_result.loc[
                    ['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio', 'ls_return',
                     'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all']).sum() >= 8:
                print(pd.concat([factor_result['all'], other_result['all']], axis=1))
                print(dict_name + ' 结果不行')
                useful_result_dict.pop(dict_name, None)

        if flag == 0:
            useful_result_dict[factor_name] = factor_result
# 3、Lars模型寻优
def Lars_find_best():
    corr_list = [0.5,0.55,0.6,0.65,0.7]
    corr_period_list = [6,12,24]
    train_days_list = [6,12,24]
    pre_days_list = [1,3,4,6]
    save_path = 'E:/FactorTest/model/Lars/'
    ridge_dict = {}
    for corr_rate in corr_list:
        for corr_period in corr_period_list:
            for train_days in train_days_list:
                for pre_days in pre_days_list:
                    name = 'corr' + str(corr_rate) + 'period' + str(corr_period) + 'train_days' + str(train_days) + 'pre_days' + str(pre_days)
                    print(name)
                    Lars_top_ind, Lars_bottom_ind, Lars_test_result, Lars_all_net_value, Lars_all_pct, Lars_factor = \
                        self.strategy4_sklearn(test_start_date, end_date, period, model_type='Lars',train_days=train_days, pre_days=pre_days, corr_rate=corr_rate,corr_period=corr_period)
                    flag = 0
                    if Lars_test_result.loc['excess_maxdown', 'all'] < -0.15:
                        print(Lars_test_result)
                        print(name + ' 回撤不行')
                        flag = 1
                        continue
                    elif -Lars_test_result.loc['excess_return', 'all'] / Lars_test_result.loc['excess_maxdown', 'all'] < 1:
                        print(Lars_test_result)
                        print(-Lars_test_result.loc['excess_return', 'all'] / Lars_test_result.loc['excess_maxdown', 'all'])
                        print(name + ' 收益回撤比不行')
                        flag = 1
                        continue

                    for dict_name in ridge_dict:
                        other_result = ridge_dict[dict_name]
                        if (Lars_test_result.loc[['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio',
                                 'ls_return', 'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all'] < \
                            other_result.loc[['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio',
                                 'ls_return', 'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all']).sum() >= 8:
                            print(pd.concat([Lars_test_result['all'], other_result['all']], axis=1))
                            print(name + ' 结果不行')
                            flag = 1
                            break
                        if Lars_test_result.loc['excess_return', 'all'] < other_result.loc['excess_return', 'all'] / 2:
                            print(pd.concat([Lars_test_result['all'], other_result['all']], axis=1))
                            print(name + ' 超额收益不行')
                            flag = 1
                            continue

                    if flag == 0:
                        Lars_test_result.to_pickle(save_path + name + '.pkl')
                        ridge_dict[name] = Lars_test_result

    # 内部还要再清洗一遍；就是因为过早放入，而导致没有被剔除的样本剔除
    useful_result_dict = {}
    factor_list = [x[:-4] for x in os.listdir(save_path)]
    for factor_name in factor_list:
        flag = 0
        factor_result = pd.read_pickle(save_path + factor_name + '.pkl')

        for dict_name in list(useful_result_dict.keys()):
            other_result = useful_result_dict[dict_name]
            if (factor_result.loc[['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio', 'ls_return',
                     'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all'] < \
                other_result.loc[['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio', 'ls_return',
                     'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all']).sum() >= 8:
                print(pd.concat([factor_result['all'], other_result['all']], axis=1))
                print(factor_name + ' 结果不行')
                flag = 1
                break
            elif (factor_result.loc[['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio', 'ls_return',
                     'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all'] > \
                other_result.loc[
                    ['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio', 'ls_return',
                     'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all']).sum() >= 8:
                print(pd.concat([factor_result['all'], other_result['all']], axis=1))
                print(dict_name + ' 结果不行')
                useful_result_dict.pop(dict_name, None)

        if flag == 0:
            useful_result_dict[factor_name] = factor_result
# Bayes模型寻优
def Bayes_find_best():
    corr_list = [0.5,0.55,0.6,0.65,0.7]
    corr_period_list = [6,12,24]
    train_days_list = [6,12,24]
    pre_days_list = [1,3,4,6]
    save_path = 'E:/FactorTest/model/Bayes/'
    ridge_dict = {}
    for corr_rate in corr_list:
        for corr_period in corr_period_list:
            for train_days in train_days_list:
                for pre_days in pre_days_list:
                    name = 'corr' + str(corr_rate) + 'period' + str(corr_period) + 'train_days' + str(train_days) + 'pre_days' + str(pre_days)
                    print(name)
                    Bayes_top_ind, Bayes_bottom_ind, Bayes_test_result, Bayes_all_net_value, Bayes_all_pct, Bayes_factor = \
                        self.strategy4_sklearn(test_start_date, end_date, period, model_type='Bayes',train_days=train_days, pre_days=pre_days, corr_rate=corr_rate,corr_period=corr_period)
                    flag = 0
                    if Bayes_test_result.loc['excess_maxdown', 'all'] < -0.15:
                        print(Bayes_test_result)
                        print(name + ' 回撤不行')
                        flag = 1
                        continue
                    elif -Bayes_test_result.loc['excess_return', 'all'] / Bayes_test_result.loc['excess_maxdown', 'all'] < 1:
                        print(Bayes_test_result)
                        print(-Bayes_test_result.loc['excess_return', 'all'] / Bayes_test_result.loc['excess_maxdown', 'all'])
                        print(name + ' 收益回撤比不行')
                        flag = 1
                        continue

                    for dict_name in ridge_dict:
                        other_result = ridge_dict[dict_name]
                        if (Bayes_test_result.loc[['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio',
                                 'ls_return', 'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all'] < \
                            other_result.loc[['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio',
                                 'ls_return', 'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all']).sum() >= 8:
                            print(pd.concat([Bayes_test_result['all'], other_result['all']], axis=1))
                            print(name + ' 结果不行')
                            flag = 1
                            break
                        if Bayes_test_result.loc['excess_return', 'all'] < other_result.loc['excess_return', 'all'] / 2:
                            print(pd.concat([Bayes_test_result['all'], other_result['all']], axis=1))
                            print(name + ' 超额收益不行')
                            flag = 1
                            continue

                    if flag == 0:
                        Bayes_test_result.to_pickle(save_path + name + '.pkl')
                        ridge_dict[name] = Bayes_test_result

    # 内部还要再清洗一遍；就是因为过早放入，而导致没有被剔除的样本剔除
    useful_result_dict = {}
    factor_list = [x[:-4] for x in os.listdir(save_path)]
    for factor_name in factor_list:
        flag = 0
        factor_result = pd.read_pickle(save_path + factor_name + '.pkl')

        for dict_name in list(useful_result_dict.keys()):
            other_result = useful_result_dict[dict_name]
            if (factor_result.loc[['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio', 'ls_return',
                     'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all'] < \
                other_result.loc[['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio', 'ls_return',
                     'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all']).sum() >= 8:
                print(pd.concat([factor_result['all'], other_result['all']], axis=1))
                print(factor_name + ' 结果不行')
                flag = 1
                break
            elif (factor_result.loc[['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio', 'ls_return',
                     'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all'] > \
                other_result.loc[
                    ['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio', 'ls_return',
                     'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all']).sum() >= 8:
                print(pd.concat([factor_result['all'], other_result['all']], axis=1))
                print(dict_name + ' 结果不行')
                useful_result_dict.pop(dict_name, None)

        if flag == 0:
            useful_result_dict[factor_name] = factor_result

# 寻优结果
def find_linear_result():
    corr_rate1,corr_period1,train_days1,pre_days1 = 0.55, 12, 12, 3 #choice1
    corr_rate2,corr_period2,train_days2,pre_days2 = 0.5, 12, 12, 4 #choice2（和choice1接近，二者可以选一）
    corr_rate3, corr_period3, train_days3, pre_days3 = 0.5, 12, 24, 6 #choice3

    linear_top_ind1, linear_bottom_ind1, linear_test_result1, linear_all_net_value1, linear_all_pct1, linear_factor1 = \
        self.strategy4_sklearn(test_start_date, end_date, period, model_type='linear',
                               train_days=train_days1, pre_days=pre_days1, corr_rate=corr_rate1, corr_period=corr_period1)
    linear_top_ind2, linear_bottom_ind2, linear_test_result2, linear_all_net_value2, linear_all_pct2, linear_factor2 = \
        self.strategy4_sklearn(test_start_date, end_date, period, model_type='linear',
                               train_days=train_days2, pre_days=pre_days2, corr_rate=corr_rate2, corr_period=corr_period2)
    linear_top_ind3, linear_bottom_ind3, linear_test_result3, linear_all_net_value3, linear_all_pct3, linear_factor3 = \
        self.strategy4_sklearn(test_start_date, end_date, period, model_type='linear',
                               train_days=train_days3, pre_days=pre_days3, corr_rate=corr_rate3, corr_period=corr_period3)

    Ridge_top_ind1, Ridge_bottom_ind1, Ridge_test_result1, Ridge_all_net_value1, Ridge_all_pct1,Ridge_factor1 = \
        self.strategy4_sklearn(test_start_date, end_date, period, model_type='Ridge',
                               train_days=train_days1, pre_days=pre_days1, corr_rate=corr_rate1, corr_period=corr_period1)
    Ridge_top_ind2, Ridge_bottom_ind2, Ridge_test_result2, Ridge_all_net_value2, Ridge_all_pct2, Ridge_factor2 = \
        self.strategy4_sklearn(test_start_date, end_date, period, model_type='Ridge',
                               train_days=train_days2, pre_days=pre_days2, corr_rate=corr_rate2, corr_period=corr_period2)
    Ridge_top_ind3, Ridge_bottom_ind3, Ridge_test_result3, Ridge_all_net_value3, Ridge_all_pct3, Ridge_factor3 = \
        self.strategy4_sklearn(test_start_date, end_date, period, model_type='Ridge',
                               train_days=train_days3, pre_days=pre_days3, corr_rate=corr_rate3, corr_period=corr_period3)

    Lars_top_ind1, Lars_bottom_ind1, Lars_test_result1, Lars_all_net_value1, Lars_all_pct1, Lars_factor1 = \
        self.strategy4_sklearn(test_start_date, end_date, period, model_type='Lars',
                               train_days=train_days1, pre_days=pre_days1, corr_rate=corr_rate1, corr_period=corr_period1)
    Lars_top_ind2, Lars_bottom_ind2, Lars_test_result2, Lars_all_net_value2, Lars_all_pct2, Lars_factor2 = \
        self.strategy4_sklearn(test_start_date, end_date, period, model_type='Lars',
                               train_days=train_days2, pre_days=pre_days2, corr_rate=corr_rate2, corr_period=corr_period2)
    Lars_top_ind3, Lars_bottom_ind3, Lars_test_result3, Lars_all_net_value3, Lars_all_pct3, Lars_factor3 = \
        self.strategy4_sklearn(test_start_date, end_date, period, model_type='Lars',
                               train_days=train_days3, pre_days=pre_days3, corr_rate=corr_rate3, corr_period=corr_period3)

    corr_rate4,corr_period4,train_days4,pre_days4 = 0.55, 24, 24, 6 #choice1
    corr_rate5,corr_period5,train_days5,pre_days5 = 0.5, 12, 24, 6 #choice2

    Bayes_top_ind1, Bayes_bottom_ind1, Bayes_test_result1, Bayes_all_net_value1, Bayes_all_pct1, Bayes_factor1 = \
        self.strategy4_sklearn(test_start_date, end_date, period, model_type='Bayes',
                               train_days=train_days4, pre_days=pre_days4, corr_rate=corr_rate4, corr_period=corr_period4)
    Bayes_top_ind2, Bayes_bottom_ind2, Bayes_test_result2, Bayes_all_net_value2, Bayes_all_pct2, Bayes_factor2 = \
        self.strategy4_sklearn(test_start_date, end_date, period, model_type='Bayes',
                               train_days=train_days5, pre_days=pre_days5, corr_rate=corr_rate5, corr_period=corr_period5)

# 激进策略：
#Lars_factor1,Lars_factor2                    Lars_test_result1, Lars_test_result2
#Ridge_factor3,Bayes_factor2             Ridge_test_result3, Bayes_test_result2

# 稳健策略：
#linear_factor1,Ridge_factor1,Lars_factor1,linear_factor2,Ridge_factor2,Lars_factor2
#linear_factor3,Ridge_factor3,Lars_factor3,Bayes_factor1, Bayes_factor2


################################ 非线性模型的寻优寻优 ########################################
# 1、Logistics模型寻优
def Logistics_find_best():
    corr_list = [0.5,0.55,0.6,0.65,0.7]
    corr_period_list = [6,12,24]
    train_days_list = [3,6,12,24]
    pre_days_list = [1,3,4,6]
    save_path = 'E:/FactorTest/model/Logistics/'
    result_dict = {}
    for corr_rate in corr_list:
        for corr_period in corr_period_list:
            for train_days in train_days_list:
                for pre_days in pre_days_list:
                    name = 'corr' + str(corr_rate) + 'period' + str(corr_period) + 'train_days' + str(train_days) + 'pre_days' + str(pre_days)
                    print(name)
                    logistic_top_ind, logistic_bottom_ind, logistic_test_result, logistic_all_net_value, logistic_all_pct, logistic_factor = \
                        self.strategy4_sklearn(test_start_date,end_date,period,model_type = 'logistic',
                        train_days = train_days, pre_days = pre_days, corr_rate = corr_rate , corr_period = corr_period)

                    flag = 0
                    if logistic_test_result.loc['excess_maxdown', 'all'] < -0.15:
                        print(logistic_test_result)
                        print(name + ' 回撤不行')
                        flag = 1
                        continue
                    elif -logistic_test_result.loc['excess_return', 'all'] / logistic_test_result.loc['excess_maxdown', 'all'] < 1.1:
                        print(logistic_test_result)
                        print(-logistic_test_result.loc['excess_return', 'all'] / logistic_test_result.loc[
                            'excess_maxdown', 'all'])
                        print(name + ' 收益回撤比不行')
                        flag = 1
                        continue

                    for dict_name in result_dict:
                        other_result = result_dict[dict_name]
                        if (logistic_test_result.loc[['excess_return','excess_sharpe','excess_winrate','excess_maxdown','excess_wlratio','ls_return','ls_sharpe','ls_winrate','ls_wlratio','ls_maxdown'],'all'] < \
                            other_result.loc[['excess_return','excess_sharpe','excess_winrate','excess_maxdown','excess_wlratio','ls_return','ls_sharpe','ls_winrate','ls_wlratio','ls_maxdown'],'all']).sum() >= 8:
                            print(pd.concat([logistic_test_result['all'],other_result['all']],axis=1))
                            print(name + ' 结果不行')
                            flag = 1
                            break
                        elif logistic_test_result.loc['excess_return','all'] < other_result.loc['excess_return','all'] / 2:
                            print(pd.concat([logistic_test_result['all'],other_result['all']],axis=1))
                            print(name + ' 超额收益不行')
                            flag = 1
                            break

                    if flag == 0:
                        logistic_test_result.to_pickle(save_path + name +'.pkl')
                        result_dict[name] = logistic_test_result

    # 内部还要再清洗一遍；就是因为过早放入，而导致没有被剔除的样本剔除
    useful_result_dict = {}
    factor_list = [x[:-4] for x in os.listdir(save_path)]
    for factor_name in factor_list:
        flag = 0
        factor_result = pd.read_pickle(save_path + factor_name + '.pkl')

        for dict_name in list(useful_result_dict.keys()):
            other_result = useful_result_dict[dict_name]
            if (factor_result.loc[['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio', 'ls_return',
                     'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all'] < \
                other_result.loc[['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio', 'ls_return',
                     'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all']).sum() >= 8:
                print(pd.concat([factor_result['all'], other_result['all']], axis=1))
                print(factor_name + ' 结果不行')
                flag = 1
                break

            elif (factor_result.loc[['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio', 'ls_return',
                     'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all'] > \
                other_result.loc[
                    ['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio', 'ls_return',
                     'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all']).sum() >= 8:
                print(pd.concat([factor_result['all'], other_result['all']], axis=1))
                print(dict_name + ' 结果不行')
                useful_result_dict.pop(dict_name, None)
                #os.remove(save_path + dict_name + '.pkl')

        if flag == 0:
            useful_result_dict[factor_name] = factor_result
# 2、RF模型寻优
def RF_find_best():
    corr_list = [0.5, 0.55, 0.6]
    corr_list = [0.65, 0.7]
    corr_period_list = [6, 12, 24]
    train_days_list = [3, 6, 12, 24]
    pre_days_list = [1, 3, 4, 6]
    save_path = 'E:/FactorTest/model/RF/'
    result_dict = {}
    for corr_rate in corr_list:
        for corr_period in corr_period_list:
            for train_days in train_days_list:
                for pre_days in pre_days_list:
                    name = 'corr' + str(corr_rate) + 'period' + str(corr_period) + 'train_days' + str(train_days) + 'pre_days' + str(pre_days)
                    print(name)
                    RF_top_ind, RF_bottom_ind, RF_test_result, RF_all_net_value, RF_all_pct, RF_factor = \
                        self.strategy4_sklearn(test_start_date, end_date, period, model_type='RandomForest',train_days=train_days, pre_days=pre_days, corr_rate=corr_rate,corr_period=corr_period)
                    flag = 0
                    if RF_test_result.loc['excess_maxdown', 'all'] < -0.15:
                        print(RF_test_result)
                        print(name + ' 回撤不行')
                        flag = 1
                        continue
                    elif -RF_test_result.loc['excess_return', 'all'] / RF_test_result.loc['excess_maxdown', 'all'] < 1.1:
                        print(RF_test_result)
                        print(-RF_test_result.loc['excess_return', 'all'] / RF_test_result.loc['excess_maxdown', 'all'])
                        print(name + ' 收益回撤比不行')
                        flag = 1
                        continue

                    for dict_name in result_dict:
                        other_result = result_dict[dict_name]
                        if (RF_test_result.loc[['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio',
                                 'ls_return', 'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all'] < \
                            other_result.loc[['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio',
                                 'ls_return', 'ls_sharpe', 'ls_winrate', 'ls_wlratio','ls_maxdown'], 'all']).sum() >= 8:
                            print(pd.concat([RF_test_result['all'], other_result['all']], axis=1))
                            print(name + ' 结果不行')
                            flag = 1
                            break
                        elif RF_test_result.loc['excess_return', 'all'] < other_result.loc['excess_return', 'all'] / 2:
                            print(pd.concat([RF_test_result['all'], other_result['all']], axis=1))
                            print(name + ' 超额收益不行')
                            flag = 1
                            break

                    if flag == 0:
                        RF_test_result.to_pickle(save_path + name + '.pkl')
                        result_dict[name] = RF_test_result

    # 内部还要再清洗一遍；就是因为过早放入，而导致没有被剔除的样本剔除
    useful_result_dict = {}
    factor_list = [x[:-4] for x in os.listdir(save_path)]
    for factor_name in factor_list:
        flag = 0
        factor_result = pd.read_pickle(save_path + factor_name + '.pkl')

        for dict_name in list(useful_result_dict.keys()):
            other_result = useful_result_dict[dict_name]
            if (factor_result.loc[
                    ['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio',
                     'ls_return',
                     'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all'] < \
                other_result.loc[
                    ['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio',
                     'ls_return',
                     'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all']).sum() >= 8:
                print(pd.concat([factor_result['all'], other_result['all']], axis=1))
                print(factor_name + ' 结果不行')
                flag = 1
                break

            elif (factor_result.loc[
                      ['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio',
                       'ls_return',
                       'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all'] > \
                  other_result.loc[
                      ['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio',
                       'ls_return',
                       'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all']).sum() >= 8:
                print(pd.concat([factor_result['all'], other_result['all']], axis=1))
                print(dict_name + ' 结果不行')
                useful_result_dict.pop(dict_name, None)
                # os.remove(save_path + dict_name + '.pkl')

        if flag == 0:
            useful_result_dict[factor_name] = factor_result
# 3、GBRT模型寻优
def GBRT_find_best():
    corr_list = [0.5, 0.55, 0.6,0.65, 0.7]
    corr_period_list = [6, 12, 24]
    train_days_list = [3, 6, 12, 24]
    pre_days_list = [1, 3, 4, 6]
    save_path = 'E:/FactorTest/model/GBRT/'
    result_dict = {}
    for corr_rate in corr_list:
        for corr_period in corr_period_list:
            for train_days in train_days_list:
                for pre_days in pre_days_list:
                    name = 'corr' + str(corr_rate) + 'period' + str(corr_period) + 'train_days' + str(
                        train_days) + 'pre_days' + str(pre_days)
                    print(name)
                    GBRT_top_ind, GBRT_bottom_ind, GBRT_test_result, GBRT_all_net_value, GBRT_all_pct, GBRT_factor = \
                        self.strategy4_sklearn(test_start_date, end_date, period, model_type='GBRT',
                                               train_days=train_days, pre_days=pre_days, corr_rate=corr_rate,corr_period=corr_period)
                    flag = 0
                    if GBRT_test_result.loc['excess_maxdown', 'all'] < -0.15:
                        print(GBRT_test_result)
                        print(name + ' 回撤不行')
                        flag = 1
                        continue
                    elif -GBRT_test_result.loc['excess_return', 'all'] / GBRT_test_result.loc['excess_maxdown', 'all'] < 1.1:
                        print(GBRT_test_result)
                        print(-GBRT_test_result.loc['excess_return', 'all'] / GBRT_test_result.loc['excess_maxdown', 'all'])
                        print(name + ' 收益回撤比不行')
                        flag = 1
                        continue

                    for dict_name in result_dict:
                        other_result = result_dict[dict_name]
                        if (GBRT_test_result.loc[
                                ['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio',
                                 'ls_return', 'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all'] < \
                            other_result.loc[
                                ['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio',
                                 'ls_return', 'ls_sharpe', 'ls_winrate', 'ls_wlratio','ls_maxdown'], 'all']).sum() >= 8:
                            print(pd.concat([GBRT_test_result['all'], other_result['all']], axis=1))
                            print(name + ' 结果不行')
                            flag = 1
                            break
                        elif GBRT_test_result.loc['excess_return', 'all'] < other_result.loc['excess_return', 'all'] / 2:
                            print(pd.concat([GBRT_test_result['all'], other_result['all']], axis=1))
                            print(name + ' 超额收益不行')
                            flag = 1
                            break

                    if flag == 0:
                        GBRT_test_result.to_pickle(save_path + name + '.pkl')
                        result_dict[name] = GBRT_test_result

    # 内部还要再清洗一遍；就是因为过早放入，而导致没有被剔除的样本剔除
    useful_result_dict = {}
    factor_list = [x[:-4] for x in os.listdir(save_path)]
    for factor_name in factor_list:
        flag = 0
        factor_result = pd.read_pickle(save_path + factor_name + '.pkl')

        for dict_name in list(useful_result_dict.keys()):
            other_result = useful_result_dict[dict_name]
            if (factor_result.loc[
                    ['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio',
                     'ls_return',
                     'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all'] < \
                other_result.loc[
                    ['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio',
                     'ls_return',
                     'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all']).sum() >= 8:
                print(pd.concat([factor_result['all'], other_result['all']], axis=1))
                print(factor_name + ' 结果不行')
                flag = 1
                break

            elif (factor_result.loc[
                      ['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio',
                       'ls_return',
                       'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all'] > \
                  other_result.loc[
                      ['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio',
                       'ls_return',
                       'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all']).sum() >= 8:
                print(pd.concat([factor_result['all'], other_result['all']], axis=1))
                print(dict_name + ' 结果不行')
                useful_result_dict.pop(dict_name, None)
                # os.remove(save_path + dict_name + '.pkl')

        if flag == 0:
            useful_result_dict[factor_name] = factor_result


    len(useful_result_dict)
# 4、LGB模型寻优
def LGB_find_best():
    corr_list = [0.5, 0.55, 0.6]
    corr_list = [0.65, 0.7]
    corr_period_list = [6, 12, 24]
    train_days_list = [3, 6, 12, 24]
    pre_days_list = [1, 3, 4, 6]
    save_path = 'E:/FactorTest/model/LGB/'
    result_dict = {}
    for corr_rate in corr_list:
        for corr_period in corr_period_list:
            for train_days in train_days_list:
                for pre_days in pre_days_list:
                    name = 'corr' + str(corr_rate) + 'period' + str(corr_period) + 'train_days' + str(
                        train_days) + 'pre_days' + str(pre_days)
                    print(name)
                    LGB_top_ind, LGB_bottom_ind, LGB_test_result, LGB_all_net_value, LGB_all_pct, LGB_factor = \
                        self.strategy4_sklearn(test_start_date, end_date, period, model_type='LGB',
                                               train_days=train_days, pre_days=pre_days, corr_rate=corr_rate,corr_period=corr_period)
                    flag = 0
                    if LGB_test_result.loc['excess_maxdown', 'all'] < -0.15:
                        print(LGB_test_result)
                        print(name + ' 回撤不行')
                        flag = 1
                        continue
                    elif -LGB_test_result.loc['excess_return', 'all'] / LGB_test_result.loc['excess_maxdown', 'all'] < 1.1:
                        print(LGB_test_result)
                        print(-LGB_test_result.loc['excess_return', 'all'] / LGB_test_result.loc['excess_maxdown', 'all'])
                        print(name + ' 收益回撤比不行')
                        flag = 1
                        continue

                    for dict_name in result_dict:
                        other_result = result_dict[dict_name]
                        if (LGB_test_result.loc[
                                ['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio',
                                 'ls_return', 'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all'] < \
                            other_result.loc[
                                ['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio',
                                 'ls_return', 'ls_sharpe', 'ls_winrate', 'ls_wlratio','ls_maxdown'], 'all']).sum() >= 8:
                            print(pd.concat([LGB_test_result['all'], other_result['all']], axis=1))
                            print(name + ' 结果不行')
                            flag = 1
                            break
                        elif LGB_test_result.loc['excess_return', 'all'] < other_result.loc['excess_return', 'all'] / 2:
                            print(pd.concat([LGB_test_result['all'], other_result['all']], axis=1))
                            print(name + ' 超额收益不行')
                            flag = 1
                            break

                    if flag == 0:
                        LGB_test_result.to_pickle(save_path + name + '.pkl')
                        result_dict[name] = LGB_test_result

    # 内部还要再清洗一遍；就是因为过早放入，而导致没有被剔除的样本剔除
    useful_result_dict = {}
    factor_list = [x[:-4] for x in os.listdir(save_path)]
    for factor_name in factor_list:
        flag = 0
        factor_result = pd.read_pickle(save_path + factor_name + '.pkl')

        for dict_name in list(useful_result_dict.keys()):
            other_result = useful_result_dict[dict_name]
            if (factor_result.loc[
                    ['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio',
                     'ls_return',
                     'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all'] < \
                other_result.loc[
                    ['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio',
                     'ls_return',
                     'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all']).sum() >= 8:
                print(pd.concat([factor_result['all'], other_result['all']], axis=1))
                print(factor_name + ' 结果不行')
                flag = 1
                break

            elif (factor_result.loc[
                      ['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio',
                       'ls_return',
                       'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all'] > \
                  other_result.loc[
                      ['excess_return', 'excess_sharpe', 'excess_winrate', 'excess_maxdown', 'excess_wlratio',
                       'ls_return',
                       'ls_sharpe', 'ls_winrate', 'ls_wlratio', 'ls_maxdown'], 'all']).sum() >= 8:
                print(pd.concat([factor_result['all'], other_result['all']], axis=1))
                print(dict_name + ' 结果不行')
                useful_result_dict.pop(dict_name, None)
                # os.remove(save_path + dict_name + '.pkl')

        if flag == 0:
            useful_result_dict[factor_name] = factor_result

def find_nonlinear_result():
    corr_rate1,corr_period1,train_days1,pre_days1 = 0.5, 24, 12, 3 # choice1
    corr_rate2,corr_period2,train_days2,pre_days2 = 0.6, 12, 24, 4 # choice1
    logistic_top_ind1, logistic_bottom_ind1, logistic_test_result1, logistic_all_net_value1, logistic_all_pct1, logistic_factor1 = \
                            self.strategy4_sklearn(test_start_date,end_date,period,model_type = 'logistic',
                            train_days = train_days1, pre_days = pre_days1, corr_rate = corr_rate1 , corr_period = corr_period1)
    logistic_top_ind2, logistic_bottom_ind2, logistic_test_result2, logistic_all_net_value2, logistic_all_pct2, logistic_factor2 = \
                            self.strategy4_sklearn(test_start_date,end_date,period,model_type = 'logistic',
                            train_days = train_days2, pre_days = pre_days2, corr_rate = corr_rate2 , corr_period = corr_period2)

    corr_rate,corr_period,train_days,pre_days = 0.5, 12, 24, 4
    RF_top_ind, RF_bottom_ind, RF_test_result, RF_all_net_value, RF_all_pct, RF_factor = \
                            self.strategy4_sklearn(test_start_date, end_date, period, model_type='RandomForest',train_days=train_days, pre_days=pre_days, corr_rate=corr_rate,corr_period=corr_period)


    corr_rate,corr_period,train_days,pre_days = 0.5, 12, 12, 6
    GBRT_top_ind, GBRT_bottom_ind, GBRT_test_result, GBRT_all_net_value, GBRT_all_pct, GBRT_factor = \
                            self.strategy4_sklearn(test_start_date, end_date, period, model_type='GBRT',
                                                   train_days=train_days, pre_days=pre_days, corr_rate=corr_rate,corr_period=corr_period)


    corr_rate,corr_period,train_days,pre_days = 0.55, 24, 6, 6
    LGB_top_ind, LGB_bottom_ind, LGB_test_result, LGB_all_net_value, LGB_all_pct, LGB_factor = \
                            self.strategy4_sklearn(test_start_date, end_date, period, model_type='LGB',
                                                   train_days=train_days, pre_days=pre_days, corr_rate=corr_rate,corr_period=corr_period)

################################ 对最后的结果进行最终集成 #############################################
def get_end_result(start_date, end_date):
    ind, period = 'SW1', 'M'
    self = StrategyTest(test_start_date=start_date, test_end_date=end_date, fee=0.001)
    test_start_date = 20151231
    #Lar1,Lar2模型
    corr_rate1,corr_period1,train_days1,pre_days1 = 0.55, 12, 12, 3 #choice1
    corr_rate2,corr_period2,train_days2,pre_days2 = 0.5, 12, 12, 4 #choice2（和choice1接近，二者可以选一）
    corr_rate3, corr_period3, train_days3, pre_days3 = 0.5, 12, 24, 6 #choice3
    '''
    Lars_top_ind1, Lars_bottom_ind1, Lars_test_result1, Lars_all_net_value1, Lars_all_pct1, Lars_factor1 = \
            self.strategy4_sklearn(test_start_date, end_date, period, model_type='Lars',
                                   train_days=train_days1, pre_days=pre_days1, corr_rate=corr_rate1, corr_period=corr_period1)
    Lars_top_ind2, Lars_bottom_ind2, Lars_test_result2, Lars_all_net_value2, Lars_all_pct2, Lars_factor2 = \
        self.strategy4_sklearn(test_start_date, end_date, period, model_type='Lars',
                               train_days=train_days2, pre_days=pre_days2, corr_rate=corr_rate2, corr_period=corr_period2)
    # 选择方式：两种，第一种是两个加权，第二种是直接使用第一个
    #new_factor1 = Lars_factor1.copy()
    # Bayes2,Ridge3模型
    Bayes_top_ind2, Bayes_bottom_ind2, Bayes_test_result2, Bayes_all_net_value2, Bayes_all_pct2, Bayes_factor2 = \
            self.strategy4_sklearn(test_start_date, end_date, period, model_type='Bayes',
                                   train_days=train_days3, pre_days=pre_days3, corr_rate=corr_rate3, corr_period=corr_period3)
    Ridge_top_ind3, Ridge_bottom_ind3, Ridge_test_result3, Ridge_all_net_value3, Ridge_all_pct3, Ridge_factor3 = \
            self.strategy4_sklearn(test_start_date, end_date, period, model_type='Ridge',
                                   train_days=train_days3, pre_days=pre_days3, corr_rate=corr_rate3, corr_period=corr_period3)
    # Logisitic1, Logistic2模型
    corr_rate4,corr_period4,train_days4,pre_days4 = 0.5, 24, 12, 3 # choice1
    corr_rate5,corr_period5,train_days5,pre_days5 = 0.6, 12, 24, 4 # choice1
    logistic_top_ind1, logistic_bottom_ind1, logistic_test_result1, logistic_all_net_value1, logistic_all_pct1, logistic_factor1 = \
                            self.strategy4_sklearn(test_start_date,end_date,period,model_type = 'logistic',
                            train_days = train_days4, pre_days = pre_days4, corr_rate = corr_rate4 , corr_period = corr_period4)
    logistic_top_ind2, logistic_bottom_ind2, logistic_test_result2, logistic_all_net_value2, logistic_all_pct2, logistic_factor2 = \
                            self.strategy4_sklearn(test_start_date,end_date,period,model_type = 'logistic',
                            train_days = train_days5, pre_days = pre_days5, corr_rate = corr_rate5 , corr_period = corr_period5)
    # RF模型
    corr_rate6,corr_period6,train_days6,pre_days6 = 0.5, 12, 24, 4
    RF_top_ind, RF_bottom_ind, RF_test_result, RF_all_net_value, RF_all_pct, RF_factor = \
                            self.strategy4_sklearn(test_start_date, end_date, period, model_type='RandomForest',
                            train_days=train_days6, pre_days=pre_days6, corr_rate=corr_rate6,corr_period=corr_period6)
    # GBRT模型
    corr_rate7,corr_period7,train_days7,pre_days7 = 0.5, 12, 12, 6
    Bagging_top_ind, Bagging_bottom_ind,Bagging_test_result, Bagging_all_net_value, Bagging_all_pct, Bagging_factor = \
        self.strategy4_sklearn(test_start_date, end_date, period, model_type='Bagging',
                            train_days=train_days7, pre_days=pre_days7, corr_rate=corr_rate7, corr_period=corr_period7)
    '''

    Lars_factor1 = self.strategy4_sklearn(test_start_date, end_date, period, model_type='Lars',
                   train_days=train_days1, pre_days=pre_days1, corr_rate=corr_rate1, corr_period=corr_period1,if_test = False)
    Lars_factor2 = self.strategy4_sklearn(test_start_date, end_date, period, model_type='Lars',
                   train_days=train_days2, pre_days=pre_days2, corr_rate=corr_rate2, corr_period=corr_period2,if_test = False)
    Bayes_factor2 = self.strategy4_sklearn(test_start_date, end_date, period, model_type='Bayes',
                    train_days=train_days3, pre_days=pre_days3, corr_rate=corr_rate3, corr_period=corr_period3,if_test = False)
    Ridge_factor3 = self.strategy4_sklearn(test_start_date, end_date, period, model_type='Ridge',
                    train_days=train_days3, pre_days=pre_days3, corr_rate=corr_rate3, corr_period=corr_period3,if_test = False)

    corr_rate4,corr_period4,train_days4,pre_days4 = 0.5, 24, 12, 3 # choice1
    corr_rate5,corr_period5,train_days5,pre_days5 = 0.6, 12, 24, 4 # choice1
    logistic_factor1 = self.strategy4_sklearn(test_start_date,end_date,period,model_type = 'logistic',
                    train_days = train_days4, pre_days = pre_days4, corr_rate = corr_rate4 , corr_period = corr_period4,if_test = False)
    logistic_factor2 = self.strategy4_sklearn(test_start_date,end_date,period,model_type = 'logistic',
                    train_days = train_days5, pre_days = pre_days5, corr_rate = corr_rate5 , corr_period = corr_period5,if_test = False)
    # RF模型
    corr_rate6,corr_period6,train_days6,pre_days6 = 0.5, 12, 24, 4
    RF_factor = self.strategy4_sklearn(test_start_date, end_date, period, model_type='RandomForest',
                    train_days=train_days6, pre_days=pre_days6, corr_rate=corr_rate6,corr_period=corr_period6,if_test = False)
    # GBRT模型
    corr_rate7,corr_period7,train_days7,pre_days7 = 0.5, 12, 12, 6
    Bagging_factor = self.strategy4_sklearn(test_start_date, end_date, period, model_type='Bagging',
                    train_days=train_days7, pre_days=pre_days7, corr_rate=corr_rate7, corr_period=corr_period7,if_test = False)


    new_factor1 = (deal_data(Lars_factor1) + deal_data(Lars_factor2)) / 2   # IC加权？
    new_factor2 = (deal_data(Bayes_factor2) + deal_data(Ridge_factor3)) / 2
    new_factor3 = logistic_factor1.copy()
    new_factor4 = logistic_factor2.copy()
    new_factor5 = RF_factor.copy()
    new_factor6 = Bagging_factor.copy()

    # 最后就是对模型的集成，形成最终的结果
    new_factor = ((deal_data(new_factor1) + deal_data(new_factor2)) + deal_data(new_factor3) +
                  deal_data(new_factor4) + deal_data(new_factor5) + deal_data(new_factor6))/6

    top_ind, bottom_ind, test_result, all_net_value, all_pct = self.concequence_result(new_factor, group=5,fee=self.fee)

    return new_factor, top_ind, bottom_ind, test_result, all_net_value, all_pct

def get_risk_factor(top_ind,read_path = 'E:/FactorTest/risk_factor/'):
    factor_list = [x[:-4] for x in os.listdir(read_path)]
    risk_dict = {}
    for factor in factor_list:
        risk_dict[factor] = pd.read_pickle(read_path + factor + '.pkl')

    risk_factor = risk_dict['risk_amtfactor'] | risk_dict['risk_turnfactor'] | risk_dict['risk_corr']
    risk_factor[risk_factor.rolling(20).sum() > 1] = 0
    risk_factor = risk_factor.rolling(20).max().fillna(0)

    self = FactorTest(test_start_date=20151231, test_end_date=20221031, ind='SW1', day=20, fee=0.001)
    top_ind1 = top_ind.reindex(self.test_date_list).ffill().dropna(how='all')
    top_ind1 = (top_ind1 & (risk_factor == False)).loc[top_ind1.index, top_ind.columns]

    new_test_result1, new_net_value1 = self.strategy_test(top_ind1)

    return risk_dict,top_ind1,new_test_result1, new_net_value1
# 因子3步：
# 第一步：更新数据集到月末
# 第二步：更新月频因子值和月频风险因子值
# 第三步：运行模型结果
start_date, end_date = 20140101, 20221111
new_factor, top_ind, bottom_ind, test_result, all_net_value, all_pct = get_end_result(start_date, end_date)


risk_dict,risk_top_ind,risk_test_result, risk_net_value = get_risk_factor(top_ind,read_path = 'E:/FactorTest/risk_factor/')

#top_ind.loc[20210730][top_ind.loc[20210730]==True]

'''
ind = 'sw_all1'
ind_name = get_ind_con(ind[:-1],int(ind[-1]))
ind_name = pd.Series(ind_name)
top_ind_name = top_ind.copy()
top_ind_name.columns = pd.Series(top_ind_name.columns).apply(lambda x:ind_name.loc[x])

top_ind_name.loc[20211231][top_ind_name.loc[20211231]==True].index
top_ind_name.loc[20220128][top_ind_name.loc[20220128]==True].index
top_ind_name.loc[20220228][top_ind_name.loc[20220228]==True].index
top_ind_name.loc[20220331][top_ind_name.loc[20220331]==True].index
top_ind_name.loc[20220429][top_ind_name.loc[20220429]==True].index
top_ind_name.loc[20220531][top_ind_name.loc[20220531]==True].index
top_ind_name.loc[20220630][top_ind_name.loc[20220630]==True].index
top_ind_name.loc[20220729][top_ind_name.loc[20220729]==True].index
top_ind_name.loc[20220831][top_ind_name.loc[20220831]==True].index
top_ind_name.loc[20220930][top_ind_name.loc[20220930]==True].index
top_ind_name.loc[20221031][top_ind_name.loc[20221031]==True].index
'''


'''
top_ind.loc[20221031][top_ind.loc[20221031]==True]
top_ind.loc[20220930][top_ind.loc[20220930]==True]
# 10月选的6个行业
801080.SI    True
801110.SI    True
801210.SI    True
801730.SI    True
801790.SI    True
801980.SI    True
# 9月选的6个行业
801080.SI    True
801110.SI    True
801150.SI    True
801710.SI    True
801790.SI    True
801880.SI    True
'''
