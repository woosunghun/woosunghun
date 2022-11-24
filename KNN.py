import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

import os
import pickle
import gzip

from fancyimpute import KNN, MatrixFactorization

#Data loading and Nan imputation by KNN 
#년도별로 Clustering 수행, NAN값은 해당 data가 속한 군집중에서 거리가 가장 가까운 값으로 대체한다.
all_data = pd.DataFrame()
for i in ['2018', '2019', '2020', '2021']:
    ok = pd.read_pickle(f'./data/OK_{i}.pkl')
    ng = pd.read_pickle(f'./data/NG_{i}.pkl')
    ok['OK/NG'] = 0
    ng['OK/NG'] = 1
    total = pd.concat([ok,ng])
    # categorical variable
    total_cate = total[total.columns[[i for i in range(-1,34)]]]
    # numerical variable
    total_num = total[total.columns[34:-1]]
    
    #numerical variable의 NAN 값에대하여 KNN imputation 시행
    total_num_filled_KNN = pd.DataFrame(KNN(k=5).fit_transform(total_num))
    total_num_filled_KNN.columns = total_num.columns
    total_filled_KNN = pd.concat([total_cate.reset_index(drop=True), total_num_filled_KNN.reset_index(drop=True)], axis = 1)
    
    # data merge
    all_data = pd.concat([all_data, total_filled_KNN])
    
all_data = all_data.reset_index(drop=True)

# CMP_NM 과 CMP_NM1 컬럼 결합
temp = all_data[['CMP_NM','CMP_NM1']].fillna("")
all_data['CMP_NM'] = temp['CMP_NM']+""+temp['CMP_NM1']
all_data = all_data.drop(['CMP_NM1'],axis=1)

dd = all_data.groupby(['BZ_TYP','OK/NG']).size().unstack().fillna(0)
dd = dd.reset_index()
dd.columns = ['BZ_TYP', 'OK', 'NG']
dd

### BZ_TYP이 M인 회사만 분석
all_data = all_data[all_data.BZ_TYP == 'M']
all_data = all_data.reset_index(drop = True)

#### 컬럼 별 NA 비율
all_data[['PBCO_GB', 'HDOF_BR_GB', 'FR_IVST_CORP_YN', 'VENT_YN',
       '결산년월', '유동자산', '매출채권', '비유동자산', '유형자산', '자산총계', '유동부채',
       '비유동부채', '부  채  총  계', '자본금', '이익잉여금(결손금）', '자본총계', '매출액', '판매비와관리비',
       '영업이익（손실）', '법인세비용차감전순손익', '법인세비용', '당기순이익(손실)', '기업순이익률(%)',
       '유보액/총자산(%)', '유보액/납입자본(%)', '매출액총이익률(%)', '매출액영업이익률(%)', '매출액순이익률(%)',
       '수지비율(%)', '경상수지비율', '영업비율(%)', '금융비용대매출액비율(%', '금융비용대부채비율(%)',
       '금융비용대총비용비율(%', '부채비율(%)', '차입금의존도(%)', '자기자본비율(%)', '순운전자본비율(%)',
       '유동부채비율(%)', '비유동부채비율(%)', '부채총계대 매출액(%)', '총자본회전율(회)', '재고자산회전율(회)',
       '매출채권회전율(회)', '매입채무회전율(회)', '미수금', '매출원가', '무형자산', '재고자산']].isna().mean()*100

for n_n in ['PBCO_GB', 'HDOF_BR_GB', 'FR_IVST_CORP_YN', 'VENT_YN']:
    print(n_n)
    print(all_data[n_n].unique())
    
X = all_data[['PBCO_GB', 'HDOF_BR_GB', 'FR_IVST_CORP_YN', 'VENT_YN',
       '유동자산', '매출채권', '비유동자산', '유형자산', '자산총계', '유동부채',
       '비유동부채', '부  채  총  계', '자본금', '이익잉여금(결손금）', '자본총계', '매출액', '판매비와관리비',
       '영업이익（손실）', '법인세비용차감전순손익', '법인세비용', '당기순이익(손실)', '기업순이익률(%)',
       '유보액/총자산(%)', '유보액/납입자본(%)', '매출액총이익률(%)', '매출액영업이익률(%)', '매출액순이익률(%)',
       '수지비율(%)', '경상수지비율', '영업비율(%)', '금융비용대매출액비율(%', '금융비용대부채비율(%)',
       '금융비용대총비용비율(%', '부채비율(%)', '차입금의존도(%)', '자기자본비율(%)', '순운전자본비율(%)',
       '유동부채비율(%)', '비유동부채비율(%)', '부채총계대 매출액(%)', '총자본회전율(회)', '재고자산회전율(회)',
       '매출채권회전율(회)', '매입채무회전율(회)', '미수금', '매출원가', '무형자산', '재고자산']]

### 카테고리형 변수 인코딩
X['FR_IVST_CORP_YN'] = pd.get_dummies(X.FR_IVST_CORP_YN, prefix='FR_IVST_CORP_YN' , drop_first=True)
X['VENT_YN'] = pd.get_dummies(X.VENT_YN, prefix='VENT_YN' , drop_first=True)

y = all_data['OK/NG']

# XGBoost
f1_scores = []
precision_scores = []
skf = StratifiedKFold(n_splits=5)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]
    
    #clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf = XGBClassifier(n_estimators=200, use_label_encoder=False)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print(classification_report(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred))
