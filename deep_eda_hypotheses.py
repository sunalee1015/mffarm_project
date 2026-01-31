import pandas as pd
import numpy as np
import os
import sys

# 한글 폰트 설정 (필요 시 주석 해제)
# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Malgun Gothic'

def log(msg):
    print(f"\n>>> {msg}")

# 데이터 로드
file_path = r'C:\Data\ICB6\workspace\mffarm04\preprocessed_data_20260131.csv'
df = pd.read_csv(file_path, encoding='utf-8-sig')

# 전처리 및 공통 변수 생성
df['주문일'] = pd.to_datetime(df['주문일'])
df['월'] = df['주문일'].dt.to_period('M')

def clean_price(val):
    if pd.isna(val): return 0
    if isinstance(val, (int, float)): return val
    try: return float(str(val).replace(',', ''))
    except: return 0

df['판매단가'] = df['판매단가'].apply(clean_price)
df['공급단가'] = df['공급단가'].apply(clean_price)
df['결제금액'] = df['결제금액'].apply(clean_price)
df['주문수량'] = pd.to_numeric(df['주문수량'], errors='coerce').fillna(0)

# 수수료 가정 (10%)
fee_rate = 0.1
df['순이익'] = df['결제금액'] - df['공급단가'] - (df['결제금액'] * fee_rate)
df['순이익률'] = (df['순이익'] / df['결제금액']).replace([np.inf, -np.inf], 0).fillna(0)

# 가설 1: 경기도 매출 셀러 분석
def hypothesis_1(df):
    log("[가설 1] 경기도권 판매의 셀러 영향력 분석")
    gg_orders = df[df['광역지역(정식)'] == '경기도']
    
    # 경기도 매출 상위 셀러
    gg_seller_rev = gg_orders.groupby('셀러명')['결제금액'].sum().sort_values(ascending=False)
    total_gg_rev = gg_seller_rev.sum()
    top_seller_pct = (gg_seller_rev.head(5).sum() / total_gg_rev)
    
    print(f"- 경기도 전체 매출: {total_gg_rev:,.0f}원")
    print(f"- 경기도 내 상위 5개 셀러 매출 비중: {top_seller_pct:.2%}")
    print("- 경기도 매출 Top 3 셀러:")
    for s, v in gg_seller_rev.head(3).items():
        print(f"  {s}: {v:,.0f}원")

# 가설 2: 이벤트 상품의 구매량 증대 효과
def hypothesis_2(df):
    log("[가설 2] 이벤트 상품 구매량 증가 효과 분석")
    event_keywords = '1\+1|증정|추가발송|이벤트|특가|한정|폭탄'
    df['is_event_item'] = df['상품명'].str.contains(event_keywords).fillna(False) | (df['이벤트 여부'] == 'Y')
    
    res = df.groupby('is_event_item').agg({
        '주문번호': 'nunique',
        '주문수량': 'mean',
        '결제금액': 'mean'
    }).rename(columns={'주문번호': '주문건수', '주문수량': '평균주문수량', '결제금액': '평균결제액'})
    print(res)

# 가설 3: 이벤트 상품의 수익성 분석
def hypothesis_3(df):
    log("[가설 3] 이벤트 상품 vs 일반 상품 수익성 비교")
    res = df.groupby('is_event_item').agg({
        '순이익': 'mean',
        '순이익률': 'mean'
    })
    print(res)
    
    # 볼륨형 vs 수익형 구분 (이벤트 상품 중)
    event_items = df[df['is_event_item'] == True]
    seller_event_perf = event_items.groupby('셀러명').agg({
        '결제금액': 'sum',
        '순이익률': 'mean'
    })
    print("\n- 매출 상위 이벤트 셀러의 순이익률:")
    print(seller_event_perf.sort_values('결제금액', ascending=False).head(5))

# 가설 4: 선물 목적 상품 선택 분석
def hypothesis_4(df):
    log("[가설 4] 선물 목적 주문의 옵션/단가 선택 특성")
    gift_keywords = '선물|포장|선물세트'
    df['is_gift_item'] = df['상품명'].str.contains(gift_keywords).fillna(False) | (df['선물세트_여부'] == 'Y')
    
    gift_dist = df.groupby('is_gift_item')['판매단가'].describe()
    print("- 선물용 vs 일반용 단가 분포:")
    print(gift_dist[['mean', '50%', 'max']])
    
    # 과수 크기 비중 비교
    gift_size = df[df['is_gift_item']==True]['과수 크기'].value_counts(normalize=True).head(3)
    home_size = df[df['is_gift_item']==False]['과수 크기'].value_counts(normalize=True).head(3)
    print("\n- 선물용 인기 과수 크기:\n", gift_size)
    print("\n- 일반용 인기 과수 크기:\n", home_size)

# 가설 5: 셀러별 재구매율 분석
def hypothesis_5(df):
    log("[가설 5] 셀러별 재구매 고객 비중 분석")
    
    # 셀러별 고객의 주문 횟수
    customer_seller_counts = df.groupby(['셀러명', 'UID'])['주문번호'].nunique().reset_index()
    customer_seller_counts['is_reorder'] = customer_seller_counts['주문번호'] > 1
    
    reorder_rate = customer_seller_counts.groupby('셀러명').agg({
        'UID': 'count',
        'is_reorder': 'sum'
    })
    reorder_rate['재구매율'] = reorder_rate['is_reorder'] / reorder_rate['UID']
    
    print("- 재구매율 상위 셀러 (모수 50명 이상):")
    top_reorder_sellers = reorder_rate[reorder_rate['UID'] >= 50].sort_values('재구매율', ascending=False).head(5)
    print(top_reorder_sellers)

# 가설 6: 셀러 상품 포트폴리오 유형화
def hypothesis_6(df):
    log("[가설 6] 셀러별 상품 구조 유형화")
    # 유형 정의: 프리미엄(고단가), 실속(가정용/저단가), 이벤트(이벤트키워드)
    # 여기서는 간단히 키워드 점유율로 분석
    seller_types = df.groupby('셀러명').agg({
        'is_event_item': 'mean',
        'is_gift_item': 'mean',
        '판매단가': 'mean'
    }).rename(columns={'is_event_item': '이벤트비중', 'is_gift_item': '선물비중', '판매단가': '평균단가'})
    
    print("- 선물 비중이 높은 '프리미엄형' 셀러:")
    print(seller_types.sort_values('선물비중', ascending=False).head(3))
    print("\n- 이벤트 비중이 높은 '프로모션형' 셀러:")
    print(seller_types.sort_values('이벤트비중', ascending=False).head(3))

# 가설 7 & 8: 셀러 유동성 및 매출 추이
def hypothesis_7_8(df):
    log("[가설 7 & 8] 월별 셀러 유동 및 매출 기여도")
    monthly_stats = df.groupby('월').agg({
        '결제금액': 'sum',
        '셀러명': 'nunique',
        '주문번호': 'nunique'
    })
    
    # 신규/이탈 분석을 위해 리스트화
    monthly_sellers = df.groupby('월')['셀러명'].apply(set).to_dict()
    sorted_months = sorted(monthly_sellers.keys())
    
    records = []
    for i in range(len(sorted_months)):
        curr_m = sorted_months[i]
        curr_s = monthly_sellers[curr_m]
        prev_s = monthly_sellers[sorted_months[i-1]] if i > 0 else set()
        
        new_s = curr_s - prev_s
        lost_s = prev_s - curr_s if i > 0 else set()
        retained_s = curr_s & prev_s
        
        records.append({
            '월': curr_m,
            '전체셀러': len(curr_s),
            '신규셀러': len(new_s),
            '이탈셀러': len(lost_s),
            '유지셀러': len(retained_s)
        })
    
    flow_df = pd.DataFrame(records)
    print("- 월별 셀러 유동성 현황:")
    print(flow_df)

if __name__ == "__main__":
    report_path = r'C:\Data\ICB6\workspace\mffarm04\deep_eda_results.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        sys.stdout = f
        print("# 셀러 및 가설 중심 심층 EDA 분석 결과\n")
        hypothesis_1(df)
        hypothesis_2(df)
        hypothesis_3(df)
        hypothesis_4(df)
        hypothesis_5(df)
        hypothesis_6(df)
        hypothesis_7_8(df)
    
    sys.stdout = sys.__stdout__
    print(f"심층 분석 완료! 결과가 다음 경로에 저장되었습니다: {report_path}")
