import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
file_path = r'C:\Data\ICB6\workspace\mffarm04\preprocessed_data_20260131.csv'
df = pd.read_csv(file_path, encoding='utf-8-sig')

# 분석을 위한 기본 전처리
df['주문일'] = pd.to_datetime(df['주문일'])

# 판매단가 숫자형 변환 (쉼표 제거 및 숫자 외 문자 처리)
def clean_price(val):
    if pd.isna(val): return 0
    if isinstance(val, (int, float)): return val
    try:
        # '39,000' 형태 처리
        return float(str(val).replace(',', ''))
    except:
        return 0

df['판매단가'] = df['판매단가'].apply(clean_price)

# 1. 단가 및 옵션 구조에 따른 주문수 분포
def analyze_price_and_option(df):
    print("\n[1] 단가 및 옵션 구조 분석")
    
    # 실제 주문-취소 수량이 0보다 큰 것 기준 (유효 주문)
    active_orders = df[df['주문-취소 수량'] > 0]
    
    # 가격대별 주문수 분포
    price_dist = active_orders.groupby('가격대')['주문번호'].nunique().sort_values(ascending=False)
    print("- 가격대별 유니크 주문수 (유효 주문 기준):")
    for k, v in price_dist.items():
        print(f"  {k}: {v}건")
    
    # 옵션 구조(과수 크기, 무게 구분 등)에 따른 주문수
    option_dist = active_orders.groupby(['과수 크기', '무게 구분'])['주문번호'].nunique().sort_values(ascending=False).head(10)
    print("\n- 주요 옵션 조합별 주문수 Top 10:")
    for i, (idx, val) in enumerate(option_dist.items()):
        print(f"  {idx}: {val}건")

# 2. 키워드 영향 분석
def analyze_keywords(df):
    print("\n[2] 키워드(이벤트/증정/선물/가정용) 영향 분석")
    
    # 키워드 추출
    df['is_gift'] = df['상품명'].str.contains('선물').fillna(False) | (df['선물세트_여부'] == 'Y')
    df['is_home'] = df['상품명'].str.contains('가정용|못난이|파지|실속').fillna(False)
    df['is_event'] = (df['이벤트 여부'] == 'Y') | df['상품명'].str.contains('이벤트|특가|한정').fillna(False)
    df['is_freebie'] = df['상품명'].str.contains('증정|사은품').fillna(False)
    
    keywords = ['is_gift', 'is_home', 'is_event', 'is_freebie']
    
    print(f"{'키워드':<10} | {'주문수':<8} | {'평균수량':<8} | {'취소율':<8}")
    print("-" * 45)
    for kw in keywords:
        sub = df[df[kw] == True]
        total_orders = sub['주문번호'].nunique()
        avg_qnty = sub['주문수량'].mean()
        cancel_rate = sub['취소여부'].apply(lambda x: 1 if x=='Y' else 0).mean()
        print(f"{kw:<10} | {total_orders:<8,d} | {avg_qnty:<8.2f} | {cancel_rate:<8.2%}")

# 3. 취소 발생과 상품 구조의 관계
def analyze_cancellation(df):
    print("\n[3] 취소 발생과 상품 구조 분석")
    
    df['is_cancelled'] = df['취소여부'].apply(lambda x: 1 if x=='Y' else 0)
    
    # 가격대별 취소율
    cancel_by_price = df.groupby('가격대')['is_cancelled'].mean().sort_values(ascending=False)
    print("- 가격대별 취소율:")
    for k, v in cancel_by_price.items():
        print(f"  {k}: {v:.2%}")
    
    # 과수 크기별 취소율
    cancel_by_size = df.groupby('과수 크기')['is_cancelled'].mean().sort_values(ascending=False).head(5)
    print("\n- 취소율 높은 과수 크기 Top 5:")
    for k, v in cancel_by_size.items():
        print(f"  {k}: {v:.2%}")

# 4. 주문수량(Qnty) 증가 특징 분석
def analyze_bulk_order(df):
    print("\n[4] 주문수량 증가가 나타나는 상품 특징")
    
    # 1회 주문 시 2개 이상 구매 건
    bulk_orders = df[df['주문수량'] >= 2]
    
    # 대량 구매가 잦은 과수 크기/무게
    bulk_char = bulk_orders.groupby(['과수 크기', '무게 구분'])['주문번호'].nunique().sort_values(ascending=False).head(5)
    print("- 대량 주문(2개 이상) 주요 특성:")
    for k, v in bulk_char.items():
        print(f"  {k}: {v}건")
    
    avg_price_bulk = bulk_orders['판매단가'].mean()
    avg_price_total = df[df['판매단가'] > 0]['판매단가'].mean()
    print(f"\n- 대량 주문 상품 평균 단가: {avg_price_bulk:,.0f}원 (전체 평균: {avg_price_total:,.0f}원)")

# 5. 입구 상품(구매 유도형 상품) 후보 도출
def identify_hero_products(df):
    print("\n[5] 입구 상품(Hero Product) 후보 도출")
    
    # 사용자별 첫 주문 데이터만 추출
    df_sorted = df.sort_values(['UID', '주문일'])
    df_first = df_sorted.drop_duplicates('UID', keep='first')
    
    # 첫 구매 유도력이 높은 상품 조합
    hero_candidates = df_first.groupby(['상품명', '과수 크기', '무게 구분'])['UID'].nunique().sort_values(ascending=False).head(10)
    
    print("- 신규 고객 첫 구매 유도 Top 10 상품:")
    for i, (idx, val) in enumerate(hero_candidates.items()):
        print(f"  {i+1}. {idx}: {val}명 유입")

# 실행
if __name__ == "__main__":
    analyze_price_and_option(df)
    analyze_keywords(df)
    analyze_cancellation(df)
    analyze_bulk_order(df)
    identify_hero_products(df)


# 실행
if __name__ == "__main__":
    import sys
    # 결과를 파일로 리다이렉션
    report_path = r'C:\Data\ICB6\workspace\mffarm04\eda_report_output.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        sys.stdout = f
        print("# 상품 구조 기반 구매 행동 분석 결과\n")
        analyze_price_and_option(df)
        analyze_keywords(df)
        analyze_cancellation(df)
        analyze_bulk_order(df)
        identify_hero_products(df)
    
    # 표준 출력 복구
    sys.stdout = sys.__stdout__
    print(f"분석 완료! 결과가 다음 경로에 저장되었습니다: {report_path}")
