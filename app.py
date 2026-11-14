import streamlit as st
import sqlite3
import pandas as pd
from sklearn.cluster import KMeans
import random

# 페이지 기본 설정
st.set_page_config(page_title="Spotify 맞춤형 곡 추천 시스템", page_icon="🎵", layout="wide")

def get_db_connection():
    return sqlite3.connect('spotify.db')

@st.cache_data
def load_all_tracks():
    conn = get_db_connection()
    query = "SELECT DISTINCT track_id, track_name, track_artist FROM TRACK"
    df = pd.read_sql_query(query, conn)
    df['display_name'] = df['track_name'] + " - " + df['track_artist']
    conn.close()
    return df

# ==========================================
# [공통 함수] 5가지 추천 알고리즘 처리 함수
# (DB 플레이리스트, 장바구니, CSV 업로드 모두 이 함수 하나로 호환됩니다)
# ==========================================
def get_virtual_recommendations(user_df, option, top_n=10):
    conn = get_db_connection()
    exclude_cond = ""
    
    if 'track_id' in user_df.columns:
        track_ids = user_df['track_id'].dropna().tolist()
        if track_ids:
            placeholders = ','.join([f"'{t}'" for t in track_ids])
            exclude_cond = f"AND track_id NOT IN ({placeholders})"
    
    # 1. 전체 평균 취향 저격
    if option == "1":
        avg_d, avg_e, avg_v, avg_a = user_df['danceability'].mean(), user_df['energy'].mean(), user_df['valence'].mean(), user_df['acousticness'].mean()
        sql = f"""
        SELECT DISTINCT '🎯 전체 평균 취향저격' AS [추천 방식], track_name AS [곡명], track_artist AS [아티스트], track_popularity AS [인기도],
            ROUND(SQRT(POWER(danceability - {avg_d}, 2) + POWER(energy - {avg_e}, 2) + POWER(valence - {avg_v}, 2) + POWER(acousticness - {avg_a}, 2)), 4) AS [오디오 거리]
        FROM TRACK WHERE 1=1 {exclude_cond} ORDER BY [오디오 거리] ASC, track_popularity DESC LIMIT {top_n};
        """
        result_df = pd.read_sql_query(sql, conn)
        
    # 2. 연관 추천 (자카드)
    elif option == "2":
        if 'track_id' not in user_df.columns or not track_ids:
            return pd.DataFrame({"오류": ["'track_id' 데이터가 없어 연관 추천을 실행할 수 없습니다."]})
        sql = f"""
        WITH playlist_stats AS (SELECT playlist_id, COUNT(DISTINCT track_id) as pl_size FROM TRACK GROUP BY playlist_id),
        playlist_similarity AS (
            SELECT tp.playlist_id, COUNT(DISTINCT tp.track_id) AS intersection_count, {len(track_ids)} AS cart_size
            FROM TRACK tp WHERE tp.track_id IN ({placeholders}) GROUP BY tp.playlist_id
        )
        SELECT DISTINCT '🎧 연관 추천' AS [추천 방식], t.track_name AS [곡명], t.track_artist AS [아티스트], t.track_popularity AS [인기도],
            ROUND(CAST(ps.intersection_count AS FLOAT) / (ps.cart_size + s.pl_size - ps.intersection_count), 4) AS [Jaccard 점수]
        FROM TRACK t JOIN playlist_similarity ps ON t.playlist_id = ps.playlist_id JOIN playlist_stats s ON ps.playlist_id = s.playlist_id
        WHERE 1=1 {exclude_cond} ORDER BY [Jaccard 점수] DESC, t.track_popularity DESC LIMIT {top_n};
        """
        result_df = pd.read_sql_query(sql, conn)
        
    # 3. 장르 히트곡
    elif option == "3":
        if 'playlist_genre' not in user_df.columns:
            return pd.DataFrame({"오류": ["'playlist_genre' 데이터가 없어 장르 추천을 실행할 수 없습니다."]})
        user_genre = user_df['playlist_genre'].mode()[0]
        user_subgenre = user_df['playlist_subgenre'].mode()[0] if 'playlist_subgenre' in user_df.columns else ""
        
        sql = f"""
        SELECT DISTINCT '🔥 장르 인기곡' AS [추천 방식], track_name AS [곡명], track_artist AS [아티스트], playlist_genre AS [상위장르], track_popularity AS [인기도]
        FROM TRACK WHERE playlist_genre = '{user_genre}' {exclude_cond}
        ORDER BY CASE WHEN playlist_subgenre = '{user_subgenre}' THEN 0 ELSE 1 END ASC, track_popularity DESC LIMIT {top_n};
        """
        result_df = pd.read_sql_query(sql, conn)
        
    # 4. 개별 곡 기준 필터링 (Item-based)
    elif option == "4":
        sample_size = min(3, len(user_df))
        sampled_df = user_df.sample(n=sample_size)
        
        recommended_list = []
        for _, row in sampled_df.iterrows():
            t_dance, t_energy, t_val, t_acous = row['danceability'], row['energy'], row['valence'], row['acousticness']
            sql = f"""
            SELECT DISTINCT '🎸 개별 곡 기준' AS [추천 방식], track_name AS [곡명], track_artist AS [아티스트], track_popularity AS [인기도],
                ROUND(SQRT(POWER(danceability - {t_dance}, 2) + POWER(energy - {t_energy}, 2) + POWER(valence - {t_val}, 2) + POWER(acousticness - {t_acous}, 2)), 4) AS [오디오 거리]
            FROM TRACK WHERE 1=1 {exclude_cond} ORDER BY [오디오 거리] ASC, track_popularity DESC LIMIT 4;
            """
            res_df = pd.read_sql_query(sql, conn)
            recommended_list.append(res_df)
            
        final_df = pd.concat(recommended_list, ignore_index=True)
        result_df = final_df.drop_duplicates(subset=['곡명', '아티스트']).sort_values('오디오 거리').head(top_n).reset_index(drop=True)

    # 5. 군집화 (K-Means Clustering)
    elif option == "5":
        if len(user_df) < 5:
            return pd.DataFrame({"오류": ["군집화를 돌리려면 최소 5곡 이상의 데이터가 필요합니다!"]})
            
        features = user_df[['danceability', 'energy', 'valence', 'acousticness']].fillna(0)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        user_df['cluster'] = kmeans.fit_predict(features)
        
        recommended_list = []
        for cluster_num in range(2):
            cluster_data = user_df[user_df['cluster'] == cluster_num]
            if cluster_data.empty:
                continue
            avg_d, avg_e, avg_v, avg_a = cluster_data['danceability'].mean(), cluster_data['energy'].mean(), cluster_data['valence'].mean(), cluster_data['acousticness'].mean()
            
            sql = f"""
            SELECT DISTINCT '🔮 군집 {cluster_num+1} 타겟' AS [추천 방식], track_name AS [곡명], track_artist AS [아티스트], track_popularity AS [인기도],
                ROUND(SQRT(POWER(danceability - {avg_d}, 2) + POWER(energy - {avg_e}, 2) + POWER(valence - {avg_v}, 2) + POWER(acousticness - {avg_a}, 2)), 4) AS [오디오 거리]
            FROM TRACK WHERE 1=1 {exclude_cond} ORDER BY [오디오 거리] ASC, track_popularity DESC LIMIT 5;
            """
            res_df = pd.read_sql_query(sql, conn)
            recommended_list.append(res_df)
            
        if recommended_list:
            final_df = pd.concat(recommended_list, ignore_index=True)
            result_df = final_df.drop_duplicates(subset=['곡명', '아티스트']).sort_values('오디오 거리').head(top_n).reset_index(drop=True)
        else:
            result_df = pd.DataFrame()

    conn.close()
    return result_df


# --- 메인 화면 UI 시작 ---
st.title("🎧 Spotify 맞춤형 곡 추천 시스템")
st.markdown("대규모 플레이리스트(100곡 이상)의 데이터 희석 문제를 방어하는 5가지 알고리즘을 모든 모드에서 지원합니다.")

mode = st.radio("🛠️ 작동 모드를 선택하세요:", ["1. 기존 데이터베이스에서 선택", "2. 내 맘대로 장바구니 담기", "3. 외부 CSV 파일 직접 업로드"])
st.markdown("---")

# ==========================================
# UI 1. 기존 데이터베이스 모드 (5가지 알고리즘 적용)
# ==========================================
if mode == "1. 기존 데이터베이스에서 선택":
    conn = get_db_connection()
    playlist_df = pd.read_sql_query("SELECT DISTINCT playlist_id, playlist_name, playlist_genre FROM TRACK", conn)
    st.subheader("⚙️ 기존 플레이리스트 선택")
    selected_playlist_name = st.selectbox("DB에 있는 플레이리스트를 고르세요:", playlist_df['playlist_name'].unique())
    target_playlist_id = playlist_df[playlist_df['playlist_name'] == selected_playlist_name].iloc[0]['playlist_id']
    
    # 선택한 플레이리스트의 전체 곡 정보를 DataFrame으로 로드
    db_playlist_df = pd.read_sql_query(f"SELECT * FROM TRACK WHERE playlist_id = '{target_playlist_id}'", conn)
    conn.close()
    
    st.success(f"'{selected_playlist_name}' 플레이리스트 (총 {len(db_playlist_df)}곡)가 로드되었습니다. 아래에서 원하는 알고리즘을 선택하세요!")
    
    st.divider()
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 1. 전체 평균 취향", "🎧 2. 연관 추천", "🔥 3. 장르 히트곡", "🎸 4. 개별 곡 기준(빠름)", "🔮 5. 군집화 분리(정교함)"])
    
    with tab1:
        if st.button("🎯 전체 평균 기반 추천", key="db_b1"): st.dataframe(get_virtual_recommendations(db_playlist_df, "1"), use_container_width=True)
    with tab2:
        if st.button("🎧 자카드 연관 추천", key="db_b2"): st.dataframe(get_virtual_recommendations(db_playlist_df, "2"), use_container_width=True)
    with tab3:
        if st.button("🔥 장르 인기곡 추천", key="db_b3"): st.dataframe(get_virtual_recommendations(db_playlist_df, "3"), use_container_width=True)
    with tab4:
        st.info("랜덤 핵심 곡 3개를 뽑아 각각 가까운 곡을 찾아 합칩니다. 연산이 빠릅니다.")
        if st.button("🎸 개별 곡 기준 추천", key="db_b4"): st.dataframe(get_virtual_recommendations(db_playlist_df, "4"), use_container_width=True)
    with tab5:
        st.info("K-Means 군집화로 거대한 취향을 2그룹으로 분리하여 엉뚱한 평균 희석을 막습니다.")
        if st.button("🔮 군집화 분리 추천", key="db_b5"): st.dataframe(get_virtual_recommendations(db_playlist_df, "5"), use_container_width=True)

# ==========================================
# UI 2. 장바구니 모드 (5가지 알고리즘 적용)
# ==========================================
elif mode == "2. 내 맘대로 장바구니 담기":
    st.subheader("🛒 내 취향 장바구니 담기")
    track_db = load_all_tracks()
    selected_display_names = st.multiselect("곡명이나 아티스트명을 검색하여 여러 곡을 담아보세요:", options=track_db['display_name'])
    
    if selected_display_names:
        selected_track_ids = track_db[track_db['display_name'].isin(selected_display_names)]['track_id'].tolist()
        
        conn = get_db_connection()
        placeholders = ','.join(['?'] * len(selected_track_ids))
        query = f"SELECT * FROM TRACK WHERE track_id IN ({placeholders})"
        cart_df = pd.read_sql_query(query, conn, params=selected_track_ids)
        conn.close()

        st.success(f"총 {len(selected_track_ids)}곡이 담겼습니다. 5가지 분석 방식을 지원합니다!")
        
        st.divider()
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 1. 전체 평균 취향", "🎧 2. 연관 추천", "🔥 3. 장르 히트곡", "🎸 4. 개별 곡 기준(빠름)", "🔮 5. 군집화 분리(정교함)"])
        
        with tab1:
            if st.button("🎯 전체 평균 기반 추천", key="c_b1"): st.dataframe(get_virtual_recommendations(cart_df, "1"), use_container_width=True)
        with tab2:
            if st.button("🎧 자카드 연관 추천", key="c_b2"): st.dataframe(get_virtual_recommendations(cart_df, "2"), use_container_width=True)
        with tab3:
            if st.button("🔥 장르 인기곡 추천", key="c_b3"): st.dataframe(get_virtual_recommendations(cart_df, "3"), use_container_width=True)
        with tab4:
            st.info("랜덤 핵심 곡 3개를 뽑아 각각 가까운 곡을 찾아 합칩니다. 연산이 빠릅니다.")
            if st.button("🎸 개별 곡 기준 추천", key="c_b4"): st.dataframe(get_virtual_recommendations(cart_df, "4"), use_container_width=True)
        with tab5:
            st.info("K-Means 군집화로 취향을 2그룹으로 분리하여 엉뚱한 평균 희석을 막습니다.")
            if st.button("🔮 군집화 분리 추천", key="c_b5"): st.dataframe(get_virtual_recommendations(cart_df, "5"), use_container_width=True)

# ==========================================
# UI 3. 외부 CSV 업로드 모드 (5가지 알고리즘 적용)
# ==========================================
elif mode == "3. 외부 CSV 파일 직접 업로드":
    st.subheader("📁 플레이리스트 CSV 업로드")
    uploaded_file = st.file_uploader("CSV 파일을 선택하세요", type=['csv'])
    
    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        if all(col in user_df.columns for col in ['danceability', 'energy', 'valence', 'acousticness']):
            st.success(f"파일 분석 완료! (총 {len(user_df)}곡) 대규모 플레이리스트 엣지 케이스를 방어하는 5가지 알고리즘을 사용해 보세요.")
            
            st.divider()
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 1. 전체 평균", "🎧 2. 연관 추천", "🔥 3. 장르 히트곡", "🎸 4. 개별 곡 기준(빠름)", "🔮 5. 군집화 분리(정교함)"])
            
            with tab1:
                if st.button("🎯 전체 평균 추천", key="f_b1"): st.dataframe(get_virtual_recommendations(user_df, "1"), use_container_width=True)
            with tab2:
                if st.button("🎧 연관 곡 추천", key="f_b2"): st.dataframe(get_virtual_recommendations(user_df, "2"), use_container_width=True)
            with tab3:
                if st.button("🔥 장르 인기곡 추천", key="f_b3"): st.dataframe(get_virtual_recommendations(user_df, "3"), use_container_width=True)
            with tab4:
                st.info("랜덤 핵심 곡을 타겟팅하여 빠르게 추천합니다.")
                if st.button("🎸 개별 곡 타겟팅", key="f_b4"): st.dataframe(get_virtual_recommendations(user_df, "4"), use_container_width=True)
            with tab5:
                st.info("머신러닝 군집화로 취향을 정확하게 나눕니다.")
                if st.button("🔮 군집화 타겟팅", key="f_b5"): st.dataframe(get_virtual_recommendations(user_df, "5"), use_container_width=True)
        else:
            st.error("업로드하신 파일에 필수 오디오 특성 열이 부족합니다.")