import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Cấu hình giao diện Streamlit Full-width chuyên nghiệp
st.set_page_config(page_title="AI Bankruptcy Prediction", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

# 1. Tải cỗ máy tài nguyên thông minh mới
@st.cache_resource
def load_resources():
    try:
        data = joblib.load('models/enterprise_resources.pkl')
        return data['models'], data['scaler'], data['features'], data['metrics'], data['importances']
    except FileNotFoundError:
        return None, None, None, None, None

def format_feature_name(f):
    mapping = {
        'feature_1': '1. ROA (Lợi nhuận/Tổng tài sản)',
        'feature_2': '2. Tỷ lệ Nợ/Vốn CSH',
        'feature_3': '3. Khả năng thanh toán',
        'feature_4': '4. Vòng quay Hàng tồn kho',
        'feature_5': '5. LN Giữ lại/Tổng tài sản',
    }
    if f in mapping:
        return mapping[f]
    elif f.startswith("feature_"):
        return f"Chỉ số phụ trợ {f.split('_')[-1]}"
    return f

# TIÊU ĐỀ
st.title("🏦 Nền Tảng Dự Báo Rủi Ro Tài Chính Doanh Nghiệp (Bản Cao Cấp)")
st.markdown("Hệ thống **Trí tuệ nhân tạo (AI)** phát hiện sớm nguy cơ sụp đổ tài chính dựa trên dữ liệu chuẩn kế toán.")
st.markdown("---")

models, scaler, feature_list, metrics, importances = load_resources()

if models is None:
    st.error("⚠️ Không tìm thấy Tài nguyên Cốt lõi (Core Engine). Vui lòng huấn luyện mô hình bằng file `train_model.py`.")
else:
    # 2. Xây dựng SIDEBAR CHUYÊN NGHIỆP
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2936/2936758.png", width=100)
    st.sidebar.markdown("## ⚙️ Thiết Lập Phân Tích")
    
    # Cho phép người dùng chọn mô hình (rất chuyên nghiệp)
    selected_model_name = st.sidebar.selectbox(
        "🧠 Engine Trí Tuệ Nhân Tạo:", 
        options=['XGBoost', 'Random Forest', 'Logistic Regression'],
        index=0,
        help="XGBoost hiện đang là thuật toán top 1 về xử lý Dữ liệu dạng Bảng."
    )
    active_model = models[selected_model_name]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Form Cân Đối Kế Toán")
    user_inputs = []
    num_main = min(5, len(feature_list))
    for i in range(num_main):
        val = st.sidebar.number_input(format_feature_name(feature_list[i]), value=0.5, step=0.01)
        user_inputs.append(val)
        
    if len(feature_list) > 5:
        with st.sidebar.expander("➕ Chi tiết tham số nâng cao (>90 chỉ số)"):
            for i in range(5, len(feature_list)):
                val = st.number_input(format_feature_name(feature_list[i]), value=0.01, step=0.01)
                user_inputs.append(val)

    # 3. CHIA GIAO DIỆN THÀNH CÁC TABS
    tab1, tab2, tab3 = st.tabs([
        "🔍 Cố vấn Doanh nghiệp (1-1)", 
        "🌍 Phân tích Thị trường Xuyên suốt (Big Data)",
        "🔬 Báo cáo Chỉ số Kiểm định Mô hình AI"
    ])

    # === TAB 1: PHÂN TÍCH 1 CÔNG TY ===
    with tab1:
        st.markdown(f"### Phân tích tình hình Tài chính hiện tại bằng thuật toán `{selected_model_name}`")
        if st.button("Phát Biểu Đo Lường", type="primary", use_container_width=True):
            # Tính toán
            data_input = np.array([user_inputs])
            data_scaled = scaler.transform(data_input)
            
            # Khác với bản cũ, bản cao cấp không cần chèn qua PCA để giữ lại thực thể dữ liệu
            probability = active_model.predict_proba(data_scaled)[0][1]
            prediction = active_model.predict(data_scaled)[0]

            st.markdown("#### 🎫 Trạng Thái Hồ Sơ Chuyên Sâu:")
            # Hộp thông báo màu sắc nổi bật
            if prediction == 1:
                st.error(f"🚨 **TÍN HIỆU ĐỎ NHẤP NHÁY**: Tổ chức hiện đứng bên bờ vực **PHÁ SẢN TRẦM TRỌNG**.")
            else:
                st.success(f"✅ **TÍN HIỆU XANH**: Cân đối kế toán rất vững chắc và **HOÀN TOÀN KHỎE MẠNH**.")

            col1, col2 = st.columns([1, 1.2])
            
            # Biểu đồ Đo tốc độ rủi ro Gauge
            with col1:
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = probability * 100,
                    number = {'suffix': "%", 'font': {'size': 60, 'color': '#ff4b4b' if prediction == 1 else '#00cc96'}},
                    delta = {'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                    title = {'text': "Chỉ báo Nguy Cơ Phá Sản", 'font': {'size': 24}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 2},
                        'bar': {'color': "rgba(0,0,0,0)"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 40], 'color': "#e6ffe6"},
                            {'range': [40, 70], 'color': "#fff0b3"},
                            {'range': [70, 100], 'color': "#ffcccc"}],
                        'threshold': {
                            'line': {'color': "#ff4b4b" if prediction == 1 else "#00cc96", 'width': 8},
                            'thickness': 1, 'value': probability * 100}}))
                fig_gauge.update_layout(height=450, margin=dict(l=30, r=30, t=50, b=30))
                st.plotly_chart(fig_gauge, use_container_width=True)

            # Biểu đồ Trọng Số - Tại sao AI quyết định như vậy? (Chỉ XGBoost/RF có)
            with col2:
                if selected_model_name in ['XGBoost', 'Random Forest']:
                    st.markdown("#### 📌 10 Tham Số Đang Kéo Tổ Chức Xuống/Lên")
                    # So sánh dữ liệu thực tế nhập vào với top features importances
                    top_keys = list(importances.keys())[:10]
                    top_vals = [user_inputs[feature_list.index(k)] for k in top_keys]
                    top_names = [format_feature_name(k) for k in top_keys]
                    
                    df_reason = pd.DataFrame({"Tham số": top_names, "Chỉ số Thực Tế": top_vals})
                    fig_reason = px.bar(df_reason, x="Chỉ số Thực Tế", y="Tham số", orientation='h', color="Chỉ số Thực Tế",
                                        title=f"Độ nhạy Cân Đối Kế Toán Theo {selected_model_name}", text_auto='.2f')
                    fig_reason.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
                    st.plotly_chart(fig_reason, use_container_width=True)
                else:
                    st.info("💡 Linear Logistic Regression hoạt động theo dạng tối ưu Gradient Descent toàn diện nên phân tán trọng số đồng đều hơn.")

    # === TAB 2: PHÂN TÍCH THỊ TRƯỜNG DỮ LIỆU LỚN ===
    with tab2:
        st.markdown("### 🧬 Rà Soát Tự Động Định Lượng Qua Dữ Liệu Lớn")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("📄 Tổng Cơ Sở Dữ Liệu Mẫu", "Chưa tải", "Cần chạy lệnh")
        c2.metric("☠️ Doanh Nghiệp Cảnh Báo Đỏ", "Chưa tải", "Chưa đánh giá")
        c3.metric("🌱 Doanh Nghiệp Phát Triển Bền Vững", "Chưa tải", "Chưa phân loại")
        st.markdown("---")
        
        if st.button("🔥 KÍCH HOẠT HỆ ĐIỀU HÀNH DỮ LIỆU LỚN", type="primary", use_container_width=True):
            with st.spinner(f"Khuôn mẫu suy diễn `{selected_model_name}` đang duyệt hồ sơ hàng nghìn công ty..."):
                try:
                    df = pd.read_excel('dữ-liệu-lớn.xlsx')
                    X = df.drop('bankrupt', axis=1, errors='ignore')
                    X_input = X[feature_list] if set(feature_list).issubset(X.columns) else X
                    
                    data_scaled = scaler.transform(X_input)
                    probabilities = active_model.predict_proba(data_scaled)[:, 1]
                    predictions = active_model.predict(data_scaled)
                    
                    # Cập nhật Metrics ở trên
                    total = len(df)
                    danger = sum(predictions)
                    safe = total - danger
                    
                    c1.metric("📄 Tổng Cơ sở Dữ liệu Kết xuất", f"{total:,}", "Gói hàng triệu Parameters")
                    c2.metric("☠️ Doanh Nghiệp Rủi Ro Cao", f"{danger:,}", f"Chiếm {(danger/total):.1%}", delta_color="inverse")
                    c3.metric("🌱 Doanh Nghiệp Vững Mạnh", f"{safe:,}", f"Chiếm {(safe/total):.1%}", delta_color="normal")
                    
                    # Kết quả Dataframe siêu mượt
                    results_df = df.copy()
                    results_df.rename(columns=lambda x: format_feature_name(x) if x in feature_list else x, inplace=True)
                    results_df.insert(0, 'Áp Lực Phá Sản (%)', np.round(probabilities * 100, 2))
                    results_df.insert(0, 'Kết Luận Trọng Tài', ["🔴 Phá Sản" if p == 1 else "🟢 Ổn Định" for p in predictions])
                    
                    st.dataframe(results_df, use_container_width=True, height=350)
                    
                    # Vẽ biểu đồ
                    col_chart1, col_chart2 = st.columns(2)
                    with col_chart1:
                        fig_pie = px.pie(values=[danger, safe], names=['🔴 Nguy Cơ', '🟢 An Toàn'], 
                                         hole=0.55, title=f"Tỉ Trọng Xâm Lấn Của Cháy Tài Khoản",
                                         color_discrete_sequence=['#ff4b4b', '#00cc96'])
                        fig_pie.update_traces(textposition='outside', textinfo='percent+label')
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col_chart2:
                        fig_box = px.histogram(results_df, x="Áp Lực Phá Sản (%)", color="Kết Luận Trọng Tài", 
                                               marginal="violin", nbins=30, title="Bản Đồ Tập Trung Dịch Chuyển Áp Lực",
                                               color_discrete_map={"🔴 Phá Sản": "#ff4b4b", "🟢 Ổn Định": "#00cc96"})
                        fig_box.update_layout(barmode="overlay", legend=dict(orientation="h", yanchor="bottom", y=1))
                        st.plotly_chart(fig_box, use_container_width=True)
                except Exception as e:
                    st.error(f"Lỗi đọc Data Pipeline: {str(e)}")

    # === TAB 3: MODEL INSIGHTS (SIÊU CHUYÊN NGHIỆP) ===
    with tab3:
        st.markdown("### 🔭 Đài Quan Sát: Phân rã Thuật toán Trí tuệ Nhân tạo")
        st.markdown("Xác minh chéo mức độ thông minh và khả năng học sâu của các thuật toán được áp dụng.")
        
        # 1. So sánh Performance
        st.markdown("#### 1. Bảng Xếp Hạng Hiệu Suất Nhận Diện")
        df_metrics = pd.DataFrame(metrics).T.reset_index().rename(columns={'index': 'Thuật Toán'})
        # Format Percentage
        for col in ['Accuracy', 'F1-Score', 'ROC-AUC']:
            df_metrics[col] = (df_metrics[col] * 100).round(2)
            
        fig_metrics = px.bar(df_metrics.melt(id_vars='Thuật Toán', var_name='Chỉ báo', value_name='Score (%)'), 
                             x='Thuật Toán', y='Score (%)', color='Chỉ báo', barmode='group', text_auto='.2f',
                             title="Cuộc Đua Tối Ưu Hóa: So Sánh Tính Trưởng Thành Của AI", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_metrics.update_layout(height=400)
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        # 2. Most Importances
        st.markdown("#### 2. Các Mắt Xích Kế Toán Tử Thần (Feature Importance)")
        st.info("Mô hình **XGBoost** bằng cơ chế Tree-based Boosting đã tìm ra đây là những thông số kế toán gây đột quỵ mạnh mẽ nhất cho mạch máu doanh nghiệp.")
        
        df_imp = pd.DataFrame(list(importances.items()), columns=["Mã lệnh", "Mức Độ Nghiêm Trọng"])
        df_imp["Lĩnh Vực"] = df_imp["Mã lệnh"].apply(format_feature_name)
        
        fig_imp = px.bar(df_imp, x="Mức Độ Nghiêm Trọng", y="Lĩnh Vực", orientation='h', color="Mức Độ Nghiêm Trọng",
                         title="Biểu đồ Lan Tỏa (Top 20 Radar Cốt Lõi)", color_continuous_scale="Reds")
        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'}, height=600)
        st.plotly_chart(fig_imp, use_container_width=True)
