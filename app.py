
# =========================================
# 🚀 PREMIUM STREAMLIT DATA SCIENCE APP
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.feature_extraction.text import CountVectorizer

# =========================================
# 🎨 PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="🚀 AI YouTube Analytics",
    layout="wide",
    page_icon="📊"
)

# =========================================
# 🌗 DARK / LIGHT MODE
# =========================================
mode = st.sidebar.toggle("🌗 Dark Mode")

if mode:
    bg = "#0f172a"
    text = "white"
else:
    bg = "#f8fafc"
    text = "#0f172a"

st.markdown(f"""
    <style>
    .main {{
        background-color: {bg};
        color: {text};
    }}
    .stMetric {{
        background: rgba(255,255,255,0.08);
        padding: 15px;
        border-radius: 15px;
    }}
    .reportview-container .main {{
        color: {text};
    }}
    </style>
""", unsafe_allow_html=True)

# =========================================
# 📂 LOAD DATA
# =========================================
@st.cache_data
def load_data():
    df = pd.read_excel("youtube_trending_video_statistics_5000.xlsx", sheet_name=None)
    df = pd.concat(df.values(), ignore_index=True)
    df.drop_duplicates(inplace=True)
    return df

df = load_data()

# =========================================
# 🧹 CLEANING
# =========================================
df.columns = df.columns.str.lower().str.replace(" ", "_")

if 'publish_time' in df.columns:
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
    df['publish_date'] = df['publish_time'].dt.date
    df['publish_hour'] = df['publish_time'].dt.hour
else:
    df['publish_date'] = pd.NaT
    df['publish_hour'] = np.nan

if 'views' in df.columns and 'likes' in df.columns:
    df['like_rate'] = np.where(df['views'] > 0, (df['likes'] / df['views']) * 100, 0)
else:
    df['like_rate'] = np.nan

# =========================================
# 🎛️ SIDEBAR FILTERS
# =========================================
st.sidebar.title("⚙️ Advanced Filters")

if 'category_name' in df.columns:
    selected_cat = st.sidebar.multiselect(
        "Category",
        sorted(df['category_name'].dropna().unique())
    )
    if selected_cat:
        df = df[df['category_name'].isin(selected_cat)]

if 'channel_title' in df.columns:
    selected_channel = st.sidebar.multiselect(
        "Channel",
        sorted(df['channel_title'].dropna().unique())
    )
    if selected_channel:
        df = df[df['channel_title'].isin(selected_channel)]

if 'publish_date' in df.columns and not df['publish_date'].isna().all():
    date_min = df['publish_date'].min()
    date_max = df['publish_date'].max()
    selected_dates = st.sidebar.date_input(
        "Publish Date Range",
        [date_min, date_max],
        min_value=date_min,
        max_value=date_max
    )
    if isinstance(selected_dates, list) and len(selected_dates) == 2:
        df = df[(df['publish_date'] >= selected_dates[0]) & (df['publish_date'] <= selected_dates[1])]

if 'views' in df.columns:
    views_range = st.sidebar.slider(
        "Views Range",
        int(df['views'].dropna().min()),
        int(df['views'].dropna().max()),
        (int(df['views'].dropna().min()), int(df['views'].dropna().max()))
    )
    df = df[(df['views'] >= views_range[0]) & (df['views'] <= views_range[1])]

if 'like_rate' in df.columns:
    like_rate_range = st.sidebar.slider(
        "Engagement Rate (%)",
        0.0,
        100.0,
        (0.0, 100.0)
    )
    df = df[(df['like_rate'] >= like_rate_range[0]) & (df['like_rate'] <= like_rate_range[1])]

st.sidebar.markdown("---")
st.sidebar.write("Built with Streamlit, Plotly, and scikit-learn")

# =========================================
# 🏠 HEADER
# =========================================
st.title("🚀 AI-Powered YouTube Analytics Dashboard")
st.markdown("### 💡 Data Science | ML | NLP | Technical Insights")

# =========================================
# DATA QUALITY INFO
# =========================================
with st.expander("📌 Data Quality & Counts", expanded=False):
    st.write(df.describe(include='all'))

# =========================================
# 📊 KPI CARDS
# =========================================
metrics = []
if 'category_name' in df.columns:
    best_category = df.groupby('category_name')['views'].mean().idxmax()
    metrics.append(("Top Category", best_category))
if 'channel_title' in df.columns:
    best_channel = df.groupby('channel_title')['views'].mean().idxmax()
    metrics.append(("Top Channel", best_channel))

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Videos", df.shape[0])
c2.metric("Avg Views", f"{df['views'].mean():,.0f}")
c3.metric("Avg Likes", f"{df['likes'].mean():,.0f}")
c4.metric("Engagement %", f"{df['like_rate'].mean():.2f}")

if metrics:
    extra_cols = st.columns(len(metrics))
    for idx, (label, value) in enumerate(metrics):
        extra_cols[idx].metric(label, value)

# =========================================
# 📊 SUMMARY CHARTS
# =========================================
st.subheader("📈 Views Trend")
if 'publish_time' in df.columns:
    fig = px.line(df.sort_values('publish_time'), x='publish_time', y='views', title="Views Over Time")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Publish time is not available in this dataset.")

st.subheader("📊 Engagement Distribution")
fig_hist = px.histogram(df, x='like_rate', nbins=40, title='Engagement Rate Distribution')
st.plotly_chart(fig_hist, use_container_width=True)

st.subheader("📊 Views vs Likes")
color_col = 'category_name' if 'category_name' in df.columns else None
fig_scatter = px.scatter(
    df,
    x='views',
    y='likes',
    color=color_col,
    title='Views vs Likes',
    opacity=0.7,
    hover_data=['title'] if 'title' in df.columns else None
)
st.plotly_chart(fig_scatter, use_container_width=True)

# =========================================
# 📊 CATEGORY / CHANNEL INSIGHTS
# =========================================
if 'category_name' in df.columns:
    st.subheader("🔥 Category Performance")
    category_summary = df.groupby('category_name')[['views', 'likes', 'like_rate']].mean().reset_index()
    fig = px.bar(category_summary.sort_values('views', ascending=False),
                 x='category_name', y='views', title='Avg Views by Category')
    st.plotly_chart(fig, use_container_width=True)

if 'channel_title' in df.columns:
    st.subheader("🎥 Top Channels by Avg Views")
    channel_summary = df.groupby('channel_title')['views'].mean().nlargest(10).reset_index()
    fig = px.bar(channel_summary, x='channel_title', y='views', title='Top Channels by Avg Views')
    st.plotly_chart(fig, use_container_width=True)

# =========================================
# 📊 CORRELATION INSIGHT
# =========================================
numeric_columns = [col for col in ['views', 'likes', 'comment_count', 'dislikes', 'like_rate'] if col in df.columns]
if len(numeric_columns) > 1:
    st.subheader("🧪 Correlation Matrix")
    corr = df[numeric_columns].corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title='Feature Correlation')
    st.plotly_chart(fig, use_container_width=True)

# =========================================
# 📉 TECHNICAL ANALYSIS
# =========================================
st.subheader("📉 Technical Analysis (Moving Averages)")
if 'publish_time' in df.columns:
    df_sorted = df.sort_values('publish_time').copy()
    df_sorted['MA7'] = df_sorted['views'].rolling(7).mean()
    df_sorted['MA30'] = df_sorted['views'].rolling(30).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_sorted['publish_time'], y=df_sorted['views'], name="Views"))
    fig.add_trace(go.Scatter(x=df_sorted['publish_time'], y=df_sorted['MA7'], name="MA7"))
    fig.add_trace(go.Scatter(x=df_sorted['publish_time'], y=df_sorted['MA30'], name="MA30"))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Publish date information is required for moving average analysis.")

# =========================================
# 🤖 MACHINE LEARNING MODEL
# =========================================
st.subheader("🤖 ML Model (Predict Views)")
features = ['likes', 'comment_count'] if 'comment_count' in df.columns else ['likes']

if len(features) == 1 and df[features[0]].dropna().empty:
    st.warning("Not enough data to train a predictive model.")
else:
    df_ml = df.dropna(subset=features + ['views'])
    if df_ml.shape[0] < 10:
        st.warning("Not enough complete rows to train the model.")
    else:
        X = df_ml[features]
        y = df_ml['views']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = r2_score(y_test, pred)
        st.metric("Model Accuracy (R²)", f"{score:.2f}")

        st.markdown("**Predict views from a new video sample**")
        input_cols = st.columns(len(features))
        sample_values = []
        for idx, feature in enumerate(features):
            minimum = int(df_ml[feature].min()) if df_ml[feature].min() >= 0 else 0
            maximum = int(df_ml[feature].max())
            default_value = int(df_ml[feature].median())
            sample_values.append(
                input_cols[idx].number_input(
                    feature.replace('_', ' ').title(),
                    min_value=minimum,
                    max_value=maximum,
                    value=default_value,
                    step=max(1, int((maximum - minimum) / 20))
                )
            )

        prediction = model.predict([sample_values])[0]
        st.metric("Predicted Views", f"{int(prediction):,}")

# =========================================
# 🧠 NLP ANALYSIS
# =========================================
st.subheader("🧠 NLP - Trending Keywords")
if 'title' in df.columns and df['title'].dropna().shape[0] > 0:
    cv = CountVectorizer(stop_words='english', max_features=25)
    words = cv.fit_transform(df['title'].dropna())
    word_freq = pd.DataFrame({
        'word': cv.get_feature_names_out(),
        'count': words.toarray().sum(axis=0)
    }).sort_values(by='count', ascending=False)
    fig = px.bar(word_freq, x='word', y='count', title="Top Keywords in Titles")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No title text found for keyword analysis.")

# =========================================
# 🔥 TOP CONTENT
# =========================================
st.subheader("🔥 Top Performing Videos")
if 'views' in df.columns:
    st.dataframe(df.sort_values('views', ascending=False).head(10))
else:
    st.info("Views column is required to show top performing videos.")

# =========================================
# 📥 EXPORT
# =========================================
st.subheader("📥 Export Data")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, "cleaned_data.csv")
