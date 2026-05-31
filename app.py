
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Retail Sales Analytics Dashboard",
    page_icon="🛒",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data

def load_data():
    df = pd.read_csv("sales_data.csv")

    # Convert date
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    # Rename column
    if 'Weekly_Sales' in df.columns:
        df.rename(columns={'Weekly_Sales': 'Sales'}, inplace=True)

    # Feature engineering
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.strftime('%B')

    return df


# Load dataset

df = load_data()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("📊 Dashboard Filters")

selected_store = st.sidebar.selectbox(
    "Select Store",
    sorted(df['Store'].unique())
)

selected_year = st.sidebar.selectbox(
    "Select Year",
    sorted(df['Year'].unique())
)

holiday_filter = st.sidebar.radio(
    "Holiday Sales",
    ["All", "Holiday Only", "Non-Holiday"]
)

# =====================================================
# FILTER DATA
# =====================================================

filtered_df = df[
    (df['Store'] == selected_store) &
    (df['Year'] == selected_year)
]

if holiday_filter == "Holiday Only":
    filtered_df = filtered_df[filtered_df['Holiday_Flag'] == 1]

elif holiday_filter == "Non-Holiday":
    filtered_df = filtered_df[filtered_df['Holiday_Flag'] == 0]

# =====================================================
# HEADER
# =====================================================

st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>
        🛒 Retail Sales Analytics Dashboard
    </h1>
    <p style='text-align: center; font-size:18px;'>
        Real-Time Business Intelligence & Sales Forecasting
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# =====================================================
# KPI SECTION
# =====================================================

st.subheader("📌 Key Business Metrics")

col1, col2, col3, col4 = st.columns(4)

# Metrics

total_sales = filtered_df['Sales'].sum()
avg_sales = filtered_df['Sales'].mean()
max_sales = filtered_df['Sales'].max()
min_sales = filtered_df['Sales'].min()

with col1:
    st.metric(
        label="💰 Total Sales",
        value=f"${total_sales:,.0f}"
    )

with col2:
    st.metric(
        label="📈 Average Sales",
        value=f"${avg_sales:,.0f}"
    )

with col3:
    st.metric(
        label="🚀 Highest Sales",
        value=f"${max_sales:,.0f}"
    )

with col4:
    st.metric(
        label="📉 Lowest Sales",
        value=f"${min_sales:,.0f}"
    )

st.markdown("---")

# =====================================================
# SALES TREND
# =====================================================

st.subheader("📈 Weekly Sales Trend")

fig1 = px.line(
    filtered_df,
    x='Date',
    y='Sales',
    markers=True,
    title='Sales Performance Over Time'
)

fig1.update_layout(
    template='plotly_white',
    xaxis_title='Date',
    yaxis_title='Sales',
    height=500
)

st.plotly_chart(fig1, use_container_width=True)

# =====================================================
# MONTHLY SALES ANALYSIS
# =====================================================

st.subheader("📅 Monthly Sales Analysis")

monthly_sales = filtered_df.groupby('Month_Name')['Sales'].sum().reset_index()

month_order = [
    'January', 'February', 'March', 'April',
    'May', 'June', 'July', 'August',
    'September', 'October', 'November', 'December'
]

monthly_sales['Month_Name'] = pd.Categorical(
    monthly_sales['Month_Name'],
    categories=month_order,
    ordered=True
)

monthly_sales = monthly_sales.sort_values('Month_Name')

fig2 = px.bar(
    monthly_sales,
    x='Month_Name',
    y='Sales',
    title='Monthly Sales Distribution',
    text_auto=True
)

fig2.update_layout(
    template='plotly_white',
    xaxis_title='Month',
    yaxis_title='Sales',
    height=500
)

st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# HOLIDAY VS NON HOLIDAY SALES
# =====================================================

st.subheader("🎉 Holiday vs Non-Holiday Sales")

holiday_sales = filtered_df.groupby('Holiday_Flag')['Sales'].mean().reset_index()

holiday_sales['Holiday_Flag'] = holiday_sales['Holiday_Flag'].replace({
    0: 'Non-Holiday',
    1: 'Holiday'
})

fig3 = px.pie(
    holiday_sales,
    names='Holiday_Flag',
    values='Sales',
    title='Sales Contribution'
)

st.plotly_chart(fig3, use_container_width=True)

# =====================================================
# TEMPERATURE VS SALES
# =====================================================

st.subheader("🌡️ Temperature vs Sales")

fig4 = px.scatter(
    filtered_df,
    x='Temperature',
    y='Sales',
    size='Fuel_Price',
    hover_data=['Date'],
    title='Effect of Temperature on Sales'
)

fig4.update_layout(
    template='plotly_white',
    height=500
)

st.plotly_chart(fig4, use_container_width=True)

# =====================================================
# DATA TABLE
# =====================================================

st.subheader("📋 Detailed Sales Data")

st.dataframe(filtered_df)

# =====================================================
# DOWNLOAD OPTION
# =====================================================

csv = filtered_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="⬇️ Download Filtered Data",
    data=csv,
    file_name='filtered_sales_data.csv',
    mime='text/csv'
)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
    <div style='text-align: center;'>
        <h4>🚀 Retail Sales Forecasting Project</h4>
        <p>Built using Streamlit, Plotly, Pandas & Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True
)
