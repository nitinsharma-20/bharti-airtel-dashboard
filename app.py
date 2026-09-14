import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Bharti Airtel Dashboard", layout="wide")
st.title("📈 Bharti Airtel Stock Market Dashboard")

df = yf.download("BHARTIARTL.NS", start="2021-01-01", progress=False, auto_adjust=False)

if isinstance(df.columns, __import__("pandas").MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.reset_index()
df["MA20"] = df["Close"].rolling(20).mean()

X = df[["Open","High","Low","Volume"]].iloc[:-1]
y = df["Close"].shift(-1).dropna()

model = LinearRegression().fit(X, y)
pred = float(model.predict(df[["Open","High","Low","Volume"]].tail(1))[0])

c1,c2,c3,c4 = st.columns(4)
c1.metric("Current Price", f"₹{float(df['Close'].iloc[-1]):.2f}")
c2.metric("52W High", f"₹{float(df['High'].max()):.2f}")
c3.metric("52W Low", f"₹{float(df['Low'].min()):.2f}")
c4.metric("Predicted", f"₹{pred:.2f}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close", line=dict(color="red")))
fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], name="20 MA", line=dict(color="orange")))
st.plotly_chart(fig, use_container_width=True)

vol = go.Figure(go.Bar(x=df["Date"], y=df["Volume"], marker_color="teal"))
st.plotly_chart(vol, use_container_width=True)

st.subheader("Recent Data")
st.dataframe(df.tail(20), use_container_width=True)


vol = go.Figure(go.Bar(x=df.Date, y=df.Volume, marker_color="teal"))
st.plotly_chart(vol, use_container_width=True)
st.dataframe(df.tail(20), use_container_width=True)
