import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from requests.exceptions import RequestException
from datetime import timedelta
import numpy as np
import hmac

def fetch_data(ticker, range_param, interval, max_retries=5, retry_delay=60):
  url = f'https://query1.finance.yahoo.com/v7/finance/chart/{ticker}?range={range_param}&interval={interval}&indicators=quote&includeTimestamps=true'
  
  headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
  }

  for attempt in range(max_retries):
    try:
      response = requests.get(url, headers=headers)
      response.raise_for_status()
      return response.json()
    except RequestException as e:
      print(f"エラーが発生しました: {e}")
      if attempt < max_retries - 1:
        print(f"{retry_delay}秒後に再試行します...")
        time.sleep(retry_delay)
      else:
        print("最大再試行回数に達しました。プログラムを終了します。")
        raise
def filter_trading_hours(df):
  # 日本の取引時間（9:00-11:30, 12:30-15:00）に制限
  df['Time'] = df['Timestamp'].dt.time
  mask = (
      ((df['Time'] >= pd.Timestamp('09:00').time()) & (df['Time'] <= pd.Timestamp('11:30').time())) |
      ((df['Time'] >= pd.Timestamp('12:30').time()) & (df['Time'] <= pd.Timestamp('15:00').time()))
  )
  return df[mask].drop('Time', axis=1)

def calculate_indicators(df):
  # 既存の指標
  # RSI
  delta = df['Close'].diff()
  gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
  rs = gain / loss
  df['RSI'] = 100 - (100 / (1 + rs))

  # ストキャスティクス
  low_14 = df['Low'].rolling(window=14).min()
  high_14 = df['High'].rolling(window=14).max()
  df['%K'] = (df['Close'] - low_14) / (high_14 - low_14) * 100
  df['%D'] = df['%K'].rolling(window=3).mean()

  # 乖離率
  df['SMA20'] = df['Close'].rolling(window=20).mean()
  df['Deviation'] = (df['Close'] - df['SMA20']) / df['SMA20'] * 100

  # 新しい指標
  # MACD
  df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
  df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
  df['MACD'] = df['EMA12'] - df['EMA26']
  df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

  # モメンタム
  df['Momentum'] = df['Close'] - df['Close'].shift(10)

  # RCI (Range Convergence Index)
  def calculate_rci(data, period):
    def rci_single(x):
      n = len(x)
      ranks = pd.Series(x).rank()
      return 100 * (1 - 6 * sum((ranks - (n+1)/2)**2) / (n * (n**2 - 1)))

    return data.rolling(window=period).apply(rci_single, raw=False)

  try:
    df['RCI'] = calculate_rci(df['Close'], 9)
    # NaN値や無限大値を除去
    df['RCI'] = df['RCI'].replace([np.inf, -np.inf], np.nan)
  except Exception as e:
    print(f"RCIの計算でエラーが発生しました: {e}")
    df['RCI'] = np.nan

  # CCI (Commodity Channel Index)
  typical_price = (df['High'] + df['Low'] + df['Close']) / 3
  df['CCI'] = (typical_price - typical_price.rolling(window=20).mean()) / (0.015 * typical_price.rolling(window=20).std())

  # ウィリアムズ%R
  df['Williams%R'] = (high_14 - df['Close']) / (high_14 - low_14) * -100

  # DMI (Directional Movement Index)
  df['TR'] = np.maximum(df['High'] - df['Low'], 
                        np.maximum(abs(df['High'] - df['Close'].shift(1)), 
                                    abs(df['Low'] - df['Close'].shift(1))))
  df['DMplus'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
                          np.maximum(df['High'] - df['High'].shift(1), 0), 0)
  df['DMminus'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
                            np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)
  df['DIplus'] = 100 * df['DMplus'].rolling(window=14).sum() / df['TR'].rolling(window=14).sum()
  df['DIminus'] = 100 * df['DMminus'].rolling(window=14).sum() / df['TR'].rolling(window=14).sum()
  df['DX'] = 100 * abs(df['DIplus'] - df['DIminus']) / (df['DIplus'] + df['DIminus'])
  df['ADX'] = df['DX'].rolling(window=14).mean()

  return df


def plot_chart(df, indicators):
  fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25])

  # ローソク足チャート
  fig.add_trace(go.Candlestick(x=df['Timestamp'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick'), row=1, col=1)

  # 指標の追加
  row = 2
  for indicator in indicators:
    if indicator == 'RSI':
      fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['RSI'], name='RSI'), row=row, col=1)
    elif indicator == 'Stochastics':
      fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['%K'], name='%K'), row=row, col=1)
      fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['%D'], name='%D'), row=row, col=1)
    elif indicator == 'Deviation':
      fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Deviation'], name='Deviation'), row=row, col=1)
    elif indicator == 'MACD':
      fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['MACD'], name='MACD'), row=row, col=1)
      fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Signal'], name='Signal'), row=row, col=1)
    elif indicator == 'Momentum':
      fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Momentum'], name='Momentum'), row=row, col=1)
    elif indicator == 'RCI':
      fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['RCI'], name='RCI'), row=row, col=1)
    elif indicator == 'CCI':
      fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['CCI'], name='CCI'), row=row, col=1)
    elif indicator == 'Williams%R':
      fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Williams%R'], name='Williams%R'), row=row, col=1)
    elif indicator == 'DMI':
      fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['DIplus'], name='DI+'), row=row, col=1)
      fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['DIminus'], name='DI-'), row=row, col=1)
      fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['ADX'], name='ADX'), row=row, col=1)
    
    row = 3 if row == 2 else 2  # 指標を2行目と3行目に交互に配置

  fig.update_layout(height=1000, title_text="Stock Chart with Indicators")
  fig.update_xaxes(rangeslider_visible=False)
  
  return fig

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


def main():
  if not check_password():
    st.stop()  # Do not continue if check_password is not True.
  st.title("Stock Chart Visualization")

  # Streamlitの入力フィールド
  ticker = st.text_input("銘柄コードを入力してください（半角数字4桁）:")
  if ticker:
    ticker += '.T'
    range_param = '60d'
    interval = '5m'

    try:
      data = fetch_data(ticker, range_param, interval)
      
      chart_data = data['chart']['result'][0]
      quotes = chart_data['indicators']['quote'][0]
      timestamps = chart_data['timestamp']

      df = pd.DataFrame({
          'Timestamp': pd.to_datetime(timestamps, unit='s').tz_localize('UTC').tz_convert('Asia/Tokyo'),
          'Open': quotes['open'],
          'High': quotes['high'],
          'Low': quotes['low'],
          'Close': quotes['close'],
          'Volume': quotes['volume']
      })

      df['Timestamp'] = df['Timestamp'].dt.tz_localize(None)
      df = df.sort_values('Timestamp')

      # 取引時間内のデータのみを抽出
      df = filter_trading_hours(df)

      # NaNを含む行を削除
      df = df.dropna()

      output_filename = f'test.csv'
      df.to_csv(output_filename, index=False)

      # 指標の計算
      try:
        df = calculate_indicators(df)
      except Exception as e:
        st.error(f"指標の計算中にエラーが発生しました: {e}")

      # チェックボックスで表示する指標を選択
      st.sidebar.header("表示する指標を選択")
      indicator_options = ['RSI', 'Stochastics', 'Deviation', 'MACD', 'Momentum', 'RCI', 'CCI', 'Williams%R', 'DMI']
      selected_indicators = st.sidebar.multiselect("指標", indicator_options)

      # チャートの描画
      fig = plot_chart(df, selected_indicators)
      st.plotly_chart(fig)

      # データフレームの表示
      st.dataframe(df)

      # 用語説明
      st.subheader("用語説明 (5分足データ)")

      terms = {
        "Timestamp": "タイムスタンプ：各5分間の区切りを示す日時",
        "Open": "始値：その5分間の開始時点での株価",
        "High": "高値：その5分間で最も高かった株価",
        "Low": "安値：その5分間で最も低かった株価",
        "Close": "終値：その5分間の終了時点での株価",
        "Volume": "出来高：その5分間に取引された株式の総数",
        "RSI": "相対力指数：株価の上昇・下降の勢いを0から100の数値で表す。一般的に、70以上で買われ過ぎ、30以下で売られ過ぎと判断される",
        "%K": "ストキャスティクス%K：直近の株価が、一定期間の価格帯のどの位置にあるかを割合で示す。0%が最安値、100%が最高値を意味する",
        "%D": "ストキャスティクス%D：%Kの3期間の平均値。%Kの動きを滑らかにし、偽のシグナルを減らす役割がある",
        "Deviation": "乖離率：現在の株価が、移動平均線からどれだけ離れているかを割合で示す。正の値は株価が平均を上回り、負の値は下回っていることを示す",
        "MACD": "移動平均収束拡散法：短期と長期の移動平均線の差を表す。株価のトレンドや勢いの変化を捉えるのに使用される",
        "Signal": "MACDシグナル線：MACDの9期間の平均線。MACDとシグナル線の交差が売買のタイミングを示すとされる",
        "Momentum": "モメンタム：現在の株価と一定期間前の株価の差。株価の勢いを数値化したもの",
        "RCI": "順位相関係数：一定期間の株価の動きを、順位相関を使って-100から100の範囲で表す。+80以上で買われ過ぎ、-80以下で売られ過ぎとされる",
        "CCI": "商品チャネル指数：株価が平均的な値からどれだけ乖離しているかを示す。+100を超えると買われ過ぎ、-100を下回ると売られ過ぎの可能性がある",
        "Williams%R": "ウィリアムズ%R：直近の株価が、一定期間の価格帯のどの位置にあるかを0%から-100%で表す。-20%以上で買われ過ぎ、-80%以下で売られ過ぎの可能性がある",
        "DIplus": "上昇方向指数：株価の上昇トレンドの強さを0から100で表す。値が大きいほど上昇トレンドが強いことを示す",
        "DIminus": "下降方向指数：株価の下降トレンドの強さを0から100で表す。値が大きいほど下降トレンドが強いことを示す",
        "ADX": "平均方向性指数：株価のトレンドの強さを0から100で表す。一般的に25以上でトレンドが強いとされ、値が大きいほどトレンドが強いことを示す"
      }

      for term, explanation in terms.items():
        if term in df.columns:
          st.markdown(f"**{term}**: {explanation}")

      # 追加の説明
      st.markdown("""
      ### 5分足データについて

      5分足データは、株価の短期的な動きを分析するのに適しています。各行が5分間の株価の動きを表しており、日中の細かい価格変動を捉えることができます。

      ### 指標の使い方

      - これらの指標は5分ごとに計算されるため、非常に短期的な変動を反映します。
      - 短期トレードやデイトレードを行う際に参考になりますが、ノイズ（意味のない変動）も多く含まれる可能性があります。
      - 複数の時間枠（例：5分足、1時間足、日足）を組み合わせて分析することで、より信頼性の高い判断ができます。
      - 各指標には一長一短があり、相場の状況によって有効性が変わることがあります。
      - 技術的指標だけでなく、出来高の変化や板の状況、ニュースなども考慮して総合的に判断することが重要です。
      - 短期の売買を頻繁に行うと、取引コストが増加するリスクがあります。
      - 投資にはリスクが伴います。これらの指標を使う際は、自己責任で慎重に判断してください。

      5分足データを用いた分析は高度な技術と経験が必要です。初心者の方は、まず長期の時間枠（日足や週足）でのトレードから始めることをおすすめします。また、必要に応じて専門家のアドバイスを受けることも検討してください。
      """)

    except Exception as e:
      st.error(f"データの取得または処理中にエラーが発生しました: {e}")

if __name__ == "__main__":
  main()