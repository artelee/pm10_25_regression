import numpy as np
import re

import os
import pandas as pd

def combine_csv_files_from_folder(folder_path):
    """
    주어진 폴더 경로에서 모든 CSV 파일을 읽어와 하나의 데이터프레임으로 합칩니다.
    """
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    data_frames = []
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path)
        data_frames.append(df)

    if data_frames:
        return pd.concat(data_frames, ignore_index=True)
    else:
        return pd.DataFrame() # 빈 데이터프레임 반환

def wide_to_long(df_wide: pd.DataFrame, value_name: str):
    """
    df_wide columns 예:
    date, 측정망, 측정소명, 1시..24시, 지점ID, 지점
    """
    df = df_wide.copy()

    # 시간 컬럼 자동 탐지
    hour_cols = [c for c in df.columns if re.match(r"^\d{1,2}시$", str(c))]
    id_cols = [c for c in df.columns if c not in hour_cols]

    # long 변환
    dfl = df.melt(
        id_vars=id_cols,
        value_vars=hour_cols,
        var_name="hour",
        value_name=value_name
    )

    # 결측 표기 처리 + 숫자화
    dfl[value_name] = dfl[value_name].replace("-", np.nan)
    dfl[value_name] = pd.to_numeric(dfl[value_name], errors="coerce")

    # date + hour -> datetime
    dfl["date"] = pd.to_datetime(dfl["date"])
    dfl["hour"] = dfl["hour"].str.replace("시", "", regex=False).astype(int)

    # 24시 처리(다음날 00시로 넘기기)
    is_24 = dfl["hour"].eq(24)
    dfl.loc[is_24, "hour"] = 0
    dfl.loc[is_24, "date"] = dfl.loc[is_24, "date"] + pd.Timedelta(days=1)

    dfl["datetime"] = dfl["date"] + pd.to_timedelta(dfl["hour"], unit="h")

    # 정렬
    key_cols = ["측정소명", "datetime"] if "측정소명" in dfl.columns else ["datetime"]
    dfl = dfl.sort_values(key_cols).reset_index(drop=True)
    return dfl

def clean_by_station(df_long: pd.DataFrame, value_col: str, station_col="측정소명"):
    out = []
    for st, g in df_long.groupby(station_col):
        x = g.sort_values("datetime").copy().set_index("datetime")

        # 중복 datetime이 있으면 평균
        x = x.groupby(level=0)[value_col].mean().to_frame()

        # 시간축 재구성(빠진 시간 포함)
        full_idx = pd.date_range(x.index.min(), x.index.max(), freq="H")
        x = x.reindex(full_idx)

        # 결측 처리(필요에 맞게 조절)
        x[value_col] = x[value_col].interpolate(method="time", limit=6)
        x[value_col] = x[value_col].ffill().bfill()

        # 이상치 클리핑(1%~99%)
        lo, hi = x[value_col].quantile([0.01, 0.99])
        x[value_col] = x[value_col].clip(lo, hi)

        x = x.reset_index(names="datetime")
        x[station_col] = st
        out.append(x)

    # station_col 외 다른 메타(측정망/지점ID 등)는 merge로 다시 붙이는 게 안전
    return pd.concat(out, ignore_index=True)



def add_features(df):
    d = df.sort_values(["측정소명","datetime"]).copy()
    d["hour"] = d["datetime"].dt.hour
    d["dow"] = d["datetime"].dt.dayofweek
    d["month"] = d["datetime"].dt.month

    for col in ["pm10","pm25"]:
        for lag in [1,2,3,24,48]:
            d[f"{col}_lag_{lag}"] = d.groupby("측정소명")[col].shift(lag)
        for win in [3,6,24]:
            s = d.groupby("측정소명")[col].shift(1)
            d[f"{col}_roll_mean_{win}"] = s.rolling(win).mean().reset_index(level=0, drop=True)
            d[f"{col}_roll_std_{win}"]  = s.rolling(win).std().reset_index(level=0, drop=True)

    return d
