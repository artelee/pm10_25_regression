from utils import *
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def main():
    pm10_folder_path = './10/'
    pm25_folder_path = './2.5/'

    pm10_df = combine_csv_files_from_folder(pm10_folder_path)
    pm25_df = combine_csv_files_from_folder(pm25_folder_path)

    pm25_long = wide_to_long(pm25_df, "pm25")
    pm10_long = wide_to_long(pm10_df, "pm10")

    pm25_clean = clean_by_station(pm25_long[["측정소명","datetime","pm25"]], "pm25")
    pm10_clean = clean_by_station(pm10_long[["측정소명","datetime","pm10"]], "pm10")

    meta_cols = [c for c in pm25_long.columns if c not in ["date","hour","datetime","pm25","측정소명"] and not re.match(r"^\d{1,2}시$", str(c))]
    meta_pm25 = pm25_long[meta_cols + ["측정소명"]].drop_duplicates(subset=["측정소명"], keep="first")

    df = pm25_clean.merge(pm10_clean, on=["측정소명","datetime"], how="inner")
    df = df.merge(meta_pm25, on="측정소명", how="left")

    df = df.sort_values(["측정소명","datetime"]).reset_index(drop=True)
    print(df.head())

    df_feat = add_features(df)


    H = 1
    tmp = df_feat.sort_values(["측정소명","datetime"]).copy()
    tmp["y_pm10"] = tmp.groupby("측정소명")["pm10"].shift(-H)
    tmp["y_pm25"] = tmp.groupby("측정소명")["pm25"].shift(-H)
    tmp = tmp.dropna().reset_index(drop=True)

    cat_cols = ["측정소명", "측정망"]
    drop_cols = ["datetime","pm10","pm25","y_pm10","y_pm25"]
    num_cols = [c for c in tmp.columns if c not in drop_cols + cat_cols]

    X = tmp[num_cols + cat_cols]
    X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)
    Y = tmp[["y_pm10","y_pm25"]]

    tmp["rank"] = tmp.groupby("측정소명")["datetime"].rank(pct=True)
    train_idx = tmp["rank"] <= 0.8
    test_idx  = tmp["rank"] > 0.8

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    Y_train, Y_test = Y.loc[train_idx], Y.loc[test_idx]

    base = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=400, random_state=42)
    model = MultiOutputRegressor(base)
    model.fit(X_train, Y_train)

    pred = model.predict(X_test)

    print("PM10 MAE/RMSE:",
          mean_absolute_error(Y_test["y_pm10"], pred[:,0]),
          mean_squared_error(Y_test["y_pm10"], pred[:,0])**0.5)

    print("PM25 MAE/RMSE:",
          mean_absolute_error(Y_test["y_pm25"], pred[:,1]),
          mean_squared_error(Y_test["y_pm25"], pred[:,1])**0.5)