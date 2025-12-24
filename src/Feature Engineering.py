### FEATURE ENGINEERING
scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test


df["Amount_scaled"] = scaler.fit_transform(df[["Amount_log"]])
df["Time_scaled"] = scaler.fit_transform(df[["Time"]])

df["Time_bin"] = pd.cut(
    df["Time"],
    bins=4,
    labels=["Very Early", "Early", "Mid", "Late"])
df = pd.get_dummies(df, columns=["Time_bin"], drop_first=True)

df["High_Amount"] = (df["Amount"] > 200).astype(int)

X = df.drop(columns=["Class"])
y = df["Class"]

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(10)

selector = SelectKBest(score_func=f_classif, k=15)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
