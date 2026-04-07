


from sklearn.decomposition import PCA

    pca = PCA()
    pca.fit(df)

    # Erklärte Varianz anzeigen
    explained_variance = pca.explained_variance_ratio_.cumsum()
    print(f"📊 Erklärte kumulierte Varianz (erste 90 Komponenten):\n{explained_variance[:90]}\n")

    # Wähle die Anzahl der Komponenten für ca. 95% der Varianz
    n_95 = next(i for i, var in enumerate(explained_variance) if var > 0.95) + 1
    n_90 = next(i for i, var in enumerate(explained_variance) if var > 0.90) + 1
    n_80 = next(i for i, var in enumerate(explained_variance) if var > 0.80) + 1
    print(f"Benötigte Anzahl der Komponenten(df: {k}):\n95%\t{n_95} (von {df.shape[1]})\n90%\t{n_90} (von {df.shape[1]})\n80%\t{n_80} (von {df.shape[1]})\n")

######

'' ### df-Reduktion durch PCA (Value, Growth, Moment)

dict_pca_reduced = {
    "pca_value": 19,
                    "momentum_latest": 5,
                   "GARCH": 1,
                    "SEASON": 1,
                    "risk_latest": 7,
                   "ts_latest": 4,
                    "allgemein_latest": 4,
                   "pca_growth": 17,
                   # "funda_static": funda_static_prep,
                  # "funda_ts_like": funda_ts_like,
                    }


for k, n_optimal in dict_pca_reduced.items():
    if k in ("pca_value", "pca_growth"):
      df = pd.read_pickle(f"/content/{k}_scaled.pickle")
    else:
      df = pd.read_parquet(f"/content/{k}_scaled.parquet")
    print(f"\n🔍 Start Analyse für: {k} (Shape: {df.shape})\n")

    # Datensatz reduzieren
    pca_reduced = PCA(n_components=n_optimal)           # n_optimal definieren !!!
    X_pca_reduced = pca_reduced.fit_transform(df)
    df_reduced = pd.DataFrame(X_pca_reduced, columns=[f"PCA{i+1}" for i in range(n_optimal)], index=df.index)

    try:
        if k in ("pca_value", "pca_growth"):
          df_reduced.to_pickle(f"/content/{k}_reduced.pickle")
        else:
          df_reduced.to_parquet(f"/content/{k}_reduced.parquet")  #             engine="fastparquet")
        print(f"💾 Speicherung von '{k}_reduced' war erfolgreich.")
    except Exception as e:
        print(f"❌ Fehler beim Speichern von {k}_reduced:\n{e}")

    # Loadings (Feature Contribution je Komponente)
    loadings = pd.DataFrame(
        pca_reduced.components_.T,
        index=df.columns,
        columns=[f"PCA{i+1}" for i in range(n_optimal)])

    # ✅ Feature-Scores: gewichtete Summe aus Absolutwerten der Loadings * Varianzanteil
    abs_loadings = np.abs(pca_reduced.components_[:n_optimal, :])  # Form: (n_optimal, n_features)
    variance_weights = pca_reduced.explained_variance_ratio_[:n_optimal]  # Länge: n_optimal
    feature_scores = abs_loadings.T @ variance_weights              # Shape: (n_features,)
    feature_scores_series = pd.Series(feature_scores, index=df.columns, name="PCA_Score").sort_values(ascending=False)

    try:
        loadings.to_parquet(f"/content/{k}_loadings.parquet")  #           engine="fastparquet")
        feature_scores_series.to_frame().to_parquet(f"/content/{k}_feature_scores.parquet")   #   engine="fastparquet")
        print(f"💾 Speicherung von '{k}_loadings' & '{k}_feature_scores' war erfolgreich.\n")
    except Exception as e:
        print(f"❌ Fehler beim Speichern von {k}_loadings bzw. ~_feature_scores:\n{e}\n")
    
