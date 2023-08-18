from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd


def multicollinearity_eppsilon_feature(X):
    eppsilon_features=[]

    eliminato = True
    while eliminato:
        max = 0

        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]  # Calcolo il vif
        vif["features"] = X.columns

        for vif_value, feature in zip(vif["VIF Factor"], vif["features"]):
            if vif_value >= 5:
                if vif_value > max:
                    max = vif_value
                    feature_da_rimuovere = feature

        if max > 0:
            eliminato = True
            X = X.drop([feature_da_rimuovere], axis=1)
            eppsilon_features.append(feature_da_rimuovere)
        else:
            eliminato = False

    return eppsilon_features