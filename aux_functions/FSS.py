import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    LabelEncoder, 
    MinMaxScaler, 
    KBinsDiscretizer, 
    StandardScaler, 
    OneHotEncoder, 
    OrdinalEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.model_selection import StratifiedKFold
# --- MÉTODOS ELIMINADOS ---
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import LinearSVC 
# from sklearn.feature_selection import RFE, SelectFromModel
# NUEVAS IMPORTACIONES
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import kruskal, entropy, norm
# --- MÉTODOS ELIMINADOS ---
# from scipy.stats import mannwhitneyu, ttest_ind
import warnings

# Ignorar advertencias para una salida más limpia
warnings.filterwarnings('ignore')

def load_data(filepath):
    """
    Carga y preprocesa los datos iniciales.
    Identifica columnas numéricas y categóricas.
    Codifica la variable objetivo (target).
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Datos cargados exitosamente desde {filepath}. Shape: {df.shape}")
        
        # Asume que la última columna es el target
        target_column = df.columns[-1]
        print(f"Variable objetivo (target) identificada: '{target_column}'")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Forzar tipos de columnas categóricas (ejemplos)
        # Ajusta esto según tus columnas categóricas reales
        potential_cat_cols = ['sector', 'state', 'In_SP500', 'In_NASDAQ', 'is_Insolvent', 'has_benefits']
        for col in potential_cat_cols:
            if col in X.columns:
                X[col] = X[col].astype('category')
                
        # Identificar tipos de columnas
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
        
        print(f"Columnas categóricas detectadas ({len(categorical_cols)}): {categorical_cols}")
        print(f"Columnas numéricas detectadas ({len(numerical_cols)}): {numerical_cols}")

        # Codificar el target
        le = LabelEncoder()
        y = le.fit_transform(y)
        n_classes = len(le.classes_)
        print(f"Target codificado. Clases ({n_classes}): {le.classes_}")
        
        return X, y, numerical_cols, categorical_cols, le, n_classes

    except FileNotFoundError:
        print(f"Error: El archivo '{filepath}' no fue encontrado.")
        return None, None, None, None, None, None
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return None, None, None, None, None, None

def _store_results(results_list, method_name, scores_df):
    """Función helper para almacenar resultados en un formato unificado."""
    scores_df = scores_df.sort_values(by='score', ascending=False)
    for i, row in enumerate(scores_df.itertuples()):
        results_list.append({
            'method': method_name,
            'feature': row.feature,
            'score': row.score,
            'rank': i + 1
        })

def run_fss_filters(X, y, numerical_cols, categorical_cols, n_classes):
    """
    Ejecuta una batería de métodos de selección de características de tipo FILTRO.
    Aplica el preprocesado adecuado (escalado, discretización) para cada método.
    """
    all_filter_results = []
    
    # --- Definir Preprocesadores ---
    # Preprocesador para features continuas (escaladas)
    continuous_preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Preprocesador para features discretas (codificadas ordinalmente)
    discrete_preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # --- PREPROCESADOR ELIMINADO ---
    # El 'hybrid_preprocessor' que usaba k-NN (MI) y 'fd' (H(X))
    # ha sido reemplazado por 'mi_preprocessor' (discretización primero).
    # hybrid_preprocessor = ColumnTransformer(transformers=[
    #     ('num', Pipeline(steps=[
    #         ('imputer', SimpleImputer(strategy='median')),
    #         ('scaler', StandardScaler())
    #     ]), numerical_cols),
    #     ('cat', discrete_preprocessor, categorical_cols)
    # ], remainder='drop')
    
    # Preprocesador para TNoM (no necesita escalado)
    tnom_preprocessor = ColumnTransformer(transformers=[
        ('num', SimpleImputer(strategy='median'), numerical_cols),
        ('cat', discrete_preprocessor, categorical_cols)
    ], remainder='drop')

    # Preprocesador para features BINARIAS (usado por OddsRatio, BNS)
    # Discretiza numéricas en 2 bins, codifica categóricas.
    binary_preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('discretizer', KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile', subsample=None, random_state=42)) 
        ]), numerical_cols),
        ('cat', discrete_preprocessor, categorical_cols)
    ], remainder='drop')

    # Preprocesador para Chi2 (Discretiza en 5 bins)
    discrete_all_preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            # Usamos 'quantile' para manejar outliers
            ('discretizer', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile', subsample=None, random_state=42)) 
        ]), numerical_cols),
        ('cat', discrete_preprocessor, categorical_cols) # Re-usa el codificador ordinal
    ], remainder='drop')

    # --- NUEVO PREPROCESADOR (para MI, GainRatio, SU) ---
    # Es idéntico a 'discrete_all_preprocessor'.
    # Discretiza numéricas en 5 bins y codifica categóricas.
    # Esto asegura que MI y H(X) se calculen sobre los mismos datos (discretos).
    mi_preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('discretizer', KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile', subsample=None, random_state=42)) 
        ]), numerical_cols),
        ('cat', discrete_preprocessor, categorical_cols)
    ], remainder='drop')


    # --- 1.1 MÉTODOS PARAMÉTRICOS (CONTINUOS) ---
    print("Procesando 1.1: Paramétricos (Continuos)...")
    try:
        X_num_scaled = continuous_preprocessor.fit_transform(X[numerical_cols], y)
        
        # --- t-test family --- (ELIMINADO)
        # if n_classes == 2: ...

        # --- ANOVA / Between-groups to within-groups sum of squares ---
        # f_classif (F-statistic) es la implementación de ANOVA, que es
        # literalmente (varianza between-groups) / (varianza within-groups).
        print("   - Ejecutando ANOVA (f_classif)...")
        selector_anova = SelectKBest(f_classif, k='all')
        selector_anova.fit(X_num_scaled, y)
        anova_results = pd.DataFrame({
            'feature': [f"num__{col}" for col in numerical_cols],
            'score': selector_anova.scores_
        })
        _store_results(all_filter_results, "ANOVA (f_classif)", anova_results)
        
    except Exception as e:
        print(f"Error en Paramétricos (Continuos): {e}")


    # --- 1.2 MÉTODOS MODEL-FREE (CONTINUOS) ---
    print("Procesando 1.2: Model-Free (Continuos)...")
    try:
        X_num_imputed = SimpleImputer(strategy='median').fit_transform(X[numerical_cols])
        
        # --- Mann-Whitney test --- (ELIMINADO)

        # --- Kruskal-Wallis test ---
        if n_classes > 2:
            print("   - Ejecutando Kruskal-Wallis...")
            scores_kw = []
            for i, col in enumerate(numerical_cols):
                args = [X_num_imputed[y == k, i] for k in np.unique(y)]
                
                # --- INICIO DE LA CORRECCIÓN ---
                # Comprobar si algún grupo tiene valores idénticos,
                # lo que causa el error "All numbers are identical" en kruskal
                
                all_groups_valid = True
                if len(args) < 2: # Necesita al menos 2 grupos
                    all_groups_valid = False
                    
                for group in args:
                    # Si el grupo está vacío o tiene < 2 valores únicos (es constante)
                    if len(group) == 0 or len(np.unique(group)) < 2:
                        all_groups_valid = False
                        break
                
                if all_groups_valid:
                    try:
                        score, pval = kruskal(*args, nan_policy='omit')
                        # np.abs(score) es redundante ya que kruskal H-statistic es >= 0
                        scores_kw.append({'feature': f"num__{col}", 'score': score})
                    except ValueError as e:
                        # Capturar otros errores, ej. "must contain at least one non-missing value"
                        # print(f"   - Omitiendo Kruskal-Wallis para '{col}': {e}")
                        scores_kw.append({'feature': f"num__{col}", 'score': 0})
                else:
                    # Si un grupo es constante, el test no es aplicable. Asignar score 0.
                    # print(f"   - Omitiendo Kruskal-Wallis para '{col}': un grupo tiene valores idénticos.")
                    scores_kw.append({'feature': f"num__{col}", 'score': 0})
                # --- FIN DE LA CORRECCIÓN ---
                
            _store_results(all_filter_results, "Kruskal-Wallis", pd.DataFrame(scores_kw))
        
        # --- Scores based on estimating density functions --- (AÑADIDO)
        print("   - Ejecutando Density_Scores (MI)...")
        # mutual_info_classif con discrete_features=False usa k-NN
        # para estimar la densidad, lo cual es el método solicitado.
        # Es CRÍTICO usar datos escalados aquí.
        X_num_scaled_density = continuous_preprocessor.fit_transform(X[numerical_cols], y)
        
        mi_density_scores = mutual_info_classif(X_num_scaled_density, y, discrete_features=False, random_state=42)
        mi_density_results = pd.DataFrame({
            'feature': [f"num__{col}" for col in numerical_cols],
            'score': mi_density_scores
        })
        _store_results(all_filter_results, "Density_Scores (MI)", mi_density_results)
        
    except Exception as e:
        print(f"Error en Model-Free (Continuos): {e}")

    # --- 1.1 MÉTODOS PARAMÉTRICOS (DISCRETOS) ---
    print("Procesando 1.1: Paramétricos (Discretos)...")
    try:
        # Preprocesamos TODOS los datos a formato binario/discreto
        X_binary_processed = binary_preprocessor.fit_transform(X, y)
        binary_feature_names = binary_preprocessor.get_feature_names_out()

        # --- Chi-squared ---
        # (Se aplica a TODAS las features (num discretizadas + cat))
        print("   - Ejecutando Chi-cuadrado...")
        X_all_discretized = discrete_all_preprocessor.fit_transform(X, y)
        all_discrete_features = discrete_all_preprocessor.get_feature_names_out()
        
        # Asegurarse de que no haya valores negativos (MinMaxScaler)
        X_all_discretized_non_neg = MinMaxScaler().fit_transform(X_all_discretized)
        
        selector_chi2 = SelectKBest(chi2, k='all')
        selector_chi2.fit(X_all_discretized_non_neg, y)
        chi2_results = pd.DataFrame({
            'feature': all_discrete_features,
            'score': selector_chi2.scores_
        })
        _store_results(all_filter_results, "Chi-cuadrado", chi2_results)

        # --- Odds Ratio (OR) y Bi-Normal Separation (BNS) ---
        if n_classes == 2:
            print("   - Ejecutando Odds_Ratio y BNS (target binario)...")
            scores_or = []
            scores_bns = []
            for i, feature_name in enumerate(binary_feature_names):
                # (a+c) = total_pos, (b+d) = total_neg
                total_pos = (y == 1).sum()
                total_neg = (y == 0).sum()
                
                # Feature presente (X=1)
                feature_present = (X_binary_processed[:, i] == 1)
                
                # a = TP (feature=1 y target=1)
                a = (feature_present & (y == 1)).sum() 
                # b = FP (feature=1 y target=0)
                b = (feature_present & (y == 0)).sum()
                
                # c = FN (feature=0 y target=1)
                c = total_pos - a
                # d = TN (feature=0 y target=0)
                d = total_neg - b

                # Odds Ratio (con corrección Haldane-Anscombe)
                a_corr = a + 0.5
                b_corr = b + 0.5
                c_corr = c + 0.5
                d_corr = d + 0.5
                odds_ratio = (a_corr * d_corr) / (b_corr * c_corr)
                scores_or.append({'feature': feature_name, 'score': np.log(odds_ratio) if odds_ratio > 0 else 0}) # Usamos log(OR)

                # BNS
                tpr = a / (a + c) # TP / (TP + FN)
                fpr = b / (b + d) # FP / (FP + TN)
                # Clip para evitar -inf/inf en ppf
                tpr_clipped = np.clip(tpr, 1e-5, 1 - 1e-5)
                fpr_clipped = np.clip(fpr, 1e-5, 1 - 1e-5)
                bns_score = np.abs(norm.ppf(tpr_clipped) - norm.ppf(fpr_clipped))
                scores_bns.append({'feature': feature_name, 'score': bns_score})

            _store_results(all_filter_results, "Odds_Ratio (Log)", pd.DataFrame(scores_or))
            _store_results(all_filter_results, "Bi-Normal_Separation", pd.DataFrame(scores_bns))

        # --- Mutual information, Gain ratio, Symmetrical uncertainty ---
        # --- INICIO DE LA CORRECCIÓN ---
        # El método anterior (k-NN para MI, 'fd' binning para H(X)) era
        # inconsistente y penalizaba a las features continuas.
        
        # NUEVO MÉTODO: Discretizar todo primero (5 bins) y luego
        # calcular MI y H(X) sobre los datos ya discretizados.
        
        print("   - Ejecutando MI, GainRatio, SU (numéricas discretizadas + categóricas)...")
        X_mi_processed = mi_preprocessor.fit_transform(X, y)
        mi_feature_names = mi_preprocessor.get_feature_names_out()
        
        # 1. Mutual Information (MI)
        # Ahora usamos discrete_features=True porque todo está discretizado.
        mi_scores = mutual_info_classif(X_mi_processed, y, discrete_features=True, random_state=42)
        mi_results = pd.DataFrame({'feature': mi_feature_names, 'score': mi_scores})
        _store_results(all_filter_results, "Mutual_Information", mi_results)
        
        # 2. Gain Ratio (GR) y Symmetrical Uncertainty (SU)
        scores_gr = []
        scores_su = []
        H_y = entropy(np.bincount(y), base=2) # Entropía del target
        
        for i, feature_name in enumerate(mi_feature_names):
            X_col = X_mi_processed[:, i]
            mi_score = mi_scores[i]
            
            # H(X) - Entropía de la feature
            # Como todo es discreto, solo necesitamos bincount
            counts = np.bincount(X_col.astype(int)[X_col >= 0])
                
            H_x = entropy(counts[counts > 0], base=2)
            
            # Gain Ratio = MI(X,Y) / H(X)
            gr_score = 0 if H_x == 0 else mi_score / H_x
            scores_gr.append({'feature': feature_name, 'score': gr_score})
            
            # Symmetrical Uncertainty = 2 * MI(X,Y) / (H(X) + H(Y))
            su_score = 0 if (H_x + H_y) == 0 else (2 * mi_score) / (H_x + H_y)
            scores_su.append({'feature': feature_name, 'score': su_score})
        
        # Mover las llamadas fuera del bucle:
        _store_results(all_filter_results, "Gain_Ratio", pd.DataFrame(scores_gr))
        _store_results(all_filter_results, "Symmetrical_Uncertainty", pd.DataFrame(scores_su))
        # --- FIN DE LA CORRECCIÓN ---

    except Exception as e:
        print(f"Error en Paramétricos (Discretos): {e}")

    # --- 1.2 MÉTODOS MODEL-FREE (DISCRETOS) ---
    print("Procesando 1.2: Model-Free (Discretos)...")
    try:
        # --- Threshold number of misclassification (TNoM) / P-metric ---
        # Esto es equivalente a ajustar un Decision Tree de profundidad 1 (un "stub")
        print("   - Ejecutando TNoM (DT Stub)...")
        dt_stub = DecisionTreeClassifier(max_depth=1, random_state=42)
        
        # --- CAMBIO AQUÍ: Usa el tnom_preprocessor (sin scaler) ---
        X_tnom_processed = tnom_preprocessor.fit_transform(X, y)
        tnom_feature_names = tnom_preprocessor.get_feature_names_out()

        scores_tnom = []
        for i, feature_name in enumerate(tnom_feature_names):
            # Entrenar árbol solo con esta feature
            dt_stub.fit(X_tnom_processed[:, [i]], y) 

            # --- CORRECCIÓN ---
            # feature_importances_ de un árbol con una sola feature siempre será 1.0.
            # Usamos el 'score' (accuracy) del stub en los datos de train como
            # un mejor proxy para TNoM (qué tan bien clasifica esa feature por sí sola).
            tnom_score = dt_stub.score(X_tnom_processed[:, [i]], y)
            
            scores_tnom.append({'feature': feature_name, 'score': tnom_score})
            
        _store_results(all_filter_results, "TNoM (DT Stub)", pd.DataFrame(scores_tnom))

    except Exception as e:
        print(f"Error en Model-Free (Discretos): {e}")

    print("Completado el análisis de Filtros.")
    return all_filter_results


# --- FUNCIÓN ELIMINADA ---
# def run_fss_wrapper_embedded(...):
# ... (todo el contenido de la función ha sido eliminado) ...


def show_results_table(all_results):
    """
    Imprime los resultados del Top 15 para cada método.
    """
    print("\n" + "="*80)
    print(" TOP 15 FEATURES POR MÉTODO DE SELECCIÓN ")
    print("="*80)
    
    if not all_results:
        print("No se generaron resultados.")
        return

    df = pd.DataFrame(all_results)
    
    # Manejar scores infinitos o NaNs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Crear Ranks (más alto score = mejor)
    df['rank'] = df.groupby('method')['score'].rank(ascending=False, method='first')
    
    # Obtener la lista de métodos únicos
    methods = df['method'].unique()
    
    # Iterar sobre cada método y mostrar su Top 15
    for method in sorted(methods):
        print("\n" + "-"*60)
        print(f" MÉTODO: {method}")
        print("-"*60)
        
        # Filtrar, ordenar y tomar el top 15
        top_15 = df[df['method'] == method].sort_values(by='rank').head(15)
        
        # Seleccionar columnas para mostrar
        top_15_display = top_15[['rank', 'feature', 'score']]
        
        # Imprimir con un formato limpio
        with pd.option_context('display.max_rows', 15, 'display.max_columns', None, 'display.width', 1000):
            print(top_15_display.to_string(index=False, formatters={'rank': '{:,.0f}'.format, 'score': '{:.4f}'.format}))


def show_cv_template(X, y):
    """
    Muestra una plantilla de cómo se debe implementar
    la validación cruzada con FSS anidado.
    """
    print("\n" + "="*80)
    print(" PLANTILLA DE VALIDACIÓN CRUZADA (10 Folds) ")
    print("="*80)
    print("El FSS y el preprocesado deben realizarse DENTRO de este bucle,")
    print("usando solo los datos de 'X_train' de cada fold.")
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold+1}/10 ---")
        
        # IMPORTANTE: X es el DataFrame de pandas crudo
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        print(f"Train: X={X_train.shape}, y={y_train.shape}")
        print(f"Test:  X={X_test.shape}, y={y_test.shape}")
        
        # --- EJEMPLO DE PIPELINE DENTRO DEL BUCLE ---
        # 1. Definir tu preprocesador (el mismo de 'run_fss_filters')
        #    (ej. 'mi_preprocessor')
        
        # 2. Definir tu selector de features (ej. SelectKBest)
        #    selector = SelectKBest(mutual_info_classif, k=15)
        
        # 3. Definir tu modelo
        #    model = LogisticRegression()
        
        # 4. Crear el Pipeline
        #    pipe = Pipeline([
        #        ('preprocessor', mi_preprocessor),
        #        ('selector', selector),
        #        ('model', model)
        #    ])
        
        # 5. Entrenar
        #    pipe.fit(X_train, y_train)
        
        # 6. Evaluar
        #    accuracy = pipe.score(X_test, y_test)
        #    print(f"Accuracy (Fold {fold+1}): {accuracy:.4f}")
        
        pass # Fin del bucle


if __name__ == "__main__":
    # --- ¡CORRECCIÓN IMPORTANTE! ---
    # Este script ahora espera el archivo de datos CRUDOS (sin estandarizar).
    # Asegúrate de que este CSV contenga los datos *SIN ESTANDARIZAR*
    # y que las columnas categóricas (sector, state, etc.) estén
    # como texto o categorías, no como números (ej. One-Hot).
    FILEPATH = r"C:\Users\SergioBeamonteGonzal\Documentos Locales\MASTER\Machine Learning\1-ML_Model\full_main_financial_metrics.csv" 
    
    # 1. Cargar los datos
    print("Iniciando carga de datos...")
    X, y, numerical_cols, categorical_cols, le, n_classes = load_data(FILEPATH)
    
    if X is not None:
        all_fss_results = []
        
        # 2. Ejecutar los algoritmos de FSS de FILTRO
        print("\nIniciando análisis de Filtros FSS...")
        filter_results = run_fss_filters(X, y, numerical_cols, categorical_cols, n_classes)
        all_fss_results.extend(filter_results)
        
        # 3. Ejecutar los algoritmos de FSS de WRAPPER y EMBEDDED (ELIMINADO)
        # print("\nIniciando análisis Wrapper y Embedded FSS...")
        # wrapper_results = run_fss_wrapper_embedded(X, y, numerical_cols, categorical_cols, n_classes)
        # all_fss_results.extend(wrapper_results)
        
        # 4. Mostrar la tabla de resultados final
        show_results_table(all_fss_results)
        
        # 5. Mostrar la plantilla de CV
        show_cv_template(X, y)
            
    else:
        print("No se pudieron cargar los datos. Saliendo.")

