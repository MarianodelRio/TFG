import numpy as np
import pandas as pd

def generate_latex_table(data_matrix, variables_predictoras):
    """Generate latex code to create a table 
    with data_matrix as data 
    """
    a, b, c, d, e, f, g, h, i, j, k, l = data_matrix.flatten()
    # Convert to str 
    a, b, c, d, e, f, g, h, i, j, k, l = map(str, [a, b, c, d, e, f, g, h, i, j, k, l])
    latex_code = r"""
            \begin{table}[H]
                \begin{center}
                    \begin{tabular}{c | c | c |}
                    FH - PH & M5P & RF & XGB \\ \hline
                    24-24 & """ + a + """ & """ + e + """ & """ + i + r"""\\
                    24-48 & """ + b + """ & """ + f + """ & """ + j + r"""\\
                    24-72 & """ + c + """ & """ + g + """ & """ + k +r"""\\
                    24-168 & """ + d + """ & """ + h + """ & """ + l + r"""\\
                    \end{tabular}
                    \caption{ """ + variables_predictoras + """}
                \end{center}
            \end{table}
            """

    return latex_code

def generate_tables(file):
    """
    Read csv file and pass to generate_latex_table function data from 12 in 12
    """
    results = pd.read_csv(file, delimiter=";")
    # Get last column of csv
    data_matrix = results.iloc[:, -1]
    data_matrix = np.array(data_matrix)
    
    # Agrupar de 8 en 8
    data_matrix = data_matrix.reshape(-1, 12)
    
    variables_predictoras = ["Precio", 
                             "Precio + Demanda",
                             "Precio + Demanda + Generación",
                             "Precio + Demanda + Generación + Precio Gas",
                             "Precio + Demanda + Generación + Precio Gas + Temperaturas",
                             "Precio + Demanda + Generación + Precio Gas + Temperaturas + Futura demanda",
                             "Precio + Demanda + Generación + Precio Gas + Temperaturas + Futura demanda + Futuro precio gas",
                             "Precio + Demanda + Generación + Precio Gas + Temperaturas + Futura demanda + Futuro precio gas + Futuras temperaturas"]
    
    for i in range(len(data_matrix)):
        data_matrix[i] = np.round(data_matrix[i], 2)
        print(generate_latex_table(data_matrix[i], variables_predictoras[i]))
        print("\n\n")

        

def generate_latex_table2(data_matrix, fh_ph):
    """Generate latex code to create a table 
    with data_matrix as data 
    """
    a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x = data_matrix.flatten()
    # Convert to str 
    a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x = map(str, [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x])
    latex_code = r"""
            \begin{table}[H]
                \begin{center}
                    \begin{tabular}{c | c | c | c |}
                    Variables & M5P & RF & XGB \\ \hline
                    P & """ + a + """ & """ + b + """ & """ + c + r"""\\
                    P+D & """ + d + """ & """ + e + """ & """ + f + r"""\\
                    P+D+G & """ + g + """ & """ + h + """ & """ + i + r"""\\
                    P+D+G+PG & """ + j + """ & """ + k + """ & """ + l + r"""\\
                    P+D+G+PG+T & """ + m + """ & """ + n + """ & """ + o + r"""\\
                    P+D+G+PG+T+FD & """ + p + """ & """ + q + """ & """ + r + r"""\\
                    P+D+G+PG+T+FD+FPG & """ + s + """ & """ + t + """ & """ + u + r"""\\
                    P+D+G+PG+T+FD+FPG+FT & """ + v + """ & """ + w + """ & """ + x + """\\
                    \end{tabular}
                    \caption{ """ + fh_ph + """}
                \end{center}
            \end{table}
            """

    return latex_code

def generate_tables2(file):
    """
    Read csv file and pass to generate_latex_table function data from 12 in 12
    """
    results = pd.read_csv(file, delimiter=";")
    fh_ph_t = ["24-24", "24-48", "24-72", "24-168"]
    fh_ph = [24, 48, 72, 168]
    
    for i in range(len(fh_ph)):
        # Obtener elementos de results que tengan fh_ph[i]
        data_matrix = results[results["PAST_HISTORY"] == fh_ph[i]]
        data_matrix = data_matrix.iloc[:, -1]
        data_matrix = np.array(data_matrix)
        # Redondear a dos decimales 
        data_matrix = np.round(data_matrix, 2)
        print(generate_latex_table2(data_matrix, fh_ph_t[i]))
        print("\n\n")


def calculate_metrics():
    features = [5, 6, 8, 9, 17]
    phs = [24, 48, 72, 168]
    models = ["m5p", "rf", "xgb"]
    url = "../results/Sin variables futuras/Train: 2021-12-01 00:00:00 - 2022-12-01 00:00:00 | Test: 2022-12-01 00:00:00 -  /"
    for feature in features:
        for ph in phs:
            for model in models:
                real = url + str(feature) + "/zscore/" + str(ph) + "/" + model + "/real.npy"
                predicted = url + str(feature) + "/zscore/" + str(ph) + "/" + model + "/predicted.npy"
                
                # Cargar datos
                real = np.load(real)
                predicted = np.load(predicted)
                metrics = ["mae", "mape"]
                print(real.shape)
                print(predicted.shape)
                break


def main(): 
    file = "../results_ml_diciembre.csv"
    generate_tables2(file)


if __name__ == "__main__":
    main()
