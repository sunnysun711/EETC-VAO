import json
import os
import re
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')


# 1. Solution info is referred to as data from json files,
#    with Status, MIPGap, Runtime, SolCount, and ... included.
# 2. Results are referred to as the text output in the logging files,
#    with e, pi, z, constructionCost, T_u, and ... included.

def parse_log_file(log_file_path: str) -> dict[str, float]:
    """parse from log file to a dict of brief results of the cases"""
    # Define a dictionary to store the extracted values
    results = {
        'e': None,
        'pi': None,
        'z1': None,
        'total_tunnel_number': None,
        'constructionCost': None,
        'T_u': None,
        'tractionCostPerTrain_u': None,
        'auxiliaryCostPerTrain_u': None,
        'No.TrainsRunning_u': None,
        'T_d': None,
        'tractionCostPerTrain_d': None,
        'auxiliaryCostPerTrain_d': None,
        'No.TrainsRunning_d': None,
        'operationCostTotal': None
    }

    # Define patterns to match the desired lines
    patterns = {
        'e': r'"e":\s*(\{.*?\})',
        'pi': r'"pi":\s*(\{.*?\})',
        'z1': r'"z1":\s*(\{.*?\})',
        'total_tunnel_number': r'total tunnel number:\s*([\d\.]+)',
        'constructionCost': r'constructionCost:\s*([\d\.]+)',
        'T_u': r'T_u:\s*([\d\.]+)',
        'tractionCostPerTrain_u': r'tractionCostPerTrain_u:\s*([\d\.]+)',
        'auxiliaryCostPerTrain_u': r'auxiliaryCostPerTrain_u:\s*([\d\.]+)',
        'No.TrainsRunning_u': r'No.TrainsRunning_u:\s*([\d]+)',
        'T_d': r'T_d:\s*([\d\.]+)',
        'tractionCostPerTrain_d': r'tractionCostPerTrain_d:\s*([\d\.]+)',
        'auxiliaryCostPerTrain_d': r'auxiliaryCostPerTrain_d:\s*([\d\.]+)',
        'No.TrainsRunning_d': r'No.TrainsRunning_d:\s*([\d]+)',
        'operationCostTotal': r'operationCostTotal:\s*([\d\.]+)'
    }

    # Open the log file and read line by line
    try:
        with open(log_file_path, 'r') as file:
            for line in file:
                for key, pattern in patterns.items():
                    match = re.search(pattern, line)
                    if match:
                        results[key] = float(match.group(1))
    except FileNotFoundError:
        print(f"Error: The file at {log_file_path} does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return results


def get_tc_log_results_from_folder(folder_path: str):
    """get train control brief results from log files in the specified folder."""
    root = fr"E:\OneDrive\Documents\00-MyResearch\20230524_VAO-EETC\EETC-VAO_202404\Cases\{folder_path}"
    model_folders = os.listdir(root)
    result_dic: dict[tuple[str, str], dict[str, float]] = {}
    for model_folder in model_folders:
        case, train, model_name = model_folder.split("__")
        print(case, train, model_name)
        log_file: str = os.path.join(root, model_folder, f"{case}__{train}__{model_name}.log")
        result_dic[case, train] = parse_log_file(log_file)
    print(result_dic)
    df = pd.DataFrame(result_dic).T.reset_index()
    print(df.head(5))
    df.to_csv(f"{folder_path}_results.csv", index=True)
    return


def get_eetcvao_log_results_from_folder(folder_path: str):
    """get EETC-VAO brief results from log files in the specified folder."""
    root = fr"E:\OneDrive\Documents\00-MyResearch\20230524_VAO-EETC\EETC-VAO_202404\Cases\{folder_path}"
    model_folders = os.listdir(root)
    result_dic: dict[tuple[str, str], dict[str, Any]] = {}
    for model_folder in model_folders:
        case, train, model_name = model_folder.split("__")
        print(case, train, model_name)
        log_file: str = os.path.join(root, model_folder, f"{model_folder}.log")
        parsed_log: dict = parse_log_file(log_file)
        result_dic[case, train] = {
            "constructionCost": parsed_log["constructionCost"],
            "operationCostTotal": parsed_log["operationCostTotal"]
        }
    print(result_dic)
    df = pd.DataFrame(result_dic).T.reset_index()
    df.to_csv(f"Cases\\__results_analysis\\{folder_path}_results.csv", index=True)
    return


def load_solution_from_file(json_file_path: str) -> dict[str, Any]:
    """Load solution information from a JSON file."""
    json_data = json.load(open(json_file_path, 'r', encoding='utf-8'))
    return json_data


def load_solution_from_case(folder_root: str, model_folder: str) -> dict[str, Any]:
    """
    Load solution information from a specific case folder.
    :param folder_root: first level full folder path, for e.g. fr"E:\...\EETC-VAO_202404\Cases\eetc-vao".
    :param model_folder: second level model folder path, for e.g. "gd1__HXD2__eetc-vao"
    :return:
    """
    case, train, model_name = model_folder.split("__")
    print(case, train, model_name)
    solution_json_file: str = os.path.join(folder_root, model_folder, f"{case}__{train}__{model_name}.json")
    solution: dict[str, Any] = load_solution_from_file(solution_json_file)
    return solution


def save_folder_solution_to_csv(folder_path: str):
    """Extract solution information from JSON files in the specified folder and save to a CSV."""
    root = fr"E:\OneDrive\Documents\00-MyResearch\20230524_VAO-EETC\EETC-VAO_202404\Cases\{folder_path}"
    model_folders = os.listdir(root)
    result_dic: dict[tuple[str, str], dict[str, float]] = {}
    for model_folder in model_folders:
        case, train, model_name = model_folder.split("__")
        json_data = load_solution_from_case(root, model_folder)

        this_dict: dict[str, float] = {
            "Status": json_data['SolutionInfo']['Status'],
            "MIPGap": json_data['SolutionInfo']['MIPGap'],
            "Runtime": json_data['SolutionInfo']['Runtime'],
            "ObjVal": json_data['SolutionInfo']['ObjVal'],
        }
        result_dic[case, train] = this_dict
    print(result_dic)
    df = pd.DataFrame(result_dic).T.reset_index()
    print(df.head(5))
    df.to_csv(f"{folder_path}_solution_info.csv", index=True)
    pass


def extract_variables_from_solution(solution_data: dict[str, Any]) -> dict[str, Any]:
    """get variables from solution info from json data"""
    solution_vars: list[dict[str, Any]] = solution_data['Vars']
    variables: dict[str, float] = {}
    for solution_var_dic in solution_vars:
        variables[solution_var_dic['VarName']] = solution_var_dic['X']
    return variables


def extract_specific_variable(named_dict: dict[str, Any], variable_name: str) -> dict[int, Any]:
    """
    Extract specific variable values from a named dictionary.
    :param named_dict: for e.g. {'e[0]': 198, 'e[1]': 198, 'e[2]': 196.00001666666668, ...}
    :param variable_name: for e.g. "e", "E_k^u",
        should be included in:
        ['c^d', 'f_bra^d', 'C^{tn,e}', 'phi^u', 'f_PWA_v^d', 'kappa^u', 'omega_tn^u',
        'omega_0^u', 'omega_r^u', 'omega_r^d', 'f_tra^u', 'phi^d', 'omega_i^u', 'omega_0^d',
        't^d', 't^u', 'f_bra^u', 'f_PWA_v^u', 'C', 'pi', 'z4', 'omega_tn^d', 'c^u', 'E_k^u',
        'z1', 'z5', 'f_tra^d', 'E_k^d', 'u^u', 'u^d', 'omega_i^d', 'e', 'gamma', 'kappa^d',
        'z2']
    :return:
    """
    vrb: dict = {}
    for k, v in named_dict.items():
        var_name, index = k[:-1].split("[")
        if var_name == variable_name:
            vrb[int(index)] = v
    return vrb


def fill_missing_values(variable: dict[int, Any], id_range) -> list:
    """Fill missing variable values with zeros in the specified index range."""
    filled_variable: list = []
    for i in id_range:
        if i in variable:
            filled_variable.append(variable[i])
        else:
            filled_variable.append(0)
    return filled_variable


def plot_tc_results(variables: dict[str, Any], ds: int = 100, bottom_left_loc=0, save_path: str = "") -> None:
    """Plot train control results using the extracted variable data."""
    E_k__u = extract_specific_variable(variables, variable_name="E_k^u")
    E_k__d = extract_specific_variable(variables, variable_name="E_k^d")
    u__u = extract_specific_variable(variables, variable_name="u^u")
    u__d = extract_specific_variable(variables, variable_name="u^d")
    e = extract_specific_variable(variables, variable_name="e")

    S = list(e.keys())[-1] - 1

    # fill up zero values
    filled_E_k__u: list = fill_missing_values(E_k__u, range(1, S + 2))
    filled_E_k__d: list = fill_missing_values(E_k__d, range(1, S + 2))
    filled_u__u: list = fill_missing_values(u__u, range(1, S + 1))
    filled_u__d: list = fill_missing_values(u__d, range(1, S + 1))

    filled_v__u = np.sqrt(filled_E_k__u)
    filled_v__d = np.sqrt(filled_E_k__d)

    distances = np.arange(0, S + 1) * ds + bottom_left_loc
    plt.figure(figsize=(8, 4))

    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212, sharex=ax1)

    marker_size = 0

    ax1.plot(distances, filled_v__d, marker='o', markersize=marker_size, ls="-", label="v_d")
    ax1.plot(distances, filled_v__u, marker='x', markersize=marker_size, ls="-", label="v_u")

    ax2.plot(distances[:-1], filled_u__d, marker='o', markersize=marker_size, ls="-", label="u_d")
    ax2.plot(distances[:-1], filled_u__u, marker='x', markersize=marker_size, ls="-", label="u_u")

    ax1.legend(fontsize="small")
    ax1.set_ylabel("Velocity (km/h)", fontsize="small")
    ax2.legend(fontsize="small")
    ax2.set_xlabel("Horizontal location (m)", fontsize="small")
    ax2.set_ylabel("Unit control force (N/kN)", fontsize="small")
    plt.tight_layout()

    # plt.show()

    plt.savefig(os.path.join(save_path, "tc_result.pdf"), dpi=600)

    return


def main():
    # get_tc_results_from_folder(folder_path="eetc_tcVI")
    # get_solution_info_from_folder(folder_path="eetc_tcVI")
    # get_eetcvao_results_from_folder("eetc-vao_VI")

    solution = load_solution_from_case(
        folder_root=r"E:\OneDrive\Documents\00-MyResearch\20230524_VAO-EETC\EETC-VAO_202404\Cases\eetc-vao_LC_VI_tcVI",
        model_folder="gd_gaoyan__CRH380AL__eetc-vao_LC_VI_tcVI")
    variables = extract_variables_from_solution(solution)
    plot_tc_results(variables, ds=100, bottom_left_loc=40000)
    pass


if __name__ == '__main__':
    main()
