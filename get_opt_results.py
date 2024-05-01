import os
import json
from typing import Any
import re
import pandas as pd


def parse_log_file(file_path) -> dict[str, float]:
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
        with open(file_path, 'r') as file:
            for line in file:
                for key, pattern in patterns.items():
                    match = re.search(pattern, line)
                    if match:
                        results[key] = float(match.group(1))
    except FileNotFoundError:
        print(f"Error: The file at {file_path} does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return results


def get_tc_results_from_folder(folder_path: str):
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


def get_eetcvao_results_from_folder(folder_path: str):
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


def get_solution_info_from_folder(folder_path: str):
    root = fr"E:\OneDrive\Documents\00-MyResearch\20230524_VAO-EETC\EETC-VAO_202404\Cases\{folder_path}"
    model_folders = os.listdir(root)
    result_dic: dict[tuple[str, str], dict[str, float]] = {}
    for model_folder in model_folders:
        case, train, model_name = model_folder.split("__")
        print(case, train, model_name)
        json_file: str = os.path.join(root, model_folder, f"{case}__{train}__{model_name}.json")
        json_data: dict[str, Any] = json.load(open(json_file, 'r', encoding="utf-8"))

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


def main():
    # get_tc_results_from_folder(folder_path="eetc_tcVI")
    # get_solution_info_from_folder(folder_path="eetc_tcVI")
    get_eetcvao_results_from_folder("eetc-vao_VI")
    pass


if __name__ == '__main__':
    main()
