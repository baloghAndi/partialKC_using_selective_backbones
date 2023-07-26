import csv
import math
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as pl
import time
import CNFmodel
import matplotlib.colors as mcolors
class Logger:

    def __init__(self, filename, column_names, expr_data):
        print(os.getcwd())
        self.f = open(filename,"a+")
        self.writer = csv.writer(self.f, delimiter=',')
        self.column_names = column_names
        self.expr_data = expr_data

    def log_expr(self, expr_name):
        self.writer.writerow([expr_name])
        self.writer.writerow(self.column_names)
        if len(self.expr_data.data) > 0:
            self.expr_data.all_expr_data[expr_name] = self.expr_data.data.copy()
        self.expr_data.exprs.append(expr_name)
        self.expr_data.data = []
        self.f.flush()
    def log(self, row):
        self.writer.writerow(row)
        self.expr_data.data.append(row)
        self.f.flush()
    def close(self):
        if len(self.expr_data.data) > 0:
            self.expr_data.all_expr_data[self.expr_data.exprs[-1]] = self.expr_data.data.copy()
        self.f.flush()
        self.f.close()

    def set_start_time(self, start):
        self.start_time = start

    def get_time_elapsed(self):
        return time.perf_counter()-self.start_time



class ExprData:

    def __init__(self, column_names):
        self.data = []
        self.all_expr_data = {}
        self.exprs = []
        self.column_names = column_names
        self.init_compilation_time = {}
        self.finish_time = {}

    def read_stats_file(self, filename):
        # print(filename)
        self.data = []
        self.all_expr_data = {}
        self.exprs = []
        self.filename = filename
        with open(filename) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            prev_line = []
            line_index = 0
            for line in reader:
                if len(line) == 1 or ".cnf" in line[0]: #if first line or start of new expr
                    line_index = 0
                    # print("expr:",line)
                    if len(self.data) > 0:
                        self.all_expr_data[self.exprs[-1]] = self.data.copy()
                        self.finish_time[self.exprs[-1]] = float(prev_line[-1]) - self.init_compilation_time[self.exprs[-1]]
                    if len(self.data) == 0 and len(self.exprs) > 0:
                        self.exprs.pop()
                    self.exprs.append(line[0]) #add expr name - should only add if it has data
                    self.data = []
                elif self.column_names == line:
                    continue
                else:
                    # print(line)
                    typed_line = []
                    # [int(x) if i != 1 else x for i,x in enumerate(line[:-1]) ]
                    for i,x in enumerate(line[:-1]):
                        if i!=1:
                            if "[" in x:
                                typed_line.append(x)
                            elif i ==3:
                                typed_line.append(float(x)) #read mc as float
                            else:
                                # print(type(x), float(x))
                                typed_line.append(int(x))
                                # typed_line.append(float(x))
                        else:
                            typed_line.append(x)
                    typed_line.append(float(line[-1]))
                    self.data.append(typed_line)
                    prev_line = line
                if line_index == 1:
                    self.init_compilation_time[self.exprs[-1]] = float(line[-1])
                    print("init compilation ", self.exprs[-1])
                line_index += 1
            if len(self.data) > 0:
                self.all_expr_data[self.exprs[-1]] = self.data.copy()
            if len(self.data) == 0 and len(self.exprs) > 0:
                self.exprs.pop()
        # print(self.all_expr_data)
        # for k in self.all_expr_data:
        #     print(k, self.all_expr_data[k])


    def plot_all_efficiencies(self, column_name, name_extension):
        print("CALCULATE")
        for expr in self.exprs:
            title = expr.split("/")[-1]
            out_file = expr.replace(".cnf", name_extension+".png")
            self.plot_efficiency(self.all_expr_data[expr], title, out_file, column_name)
            # self.plot_efficiency_MC_BDD(self.all_expr_data[expr], title, out_file)

    def plot_all_efficiencies_percentage(self):
        print("CALCULATE")
        for expr in self.exprs:
            title = expr.split("/")[-1]
            out_file = expr.replace(".cnf", "_percentage2.png")
            self.plot_efficiency_percentage(self.all_expr_data[expr], title, out_file)

    def plot_efficiency(self, data, title, file, column_name):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        print(data)
        x = [i for i in range(1, len(data))]
        column_index = self.column_names.index(column_name)
        #use this y value to compare incrementally
        y = []
        for i in range(len(data) - 1):
            if data[i][column_index] == 0:
                if data[i + 1][column_index] == 0:
                    y.append(0)
                else:
                    y.append(100)
            else:
                y.append(100 * (data[i][column_index] - data[i + 1][column_index]) / data[i][column_index])
        # y = [100 * (data[i][column_index] - data[i + 1][column_index]) / data[i][column_index] if  data[i][column_index]!=0 else 100 for i in range(len(data) - 1)]
        file = file.replace(".png", "_" + column_name + ".png")

        #use the below y to compate to the original problem - initcompare
        # init_value = data[0][column_index]
        # y = [100 * (init_value - data[i][column_index]) / init_value for i in range(1,len(data))]
        # file = file.replace(".png", "_"+column_name+"_initcompare.png")

        print(x)
        print(y)
        ax1.scatter(x, y, c="green", label=column_name+" ratio")
        ax1.plot(x, y, c="green")
        plt.xticks(x)
        title = title.replace(".cnf", "")
        plt.xlabel("Size of selective backbone")
        plt.ylabel("Percentage of "+column_name+" reduction")
        plt.title(title)
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels)
        fig.tight_layout()
        plt.grid()


        print(file)
        plt.savefig(file)

    def plot_efficiency_MC_BDD(self, data, title, file):
        """
        This plots the percentage reduction
        :param data:
        :param title:
        :param file:
        :return:
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        print(data)
        # x = [i for i in range(1, len(data))]
        # print([data[i][3] for i in range(len(data))])
        # print([data[i][3] - data[i + 1][3] for i in range(len(data) - 1)])
        column_index_MC= self.column_names.index("MC")
        column_index_BDD= self.column_names.index("dag_size")

        #incremental
        # x = [100 * (data[i][column_index_MC] - data[i + 1][column_index_MC]) / data[i][column_index_MC] for i in
        #      range(len(data) - 1)]
        # y = [100 * (data[i][column_index_BDD] - data[i + 1][column_index_BDD]) / data[i][column_index_BDD] for i in
        #      range(len(data) - 1)]
        # file = file.replace(".png", "_Mc_BDD_incremental" + ".png")

        #use the below x and y to compare to initial problem
        x_init = data[0][column_index_MC]
        x = [100 * (x_init - data[i][column_index_MC]) / x_init for i in range(len(data) - 1)]
        y_init = data[0][column_index_BDD]
        y = [100 * (y_init - data[i][column_index_BDD]) / y_init for i in range(len(data) - 1)]
        file = file.replace(".png", "_Mc_BDD_initcompare" + ".png")

        # y = [ data[i][3]-data[i+1][3] for i in range(len(data)-1) ]
        print(x)
        print(y)
        ax1.scatter(x, y, c="green", label= "")
        ax1.plot(x, y, c="green")
        # plt.xticks(x)
        # plt.xlim(0, 100)
        # plt.ylim(0, 100)
        ax1.axline([0, 0], [100, 100], color="grey")
        title = title.replace(".cnf", "")
        plt.xlabel("BDD node count reduction percentage")
        plt.ylabel("Model count reduction percentage")
        plt.title(title)
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels)
        fig.tight_layout()
        plt.grid()

        print(file)
        plt.savefig(file)



    def plot_efficiency_percentage(self, data, title, file):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        print(data)
        n = len(data)
        x = [i for i in range(1, len(data))]
        # x = [ 100*(i/n) for i in range(1,len(data)) ]
        max_mc = data[0][3]
        y = [100 * (max_mc - data[i][3]) / max_mc for i in range(1, len(data))]
        # y = [ data[i][3]-data[i+1][3] for i in range(len(data)-1) ]
        print(x)
        print(y)
        ax1.scatter(x, y, c="green", label="model count ratio")
        ax1.plot(x, y, c="green")
        plt.xticks(x)
        # plt.yticks(y)
        title = title.replace(".cnf", "")
        plt.xlabel("Percentage of selective backbone")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45, ha='right')
        plt.ylabel("Percentage of model count reduction from initial")
        plt.title(title)
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels)
        fig.tight_layout()
        plt.grid()
        # file = file.replace(".png", "_MC.png")
        print(file)
        plt.savefig(file)

    def plot_all_exprs(self, file, column_name):
        all_expr_data = []
        title = "All experiments"
        all_y = []
        all_x = []
        column_index = self.column_names.index(column_name)
        for expr in self.exprs:
            data=  self.all_expr_data[expr]
            y = [100 * (data[i][column_index] - data[i + 1][column_index]) / data[i][column_index] for i in
                 range(len(data) - 1)]
            all_y.append(y.copy())

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        x = [i for i in range(1, len(data))]
        # use this y value to compare incrementally


        file = file.replace(".png", "_" + column_name + "_incremental.png")

        n_lines = 10
        colors = pl.cm.jet(np.linspace(0, 1, n_lines))
        for i,y in  enumerate(all_y):
            ax1.plot(x, y,color="green")
        print(x)
        print(y)
        # ax1.scatter(x, y, c="green", label=column_name + " ratio")
        # ax1.plot(x, y, c="green")
        # plt.xticks(x)
        # title = title.replace(".cnf", "")
        plt.xlabel("Size of selective backbone")
        plt.ylabel("Percentage of " + column_name + " reduction")
        plt.title(title)
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels)
        fig.tight_layout()
        plt.grid()

        print(file)
        plt.savefig(file)

    def best_ratio_table_per_alg(self):
        """
        create  table with header(expr name, N(nb variables), Best p ( nb vars where best ratio was achieved), best ratio MC/BDD size, Initial BDD size, Initial MC)
        actually in here we only have access to one alg type
        so we need to iterate through all exprs and return best for all exprs
        :return:
        """
        #TODO: should we look at best ratio per alg or overall?
        result = {e:{} for e in self.exprs}
        ratios =[]
        mc_index = self.column_names.index("MC")
        bdd_index = self.column_names.index("dag_size")
        N_index = self.column_names.index("n_vars")
        for expr in self.exprs:
            init_MC = self.all_expr_data[expr][0][mc_index]
            init_BDD = self.all_expr_data[expr][0][bdd_index]
            best_index = 0
            best_mc = 0
            best_bdd = 0
            best_ratio = 0.0
            for i,data in enumerate(self.all_expr_data[expr]):
                r = (data[mc_index]/init_MC) / (data[bdd_index]/init_BDD)
                if r > best_ratio :
                    best_ratio = r
                    best_index = i
                    best_mc = data[mc_index]
                    best_bdd = data[bdd_index]
            result[expr] = {"ratio":best_ratio, "mc":best_mc, "bdd":best_bdd, "index": best_index , "N": data[N_index],
                            "init_bdd":self.all_expr_data[expr][0][bdd_index], "init_MC": self.all_expr_data[expr][0][mc_index]}
            ratios.append(best_ratio)
        # TODO: check this code
        return result, ratios

    def count_proper_backbones(self):
        """
        Can only count at init stat file
        :return:
        """
        backbones = []
        if "init" not in self.filename:
            print("can't count backbones")
            return
        mc_index = self.column_names.index("MC")
        for expr in self.exprs:
            nb_backbone = 0
            for data in self.all_expr_data[expr]:
                if data[mc_index] == 0:
                    nb_backbone += 1
            if nb_backbone > 0:
                n_index = self.column_names.index("n_vars")
                N = self.all_expr_data[expr][0][n_index]
                cnf, mc = self.reload_expr(expr, N)
                print(expr, nb_backbone, N)
            # print(expr, ",", nb_backbone)
            backbones.append(nb_backbone)
        return backbones


    def reload_expr(self, expr_name,N):
        cnf = CNFmodel.CNF()
        literals = ["x" + str(i) for i in range(1, N+1)]
        cnf.bdd.declare(*literals)
        bdd_file = expr_name.replace(".cnf", ".dddmp")
        loaded_bdd = cnf.bdd.load(bdd_file)
        root = loaded_bdd[0]
        mc = root.count(len(literals))
        cnf.root_node = root
        return cnf, mc

    def get_metric_wrt_initial_per_expr(self, metric):
        """
        Calculate ratio of each p with respect to the initial MC/BDD ratio
        :return:
        """
        # result = {e: [] for e in self.exprs}
        result = {}
        mc_index = self.column_names.index("MC")
        bdd_index = self.column_names.index("dag_size")
        smallest_n = 600
        for expr in self.exprs:
            curren_n = self.all_expr_data[expr][0][self.column_names.index("n_vars")]
            # if curren_n >= 100:
            #     print("----------------------------skipped:", expr)
            #     continue
            if curren_n < smallest_n:
                smallest_n = curren_n
            ratios = []
            if metric == "ratio":
                init_ratio = self.all_expr_data[expr][0][mc_index] / self.all_expr_data[expr][0][bdd_index]
            elif metric == "MC":
                init_ratio = self.all_expr_data[expr][0][mc_index]
            elif metric == "BDD":
                init_ratio = self.all_expr_data[expr][0][bdd_index]
            ratios.append(1.0)
            for i in range(1, len(self.all_expr_data[expr])):
                if metric == "ratio":
                    r = self.all_expr_data[expr][i][mc_index] / self.all_expr_data[expr][i][bdd_index]
                elif metric == "MC":
                    r = self.all_expr_data[expr][i][mc_index]
                elif metric == "BDD":
                    r = self.all_expr_data[expr][i][bdd_index]
                ratios.append(r / init_ratio)
            result[expr] = ratios.copy()
        return result, smallest_n

def get_metric_per_alg(folder, labels, metric):
    columns = ["p", "var", "value", "MC", "BDD len", 'n_vars', 'n_nodes', 'n_reorderings', 'dag_size', 'time']
    results = []
    for type in labels:
        stats_file = folder + "dataset_stats_" + type + "_reorder.csv"
        expr_data = ExprData(columns)
        expr_data.read_stats_file(stats_file)
        percentage_results, smallest_n = expr_data.get_metric_wrt_initial_per_expr(metric)
        results.append(percentage_results)
    return results, smallest_n

def all_best_ratios_grouped(expr_data_per_alg, algs, table_header):
    """
    return data such that it contains min and max for the set of experiments
    :param expr_data_per_alg:
    :param algs:
    :param table_header:
    :return:
    """
    expr_names = expr_data_per_alg[0].exprs
    table_content = []
    all_ratios = {}
    best_ratio_table = {i:{} for i in expr_names}
    name = expr_names[0].replace(".cnf", "").replace("./", "")
    folder = "/".join(name.split("/")[:-1])
    if "Planning" in folder:
        folder = folder.replace("pddlXpSym/", "")
    stats_file = "./paper_data/" + folder + "/dataset_stats_init_reorder.csv"
    print(type, stats_file)
    columns = ["p", "var", "value", "MC", "BDD len", 'n_vars', 'n_nodes', 'n_reorderings', 'dag_size', 'time']
    init_expr = ExprData(columns)
    init_expr.read_stats_file(stats_file)
    proper_backbone = init_expr.count_proper_backbones()
    min_table_content = {k:None for k in table_header}
    max_table_content = {k:None for k in table_header}

    for exp_data, alg in zip(expr_data_per_alg, algs):
        result, ratios = exp_data.best_ratio_table_per_alg()
        all_ratios[alg] = [result, ratios]
    for index in range(len(expr_names)):
        best_ratio = 0
        best_result = []
        best_alg = ""
        for alg in algs:
            ratio = all_ratios[alg][1][index]
            if ratio >= best_ratio:
                best_ratio = ratio
                best_result = all_ratios[alg][0][expr_names[index]]
                best_alg = alg
        best_ratio_table[expr_names[index]] = {"best_ratio": best_ratio, "best_alg":best_alg, "details": best_result}
        name = expr_names[index].replace(".cnf","").replace("./","")
        name = name.split("/")[-1]
        row = [name, round(best_result["index"]/best_result["N"], 3), round(proper_backbone[index]/best_result["N"], 3) ,round(best_ratio,3), best_alg, best_result["init_bdd"], best_result["init_MC"],  best_result["index"], best_result["N"]]
        table_content.append(row)
    for e in best_ratio_table.keys():
        print(e, best_ratio_table[e])
    print(table_header)
    for l in table_content:
        print(l)
    return table_content

def all_best_ratios(expr_data_per_alg, algs, table_header):
    expr_names = expr_data_per_alg[0].exprs
    table_content = []
    all_ratios = {}
    best_ratio_table = {i:{} for i in expr_names}
    name = expr_names[0].replace(".cnf", "").replace("./", "")
    folder = "/".join(name.split("/")[:-1])
    if "Planning" in folder:
        folder = folder.replace("pddlXpSym/", "")
    stats_file = "./paper_data/" + folder + "/dataset_stats_init_reorder.csv"
    # print(type, stats_file)
    columns = ["p", "var", "value", "MC", "BDD len", 'n_vars', 'n_nodes', 'n_reorderings', 'dag_size', 'time']
    init_expr = ExprData(columns)
    init_expr.read_stats_file(stats_file)
    proper_backbone = init_expr.count_proper_backbones()
    nb_backbone_count = sum(i > 0 for i in proper_backbone)
    nb_clauses = [get_nb_clauses(e) for e in expr_names ]

    for exp_data, alg in zip(expr_data_per_alg, algs):
        result, ratios = exp_data.best_ratio_table_per_alg()
        all_ratios[alg] = [result, ratios]
    for index in range(len(expr_names)):
        best_ratio = 0
        best_result = []
        best_alg = ""
        for alg in algs:
            ratio = all_ratios[alg][1][index]
            if ratio >= best_ratio:
                best_ratio = ratio
                best_result = all_ratios[alg][0][expr_names[index]]
                best_alg = alg
        best_ratio_table[expr_names[index]] = {"best_ratio": best_ratio, "best_alg":best_alg, "details": best_result}
        name = expr_names[index].replace(".cnf","").replace("./","")
        name = name.split("/")[-1]

        # table_header = ["Expr", "P/N", "nb backbone/N", "Best adjusted ratio","Best alg", "Initial BDD size",
        # "Initial MC", "P", "N", "M", "mc","bdd",
        # "m/n", "mc/2^n","instance count", "nb inst with B"]


        row = [name, round(best_result["index"]/best_result["N"], 3), round(proper_backbone[index]/best_result["N"], 3) ,round(best_ratio,3), best_alg,
               best_result["init_bdd"], best_result["init_MC"],  best_result["index"], best_result["N"] , nb_clauses[index],  best_result["mc"], best_result["bdd"],
               round(nb_clauses[index]/ best_result["N"], 3), round(best_result["mc"]/ math.pow(2, best_result["N"]), 3),  len(expr_names), nb_backbone_count ]
        table_content.append(row)
    # for e in best_ratio_table.keys():
    #     print(e, best_ratio_table[e])
    # print(table_header)
    # for l in table_content:
    #     print(l)
    return table_content

def get_nb_clauses(filename):
    with open(filename, "r") as f:
        content = f.readline()
        nb_clauses = int(content.strip().split(" ")[3])
    return nb_clauses
def get_best_ratio_data(folder, labels, table_header, aggregate):
    columns = ["p", "var", "value", "MC", "BDD len", 'n_vars', 'n_nodes', 'n_reorderings', 'dag_size', 'time']
    expr_datas = []
    for type in labels:
        stats_file = folder + "dataset_stats_" + type + "_reorder.csv"
        print(type, stats_file)
        expr_data = ExprData(columns)
        expr_data.read_stats_file(stats_file)
        expr_datas.append(expr_data)
    table_content = all_best_ratios( expr_datas, labels, table_header)
    # table_header = ["Expr", "P/N", "nb backbone/N", "Best adjusted ratio","Best alg", "Initial BDD size", "Initial MC", "P", "N", "M", "mc","bdd", "m/n", "mc/2^n","instance count", "nb inst with B"]

    if aggregate: #get min and max per experiments in a folder per column
        aggregated_table_content = [[], {}]
        aggregated_table_content[0] = [old_value for old_value in table_content[0]]
        # aggregated_table_content[1] = [old_value for old_value in table_content[-1]]
        aggregated_table_content[1] = [ 0, 0 ]
        print("===================aggregate")
        for i in [1,2,3,5,6,7,8,9 ]: #column indexes to aggregate
            temp = [t[i] for t in table_content]
            aggregated_table_content[0][i].extend([ min(temp), max(temp)])
            # aggregated_table_content[1][i] = max(temp)
            if i == 3:
                aggregated_table_content[1][0] = np.average(temp)
                aggregated_table_content[1][1] = np.median(temp)
            print(folder,temp, table_header[i])
        return aggregated_table_content
    return table_content

def create_average_ratio_plot(folders, outputfile, title, labels, min_n):
    ratios_to_average = [[] for i in labels]
    smallest_n = 600
    for folder in folders:
        print(folder)
        folder_ratio_percentages, folder_smallest_n = get_metric_per_alg(folder, labels, "ratio")
        if folder_smallest_n < smallest_n:
            smallest_n = folder_smallest_n
        for i, data in enumerate(folder_ratio_percentages):
            for d in data:
                print(d, data[d])
                ratios_to_average[i].append(data[d])
                # print(d, data[d])
            # print("avg", ratios_to_average[i])
    #print per label all folders - plot should be here - need to first normalize to same length
    if smallest_n < min_n:
        smallest_n = min_n
    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(111)
    x = [(100*i) /smallest_n for i in range(smallest_n+1)]
    # print(x)
    # exit(99)
    # x = [i for i in range(smallest_n+1)]

    colors = ["blue", "cyan", mcolors.CSS4_COLORS["steelblue"], "orange", 'red', "green", "olive"]
    marks = ["s", "o", "p", "*", "x", "v", "^"]
    labels = [s.replace("_1234", "") for s in labels]
    labels[2] = "random_selection_ratio"
    plt.xlabel("Percentage of selective backbone size")
    plt.ylabel("Average of MC/BDDsize ratio percentage wrt initial ratio")
    for i, l in enumerate(labels):
        # print(l)
        label_average = [0 for i in range(smallest_n+1)]
        exprs_to_avg = len(ratios_to_average[i])
        for d in ratios_to_average[i]:
            # print(d)
            sampled_data = sample_data(d,smallest_n+1)
            # print(len(sampled_data))
            label_average = [label_average[i]+sampled_data[i] for i in range(smallest_n+1)]
            # print(label_average)
        label_average = [label_average[i]/exprs_to_avg for i in range(smallest_n+1)]
        # print(len(x), len(label_average), smallest_n)
        ax1.scatter(x, label_average, c=colors[i], label=l, marker=marks[i])
        ax1.plot(x, label_average, c=colors[i])

        # print(label_average)
    # print(smallest_n)
    # plt.xticks(x)
    # title = folders
    # plt.xticks(x)
    plt.ylim(top=5.5)
    # plt.title(title)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    fig.tight_layout()
    plt.grid()

    # plt.show()
    # file = "./paper_data/all_avg_ratios_above100.png"
    # file = "./paper_data/all_avg_ratios_below100.png"
    # file = "./paper_data/all_avg_ratios.png"
    # expr_name = folders[0].split("/")[-2].replace(".cnf", "")
    # if len(folders) == 1:
    #     file = "./paper_data/" + expr_name+ "_avg_ratios.png"
    print(outputfile)
    plt.savefig(outputfile)

def create_average_efficiency_plot(folders,outputfile, title,  labels, min_n):
    MC_to_average = [[] for i in labels]
    BDD_to_average = [[] for i in labels]
    smallest_n = 600
    for folder in folders:
        print(folder)
        folder_MC_percentages, folder_smallest_n = get_metric_per_alg(folder, labels, "MC")
        folder_BDD_percentages, folder_smallest_n = get_metric_per_alg(folder, labels, "BDD")
        if folder_smallest_n < smallest_n:
            smallest_n = folder_smallest_n
        for i, data in enumerate(folder_MC_percentages):
            for d in data:
                MC_to_average[i].append(data[d])
                # print(d, data[d])
            # print("avg", MC_to_average[i])
        for i, data in enumerate(folder_BDD_percentages):
            for d in data:
                BDD_to_average[i].append(data[d])
    #print per label all folders - plot should be here - need to first normalize to same length
    if min_n > smallest_n:
        smallest_n = min_n
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111)
    colors = ["blue", "cyan", mcolors.CSS4_COLORS["steelblue"],"orange", "red", "green", "olive"]
    marks = ["s", "o", "p", "*", "x", "v", "^"]
    plt.xlabel("Average BDD size percentage")
    plt.ylabel("Average Model count percentage")
    labels = [s.replace("_1234", "") for s in labels]
    labels[2] = "random_selection_ratio"
    for i, l in enumerate(labels):
        # print(l)
        mc_average = [0 for i in range(smallest_n+1)]
        bdd_average = [0 for i in range(smallest_n+1)]
        exprs_to_avg = len(MC_to_average[i])
        for mc_d, bdd_d in zip(MC_to_average[i], BDD_to_average[i]):
            # print(d)
            sampled_mc_data = sample_data(mc_d,smallest_n+1)
            mc_average = [mc_average[i]+sampled_mc_data[i] for i in range(smallest_n+1)]

            sampled_bdd_data = sample_data(bdd_d,smallest_n+1)
            bdd_average = [bdd_average[i]+sampled_bdd_data[i] for i in range(smallest_n+1)]
            # print(label_average)
        mc_average = [mc_average[i]/exprs_to_avg for i in range(smallest_n+1)]
        bdd_average = [bdd_average[i]/exprs_to_avg for i in range(smallest_n+1)]
        # print(mc_average)
        # print(bdd_average)
        ax1.scatter(bdd_average, mc_average, c=colors[i], label=l, marker=marks[i])
        ax1.plot(bdd_average, mc_average, c=colors[i],  alpha=0.7, linewidth=1)


    plt.ylim(0, 1)
    plt.xlim(1, 0)

    ax1.axline([1, 1], [0, 0], color="grey")
    # print(smallest_n)
    # plt.xticks(x)
    # title = folders
    # plt.title(title)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    fig.tight_layout()
    plt.grid()
    # plt.show()
    # file = "./paper_data/all_avg_efficiency_AB.png"
    # file = "./paper_data/all_avg_efficiency_exceptAB.png"
    # file = "./paper_data/all_avg_efficiency_below100.png"
    # file = "./paper_data/all_avg_efficiency_above100.png"
    # file = "./paper_data/all_avg_efficiency.png"
    # expr_name = folders[0].split("/")[-2].replace(".cnf", "")
    # print(expr_name)
    # if len(folders) == 1:
    #     file = "./paper_data/" + expr_name+"_avg_efficiency.png"
    print(outputfile)
    plt.savefig(outputfile)
def sample_data(data, smallest_n):
    n = len(data)
    # print(len(data),smallest_n)
    return [data[int((i*n)/smallest_n)] for i in range(smallest_n)]
def create_best_ratio_table(out_folder, folders, labels,aggregate ):
    f = open(out_folder, "a+")
    writer = csv.writer(f, delimiter=',')
    table_header = ["Expr", "P/N", "nb backbone/N", "Best adjusted ratio","Best alg", "Initial BDD size", "Initial MC", "P", "N", "M", "mc","bdd", "m/n", "mc/2^n","instance count", "nb inst with B"]
    # row = [name, best_result["index"] / best_result["N"], proper_backbone / best_result["N"], round(best_ratio, 3),  best_alg, best_result["init_bdd"], best_result["init_MC"], best_result["index"], best_result["N"]]
    writer.writerow(table_header)
    for folder in folders:
        print("analyze ", folder)
        table_content = get_best_ratio_data(folder, labels, table_header, aggregate)
        for line in table_content:
            writer.writerow(line)

def read_ratio_table(filename, metric):
    # Permanently changes the pandas settings
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)

    df = pd.read_csv(filename)
    print(df)
    metric = "Best adjusted ratio"
    df = df.sort_values(metric)
    print(df)
    # print(df.to_string())
    print("no improvement for : ", len(df[df['P/N']==0.00]))

    min_df = df.groupby(pd.cut(df["Best adjusted ratio"], [0, 1, 2, 3, 4, 5, 9, 27])).min()
    print(min_df)
    # min_df.to_csv('./paper_data/temp_min.csv')
    max_df = df.groupby(pd.cut(df["Best adjusted ratio"], [0, 1, 2, 3, 4, 5, 9, 27])).max()
    print(max_df)
    # max_df.to_csv('./paper_data/temp_max.csv')


    count_df = df.groupby(pd.cut(df["Best adjusted ratio"], [0, 1, 2, 3, 4, 5, 9, 27])).count()
    print("count")
    print(count_df)

    m = "Best adjusted ratio"
    m = "mc/2^n"
    temp = df.groupby(pd.qcut(df[m], 10)).count()
    print(temp)
    temp = df.groupby(pd.qcut(df[m], 10)).min()
    print(temp)

    temp = df.groupby(pd.qcut(df[m], 10)).max()
    print(temp)

    temp = df.groupby(pd.qcut(df[m], 10)).mean()
    print(temp)

    temp = df.groupby(pd.qcut(df[m], 10)).median()
    print(temp)

#section to plot multiple experiments in one - same expr with a different selection criteria
def plot_multiple(folder, expr_data_list, column_name, labels, plot_type):
    expr_data1 = expr_data_list[0]
    for expr in expr_data1.exprs:
        print("=======================================",expr)
        title = expr.split("/")[-1]
        out_file = folder+title.replace(".cnf", "_multiple.png")
        print([item.all_expr_data.keys() for item in expr_data_list])
        expr_data_list_values = [item.all_expr_data[expr] for item in expr_data_list]
        check_lens = [len(expr_d) for expr_d in expr_data_list_values[1:]]
        if not check_lens.count(check_lens[0]) == len(check_lens):
            print("EXP DID NOT FINISH FOR ALL ALGS: ", expr, check_lens)
            continue
        column_index_MC = expr_data1.column_names.index("MC")
        column_index_BDD = expr_data1.column_names.index("dag_size")
        if column_name == "efficiency":
            plot_efficiency_MC_BDD_ratio(expr_data_list_values, title, out_file, column_index_MC, column_index_BDD, labels,plot_type)
        elif column_name == "ratio":
            plot_mc_bdd_ratio(expr_data_list_values, title, out_file, column_index_MC, column_index_BDD, labels,plot_type)
        else:
            print("----------------------------------------------",out_file)
            column_index = expr_data1.column_names.index(column_name)
            plot_multiple_efficiency(expr_data_list_values, title, out_file, column_name, column_index, labels, plot_type)
        # break


def plot_multiple_efficiency(expr_data_list_values, title, out_file, column_name, column_index, labels, plot_type):
    fig = plt.figure(figsize=(10,7))
    ax1 = fig.add_subplot(111)
    x = [i for i in range(1, len(expr_data_list_values[0]))]
    # colors = ["blue", "orange", "red", "green", "olive", "cyan", 'red']
    colors = ["blue", "cyan", mcolors.CSS4_COLORS["steelblue"],"orange", "red", "green", "olive", 'red']
    marks = ["s", "o", "p", "*", "x", "v", "^"]
    plt.xlabel("Size of selective backbone")
    plt.ylabel("Percentage of " + column_name + " reduction")
    if column_name == "dag_size":
        plt.ylabel("Percentage of BDD size reduction")
    else:
        plt.ylabel("Percentage of Model count reduction")


    for index, label in enumerate(labels):
        expr_data = expr_data_list_values[index]

        if plot_type == "inc":
            # use this y value to compare incrementally
            y_data = []
            for i in range(len(expr_data) - 1):
                if expr_data[i][column_index] == 0:
                    if expr_data[i + 1][column_index] == 0:
                        y_data.append(0)
                    else:
                        y_data.append(100)
                else:
                    y_data.append(100 * (expr_data[i][column_index] - expr_data[i + 1][column_index]) / expr_data[i][column_index])
            # y = [100 * (expr_data[i][column_index] - expr_data[i + 1][column_index]) / expr_data[i][column_index] for i in range(len(expr_data) - 1)]
            file = out_file.replace(".png", "_" + column_name + "_incremental.png")
            print(len(x),len(y_data))
            ax1.scatter(x, y_data, c=colors[index], label=column_name + " " + label + " ratio")
            ax1.plot(x, y_data, c=colors[index])

        elif plot_type == "init":
            # use the below y to compate to the original problem - initcompare
            file = out_file.replace(".png", "_" + column_name + "_initcompare.png")
            init_value = expr_data[0][column_index]
            y_data = [100 * (init_value - expr_data[i][column_index]) / init_value for i in range(1, len(expr_data))]
            # todo : expr_data2[i][column_index]) / init_value2 calculate this
            ax1.scatter(x, y_data, c=colors[index], label=column_name + " " + label + " ratio")
            ax1.plot(x, y_data, c=colors[index])

        else: #raw

            file = out_file.replace(".png", "_" + column_name + "_raw.png")
            x = [i for i in range(0, len(expr_data_list_values[0]))]
            y_data = [ d[column_index] for d in expr_data ]
            if column_name == "dag_size":
                plt.ylabel("BDD size")
            else:
                plt.ylabel("Model count")
            ax1.scatter(x, y_data, c=colors[index], label=column_name + " " + label )
            ax1.plot(x, y_data, c=colors[index], marker=marks[index])
            ax1.set_yscale('symlog')
            # plt.xticks(rotation=30, ha='left')
            # plt.setp(ax1.xaxis.get_majorticklabels(), rotation=-30, ha="left", rotation_mode="anchor")
            # plt.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')





    plt.xticks(x)
    title = title.replace(".cnf", "")

    plt.title(title)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    fig.tight_layout()
    # plt.grid()

    print(file)
    plt.savefig(file)
def plot_efficiency_MC_BDD( expr_data_list_values, title, file, column_index_MC, column_index_BDD, labels, plot_type):

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    colors = ["blue", "orange", "green"]
    plt.xlabel("BDD node count reduction percentage")
    plt.ylabel("Model count reduction percentage")
    for index, label in enumerate(labels):
        expr_data = expr_data_list_values[index]
        if plot_type == "inc":
        #incremental
            y = [100 * (expr_data[i][column_index_MC] - expr_data[i + 1][column_index_MC]) / expr_data[i][column_index_MC] for i in
                 range(len(expr_data) - 1)]
            x = [100 * (expr_data[i][column_index_BDD] - expr_data[i + 1][column_index_BDD]) / expr_data[i][column_index_BDD] for i in
                 range(len(expr_data) - 1)]
            ax1.scatter(x, y, c=colors[index], label=label)
            ax1.plot(x, y, c=colors[index])
            file = file.replace(".png", "_Mc_BDD_incremental" + ".png")
        elif plot_type == "init":
            #use the below x and y to compare to initial problem
            y_init = expr_data[0][column_index_MC]
            y = [100 * (y_init - expr_data[i][column_index_MC]) / y_init for i in range(len(expr_data) - 1)]
            x_init = expr_data[0][column_index_BDD]
            x = [100 * (x_init - expr_data[i][column_index_BDD]) / x_init for i in range(len(expr_data) - 1)]
            file = file.replace(".png", "_Mc_BDD_initcompare" + ".png")
            ax1.scatter(x, y, c=colors[index], label=label)
            # ax1.plot(x, y, c=colors[index])
        else:
            y = [ d[column_index_MC] for d in expr_data[1:] ]
            x = [ d[column_index_BDD] for d in expr_data[1:] ]

            ax1.scatter(x, y, c=colors[index], label=label)
            # ax1.plot(x, y, c=colors[index])
            file = file.replace(".png", "_Mc_BDD_raw" + ".png")
            plt.xlabel("BDD size ")
            plt.ylabel("Model count ")




    # plt.xticks(x)
    # plt.xlim(0, 100)
    # plt.ylim(0, 100)
    ax1.axline([0, 0], [100, 100], color="grey")
    title = title.replace(".cnf", "")

    plt.title(title)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    fig.tight_layout()
    plt.grid()

    print(file)
    plt.savefig(file)
def plot_mc_bdd_ratio(expr_data_list_values, title, file, column_index_MC, column_index_BDD, labels, plot_type):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # colors = ["blue", "orange",'red', "green","olive", "cyan" ]
    colors = ["blue", "cyan", mcolors.CSS4_COLORS["steelblue"],"orange", "red", "green", "olive", 'red']
    plt.ylabel("Model count/ BDD node count ")
    plt.xlabel("selective backbone")
    file = file.replace(".png", "_Mc_BDD_mc_bdd_ratio" + ".png")
    x = [i for i in range(0, len(expr_data_list_values[0]))]
    for index, label in enumerate(labels):
        expr_data = expr_data_list_values[index]

        #use the below x and y to compare to initial problem
        y = [expr_data[i][column_index_MC] / expr_data[i][column_index_BDD] for i in range(0,len(expr_data) )]

        print(label)
        ax1.scatter(x, y, c=colors[index], label=label)
        ax1.plot(x, y, c=colors[index])



    # plt.xticks(x)
    # plt.xlim(0,1)
    # plt.ylim(0, 100)
    # ax1.axline([0, 0], [1, 1], color="grey")

    plt.title(title)
    title = title.replace(".cnf", "")
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    fig.tight_layout()
    plt.grid()

    print(file)
    plt.savefig(file)
def plot_efficiency_MC_BDD_ratio( expr_data_list_values, title, file, column_index_MC, column_index_BDD, labels, plot_type):
    #Plot for efficiency
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111)
    colors = ["blue", "cyan", mcolors.CSS4_COLORS["steelblue"],"orange", "red", "green", "olive", 'red']
    marks = ["s", "o","p", "*", "x", "v", "^"]
    plt.xlabel("BDD node count percentage")
    plt.ylabel("Model count percentage")
    # file = file.replace(".png", "_Mc_BDD_efficiency" + ".png")
    file = file.replace(".png", "_Mc_BDD_efficiency_reorder" + ".png")
    for index, label in enumerate(labels):
        expr_data = expr_data_list_values[index]

            #use the below x and y to compare to initial problem
        y_init = expr_data[0][column_index_MC]
        y = [ expr_data[i][column_index_MC] / y_init for i in range(1,len(expr_data))]
        x_init = expr_data[0][column_index_BDD]
        x = [ expr_data[i][column_index_BDD] / x_init for i in range(1, len(expr_data))]
        ax1.scatter(x, y, c=colors[index], label=label, marker=marks[index])
        ax1.plot(x, y, c=colors[index], alpha=0.7, linewidth=1)

    plt.ylim(0,1)
    plt.xlim(1, 0)

    ax1.axline( [1, 1], [0,0], color="grey")
    title = title.replace(".cnf", "")

    plt.title(title)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    fig.tight_layout()
    plt.grid()

    print(file)
    plt.savefig(file)


def evaluate_folder(folder, labels):
    columns = ["p", "var", "value", "MC", "BDD len", 'n_vars', 'n_nodes', 'n_reorderings', 'dag_size', 'time']
    expr_datas = []
    for type in labels:
        stats_file = folder + "dataset_stats_" + type + "_reorder.csv"
        expr_data = ExprData(columns)
        expr_data.read_stats_file(stats_file)
        expr_datas.append(expr_data)


    plot_type = "raw"
    plot_multiple(folder, expr_datas, "efficiency", labels, "init")
    plot_multiple(folder, expr_datas, "ratio", labels, "init")
    plot_multiple(folder,expr_datas, "MC",labels, plot_type)
    plot_multiple(folder, expr_datas, "dag_size", labels, plot_type)


def count_all_backbones():
    exprs = [ "./paper_data/DatasetA/", "./paper_data/DatasetB/",
             "./paper_data/iscas/iscas89/" , "./paper_data/iscas/iscas93/","./paper_data/iscas/iscas99/",
            "./paper_data/Planning/blocks/",  "./paper_data/Planning/bomb/",  "./paper_data/Planning/coins/", "./paper_data/Planning/comm/",
              "./paper_data/Planning/emptyroom/",  "./paper_data/Planning/flip/", "./paper_data/Planning/safe/", "./paper_data/Planning/sort/", "./paper_data/Planning/uts/"]
    #, "./paper_data/Planning/comm/"]
    # exprs = ["./paper_data/DatasetA/", "./paper_data/DatasetB/",
    #          "./paper_data/iscas/iscas89/", "./paper_data/iscas/iscas89/", "./paper_data/iscas/iscas93/",
    #          "./paper_data/iscas/iscas99/",
    #          "./paper_data/Planning/blocks/", "./paper_data/Planning/bomb/",
    #          "./paper_data/Planning/sort/", "./paper_data/Planning/uts/"]

    columns = ["p", "var", "value", "MC", "BDD len", 'n_vars', 'n_nodes', 'n_reorderings', 'dag_size', 'time']
    proper_backbones = []
    for folder in exprs:
        type = "init"
        stats_file = folder + "dataset_stats_" + type + "_reorder.csv"
        expr_data = ExprData(columns)
        print(stats_file)
        expr_data.read_stats_file(stats_file)
        b = expr_data.count_proper_backbones()
        proper_backbones.extend(b)
    c = 0
    for x in proper_backbones:
        if x > 0:
            c+=1
    print(c)
def plot_init():
    # exprs = ["./paper_data/DatasetA/", "./paper_data/DatasetB/"]
    exprs = [   "./paper_data/iscas/iscas89/", "./paper_data/iscas/iscas93/", "./paper_data/iscas/iscas99/"]
    # exprs = [  "./paper_data/Planning/blocks/", "./paper_data/Planning/bomb/", "./paper_data/Planning/coins/",
    #          "./paper_data/Planning/flip/", "./paper_data/Planning/sort/",
    #          "./paper_data/Planning/uts/"]  # , "./paper_data/Planning/comm/"]

    columns = ["p", "var", "value", "MC", "BDD len", 'n_vars', 'n_nodes', 'n_reorderings', 'dag_size', 'time']

    for folder in exprs:

        type = "init"
        stats_file = folder + "dataset_stats_" + type + "_reorder.csv"
        expr_data = ExprData(columns)
        print(stats_file)
        expr_data.read_stats_file(stats_file)
        for e in expr_data.all_expr_data.keys():
            fig = plt.figure()
            d = expr_data.all_expr_data[e]
            print(len(d))
            plt.plot([i for i in range(len(d))],[x[3] for x in d])
            f = e.replace(".cnf",".initplot.png")
            print(f)
            plt.savefig(f)
            # plt.show()

def read_ratio(ratio_file):
    df = pd.read_csv(ratio_file)
    df = df.iloc[:, -3:]
    c = list(df.columns)
    print(df)
    # df[[c[1], c[1]]].sub(df[c[0]], axis=0)
    # temp = df[c[1]] - df[c[0]]
    # df = df.drop(c[0], axis=1)
    # df[c[1]]= temp
    first_half = df.iloc[20:36]
    print(df)
    first_half.plot(kind="bar")
    plt.yscale("log")
    plt.show()

def create_time_table(folders, labels):
    columns = ["p", "var", "value", "MC", "BDD len", 'n_vars', 'n_nodes', 'n_reorderings', 'dag_size', 'time']
    expr_datas = []
    f = open("./paper_data/times_table.csv", "a+")
    writer = csv.writer(f, delimiter=',')
    writer.writerow(["Expr type", "Init compilation"]+labels)
    for folder in folders:
        init_compilation = 0
        label_compilations = []
        for type in labels:
            stats_file = folder + "dataset_stats_" + type + "_reorder.csv"
            expr_data = ExprData(columns)
            expr_data.read_stats_file(stats_file)
            init_compilation += sum([v for v in expr_data.init_compilation_time.values()]) / len(expr_data.init_compilation_time)
            last_compilation = sum(list(expr_data.finish_time.values())) / len(expr_data.finish_time)
            label_compilations.append(last_compilation)
        init_compilation = init_compilation / len(labels)
        writer.writerow([folder, round(init_compilation,3)]+[round(x,3) for x in label_compilations])



if __name__ == "__main__":
    # read_ratio("paper_data/ratio_table.csv")
    # plot_init()
    # count_all_backbones()
    # exit(666)

    # write func to calculate init MC where nb backbone > 0 - use the ddmmp files to load the compilation and just count using the root
    # update stats files
    # rerun evals

    labels = ["random_1234",  "random_selection_1234",  "random_ratio_selection_1234", "static", "static_ratio", "dynamic", "dynamic_ratio"]
    # folders = ["./paper_data/DatasetA/"] #, "./paper_data/DatasetB/", "./paper_data/iscas/iscas99/"]
    # exprs = ["./paper_data/DatasetA/", "./paper_data/DatasetB/", "./paper_data/iscas/iscas99/"]
    # exprs = ["./paper_data/DatasetA/", "./paper_data/DatasetB/",
    #          "./paper_data/iscas/iscas89/", "./paper_data/iscas/iscas89/" , "./paper_data/iscas/iscas93/","./paper_data/iscas/iscas99/",
    #                   "./paper_data/Planning/blocks/",  "./paper_data/Planning/bomb/",
    #                      "./paper_data/Planning/sort/", "./paper_data/Planning/uts/",
    #          "./paper_data/Planning/safe/", "./paper_data/Planning/emptyroom/",
    #          "./paper_data/Planning/flip/", "./paper_data/Planning/ring/" ]
    #
    # for folders in exprs:

    ################ paper results ###################

    ################### avg dataset A and B
    # exprs = ["./paper_data/DatasetA/", "./paper_data/DatasetB/"]
    # outfile = "./paper_data/AB_avg_"
    # title = "Average efficiency over Dataset A and B"
    # create_average_efficiency_plot( exprs , outfile+"efficiency.png", title,  labels, 1)
    # title = "Average ratio over Dataset A and B"
    # create_average_ratio_plot(  exprs ,  outfile+"ratio.png", title, labels, 1)
    # create_time_table(exprs, labels)

    ################### avg iscas
    # exprs = ["./paper_data/iscas/iscas89/", "./paper_data/iscas/iscas89/" , "./paper_data/iscas/iscas93/","./paper_data/iscas/iscas99/"]
    # outfile = "./paper_data/iscas_avg50_"
    # title = "Average efficiency over iscas instances"
    # create_average_efficiency_plot(exprs, outfile + "efficiency.png", title, labels, 50)
    # title = "Average ratio over iscas instances"
    # create_average_ratio_plot(exprs, outfile + "ratio.png", title, labels, 50)

    ################### avg planning
    # exprs = ["./paper_data/Planning/blocks/", "./paper_data/Planning/bomb/", "./paper_data/Planning/coins/",
    #          "./paper_data/Planning/comm/",
    #          "./paper_data/Planning/emptyroom/", "./paper_data/Planning/flip/", "./paper_data/Planning/ring/",
    #          "./paper_data/Planning/safe/", "./paper_data/Planning/sort/", "./paper_data/Planning/uts/"]
    # outfile = "./paper_data/planning_avg50_"
    # title = "Average efficiency over planning instances"
    # create_average_efficiency_plot(exprs, outfile + "efficiency.png", title, labels, 50)
    # title = "Average ratio over planning instances"
    # create_average_ratio_plot(exprs, outfile + "ratio.png", title, labels, 50)

    ###################
    # exprs = ["./paper_data/DatasetA/", "./paper_data/DatasetB/",
    #          "./paper_data/iscas/iscas89/", "./paper_data/iscas/iscas93/",
    #          "./paper_data/iscas/iscas99/",
    exprs = ["./paper_data/Planning/blocks/", "./paper_data/Planning/bomb/", "./paper_data/Planning/coins/",
             "./paper_data/Planning/comm/",
             "./paper_data/Planning/emptyroom/", "./paper_data/Planning/flip/", "./paper_data/Planning/ring/",
             "./paper_data/Planning/safe/", "./paper_data/Planning/sort/", "./paper_data/Planning/uts/"]
    labels = ["random_1234", "random_selection_1234",  "random_ratio_selection_1234", "static", "static_ratio", "dynamic", "dynamic_ratio"]


    # for f in exprs:
    #     type = f.split("/")[-2]
    #     title = "Average efficiency over "+type+" instances"
    #     create_average_efficiency_plot([f], f+type+"_avg_efficiency", title, labels, 1)
    #     title = "Average ratio over "+type+" instances"
    #     create_average_ratio_plot([f], f+type+"_avg_ratio", title, labels, 1)
    #     evaluate_folder( f, labels)

    ##################
    # columns =  ["p", "var", "value", "MC", "BDD len", 'n_vars', 'n_nodes', 'n_reorderings', 'dag_size', 'time']
    # type = "dynamic_ratio"
    # stats_file = "./paper_data/Planning/uts/" + "dataset_stats_" + type + "_reorder.csv"
    # expr_data_dynamic2 = ExprData(columns)
    # expr_data_dynamic2.read_stats_file(stats_file)
    # expr_data_dynamic2.best_ratio_table_per_alg()

    ###################
    exprs = ["./paper_data/DatasetA/", "./paper_data/DatasetB/",
             "./paper_data/iscas/iscas89/", "./paper_data/iscas/iscas93/",
             "./paper_data/iscas/iscas99/",
        "./paper_data/Planning/blocks/", "./paper_data/Planning/bomb/",  "./paper_data/Planning/coins/", "./paper_data/Planning/comm/",
             "./paper_data/Planning/emptyroom/", "./paper_data/Planning/flip/", "./paper_data/Planning/ring/",
                  "./paper_data/Planning/safe/", "./paper_data/Planning/sort/", "./paper_data/Planning/uts/" ]
    labels = ["random_1234", "random_selection_1234",  "random_ratio_selection_1234", "static", "static_ratio", "dynamic", "dynamic_ratio"]
    # create_best_ratio_table("paper_data/ratio_table.csv", exprs, labels, aggregate=False)
    create_time_table(exprs, labels)
    # metric = "mc"
    # read_ratio_table("paper_data/base_ratio_table.csv", metric)
    # exprs = ["./paper_data/DatasetA/"]
    # for f in exprs:
    #     evaluate_folder(f, labels)

    ################ paper results ###################
    # look at ratio table

    # exprs = ["./paper_data/DatasetA/", "./paper_data/DatasetB/",
    #          "./paper_data/iscas/iscas89/", "./paper_data/iscas/iscas93/",
    #          "./paper_data/iscas/iscas99/",
    #          "./paper_data/Planning/blocks/", "./paper_data/Planning/bomb/", "./paper_data/Planning/coins/",
    #          "./paper_data/Planning/comm/",
    #          "./paper_data/Planning/emptyroom/", "./paper_data/Planning/flip/", "./paper_data/Planning/ring/",
    #          "./paper_data/Planning/safe/", "./paper_data/Planning/sort/", "./paper_data/Planning/uts/"]
    # create_best_ratio_tables(exprs)

    # count_all_backbones()
    # folder = "./paper_data/DatasetA/"




    # alg_types = [ "init", "random", "random_selection", "static","static_ratio", "dynamic","dynamic_ratio"]
    #
    # exprs = ["./paper_data/BayesianNetwork/","./paper_data/DatasetA/","./paper_data/DatasetB/",
    #          "./paper_data/iscas/iscas89/", "./paper_data/iscas/iscas89/" , "./paper_data/iscas/iscas93/","./paper_data/iscas/iscas99/",
    #          "./paper_data/Planning/base/", "./paper_data/Planning/blocks/",  "./paper_data/Planning/bomb/",  "./paper_data/Planning/coins/",
    #          "./paper_data/Planning/flip/", "./paper_data/Planning/sort/", "./paper_data/Planning/uts/", "./paper_data/Planning/comm/"]

    # exprs = ["./paper_data/Planning/comm/", "./paper_data/Planning/coins/"]
    # for e in exprs:
    #     evaluate_folder(e, reorder=True)