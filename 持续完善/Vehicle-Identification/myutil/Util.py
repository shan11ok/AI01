#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class Util(object):
    @staticmethod
    def save_dict(save_path, dict):
        with open(save_path, mode='w', encoding="utf-8") as dict_file:
            for key, val in dict.items():
                dict_file.write(f"{val}:{key}")
                dict_file.write("\n")

    @staticmethod
    def recover_dict(dict_path):
        dict = {}
        with open(dict_path, mode='r', encoding="utf-8") as file:
            for line in file.readlines():
                vals = line.rstrip("\n").split(sep=":")
                key, val = int(vals[0]), vals[1]
                dict[key] = val
        return dict

    @staticmethod
    def count_file_in_dir(path):
        '''
        遍历统计目录下所有文件的个数
        :param path:
        :return:
        '''
        import os
        count = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for file in filenames:
                count += 1
        return count

    @staticmethod
    def count_subdir_in_dir(path):
        '''
        遍历统计目录下所有文件的个数
        :param path:
        :return:
        '''
        import os
        count = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for dir in dirnames:
                count += 1
        return count