#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@author: Gong
"""

import numpy as np

DEFAULT_STR = 'NaN'
DEFAULT_VAL = -1.0


def classify(d):
    return max(d, key=d.get) if len(d) > 0 else DEFAULT_STR


def format_percent(score):
    return "({}%)".format(int(score * 100))


class Person:

    def __init__(self, idx, start):
        self.id = idx
        self.start = start
        self.end = DEFAULT_VAL
        self.age = DEFAULT_STR
        self.gender = DEFAULT_STR
        self.height = DEFAULT_VAL
        self.latest_age = DEFAULT_STR
        self.latest_age_score = DEFAULT_VAL
        self.latest_gender = DEFAULT_STR
        self.latest_gender_score = DEFAULT_VAL
        self.latest_height = DEFAULT_VAL
        self.age_dict = {}
        self.gender_dict = {}
        self.height_list = []

    def get_label(self):
        height = self.latest_height / 100.0
        gender_label = DEFAULT_STR
        if self.latest_gender != DEFAULT_STR:
            gender_label = "%s %s" % (self.latest_gender, format_percent(self.latest_gender_score))
        age_label = DEFAULT_STR
        if self.latest_age != DEFAULT_STR:
            age_label = "%s %s" % (self.latest_age, format_percent(self.latest_age_score))
        return ["%s" % self.id,
                gender_label,
                age_label,
                "%.2f M" % height, ]

    def get_info(self):
        self.age = classify(self.age_dict)
        self.gender = classify(self.gender_dict)
        self.height = np.mean(self.height_list)
        return [str(self.id), str(self.start), str(self.end), self.age, self.gender, str(self.height)]

    def __str__(self):
        return ",".join(self.get_info())
