# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains the logic for loading data for all Conditional Generation tasks.
"""
from comet.train.common import *
from openprompt.data_utils.utils import InputExample
import os
import json, csv
import pandas as pd
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable

from openprompt.utils.logging import logger
from openprompt.data_utils.data_processor import DataProcessor





class ATOMICProcessor(DataProcessor):
    """
    # TODO citation

    Examples:

    .. code-block:: python

        from openprompt.data_utils.conditional_generation_dataset import PROCESSORS

        base_path = "datasets/CondGen"

        dataset_name = "webnlg_2017"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        train_dataset = processor.get_train_examples(dataset_path)
        valid_dataset = processor.get_train_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert len(train_dataset) == 18025
        assert len(valid_dataset) == 18025
        assert len(test_dataset) == 4928
        assert test_dataset[0].text_a == " | Abilene_Regional_Airport : cityServed : Abilene,_Texas"
        assert test_dataset[0].text_b == ""
        assert test_dataset[0].tgt_text == "Abilene, Texas is served by the Abilene regional airport."
    """

    def __init__(self):
        super().__init__()
        self.labels = None
        self.data = {}

    #def get_manual_template(self):
    #    return self.manual_template
    def get_val_data(self):
        print(self.data.keys())
        return self.data["dev"]
    def get_examples(self, data_dir: str, split: str) -> List[InputExample]:
        examples = []
        path = os.path.join(data_dir, "{}.tsv".format(split))
        split_df = pd.read_table(path)
        print(split)
        method = "sup"
        lang = "en"
        wrap = False
        frozen = False
        ignore_blanks = False
        include = ""
        exclude = ""
        num_samples = 2000
        qtemp, anstemp = create_templates(method, wrap, frozen)
        qtemp = "{event}"
        anstemp = "{resp}"
        print("qtemp: ", qtemp)
        print("ans temp: ", anstemp)
        include, exclude = filter_inputs(include, exclude, lang)

        (data, 
         atomic_flattened,
         num_records
        )= fill_data(split_df, split,
                            qtemp, anstemp,
                            num_samples, 
                            ignore_blanks,
                            include,
                            exclude)
        i = 0
        self.data[split] = data
        print("num records ", split, num_records)
        for src, tgt in atomic_flattened:
            example = InputExample(guid=str(i), text_a=src, tgt_text=tgt)
            examples.append(example)
            i += 1
        return examples

PROCESSORS = {
    "atomic": ATOMICProcessor,
    # "e2e": E2eProcessor,
    # "dart" : DartProcessor,
}
