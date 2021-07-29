import os
import sys

# Suppress as many warnings as possible
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings('ignore')
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER

def main():
    # Download data
    if not os.path.isdir("./data"):
        data_utils.download_data("./")

    # Load model without GPU
    ws = WS("./data")
    pos = POS("./data")
    ner = NER("./data")

    # Load model with GPU
    # ws = WS("./data", disable_cuda=False)
    # pos = POS("./data", disable_cuda=False)
    # ner = NER("./data", disable_cuda=False)

    # Create custom dictionary
    word_to_weight = {
        "": 1
    }
    # dictionary = construct_dictionary(word_to_weight)
    # print(dictionary)

    # Run WS-POS-NER pipeline
    sentence_list = []
    with open("list.txt", "r") as f:
        sentence_list += f.readlines()
    f.close()
    sentence_list = [ list[:-1] for list in sentence_list ]
    word_sentence_list = ws(sentence_list, sentence_segmentation=True)
    pos_sentence_list = pos(word_sentence_list)
    entity_sentence_list = ner(word_sentence_list, pos_sentence_list)

    # Release model
    del ws
    del pos
    del ner

    # Show results
    def print_word_pos_sentence(word_sentence, pos_sentence):
        assert len(word_sentence) == len(pos_sentence)
        for word, pos in zip(word_sentence, pos_sentence):
            print(f"{word}({pos})", end="\u3000")
        print()
        return

    for i, sentence in enumerate(sentence_list):
        attr = []
        print()
        print(f"'{sentence}'")
        if f"{sentence}" == "":
            print("Error: Your subject cannot be empty.")
            continue
        for word, pos in zip(word_sentence_list[i], pos_sentence_list[i]):
            if (f"{pos}".find("N") != -1) or (f"{pos}".find("V") != -1):
                if f"{word}" not in attr:
                    attr.append(f"{word}")

        garbage_str = ['(', ')', '{', '}', '<', '>', '〈', '〉']
        for str in garbage_str:
            if str in attr:
                attr.remove(str)
        if attr == []:
            attr = f"{sentence}".split()
        print(attr)

    print()

    return

if __name__ == "__main__":
    main()
    sys.exit()
