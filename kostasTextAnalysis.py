import matplotlib.pyplot as plt


def import_text_split_scenes(txt_filename):
    """
    Import play and delete delimiters and split scenes
    """
    file = open(txt_filename, 'r', encoding="latin-1")

    text = file.read()

    delimiters = [',', '.', '\n', ';', '[', '(', ']', ')']
    for d in delimiters:
        text = text.replace(d, ' ')

    word_arr = text.split(" ")
    word_arr = [v for v in word_arr if v != '']

    file = open('partB/positive-words.txt', 'r', encoding="latin-1")
    positive = file.read()

    file = open('partB/negative-words.txt', 'r', encoding="latin-1")
    negative = file.read()

    # load dictionaries
    positive = [word for word in positive.split(";;\n")[2].split("\n") if word != '']
    negative = [word for word in negative.split(";;\n")[2].split("\n") if word != '']

    # split scenes
    firstSceneI = word_arr.index('SCENE')
    scene_dict = {0: []}
    i = firstSceneI
    cur_scene = -1
    while i < len(word_arr):
        if word_arr[i] == 'SCENE':
            cur_scene += 1
            scene_dict[cur_scene] = []
        # keep only words in dictionary
        if word_arr[i].lower() in positive or word_arr[i].lower() in negative:
            scene_dict[cur_scene].append(word_arr[i].lower())
        i += 1
    return scene_dict, positive, negative


def max_count(scenes, positive, negative):
    """
    Raw count
    """
    scene_vectors = {}
    word_space = positive + negative
    n_pos = len(positive)
    n = len(word_space)
    for i in scenes.keys():
        v = [0 for i in range(n)]
        for index, word in enumerate(word_space):
            if index >= n_pos:
                v[index] = - scenes[i].count(word)
            else:
                v[index] = scenes[i].count(word)
        scene_vectors[i] = sum(v)

    plt.plot(list(scene_vectors.keys()), list(scene_vectors.values()))
    plt.show()

scenes, positive, negative = import_text_split_scenes('partB/midsummer.txt')
max_count(scenes, positive, negative)
