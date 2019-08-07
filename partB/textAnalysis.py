import math
import matplotlib.pyplot as plt
import numpy as np
import re

play_characters = ['THESEUS', 'EGEUS', 'LYSANDER', 'DEMETRIUS', 'PHILOSTRATE', 'QUINCE', 'SNUG', 'BOTTOM', 'FLUTE',
                   'SNOUT', 'STARVELING', 'HIPPOLYTA', 'HERMIA', 'HELENA', 'OBERON', 'TITANIA', 'PUCK', 'PEASBLOSSOM',
                   'COBWEB', 'MUSTARDSEED', 'PYRAMUS', 'THISBE', 'WALL', 'MOONSHINE', 'LION']


def import_text(txt_filename):
    """
    Imports the text from the file provided as an argument.
    :param: txt_filename: name/location of txt to be imported
    :return: text as string
    """
    file = open(txt_filename, 'r', encoding="latin-1")
    return file.read()


def filter_tokenize(txt):
    """
    Apply filter to remove punctuation and meaningless character and tokenize text into a list of words.
    :param txt: text as string
    :return: Returns the tuple (List of words, filtered text)
    """
    # Dictionary for unwanted chars and their replacement
    filter_out_list = {',': ' ', '.': ' ', '\n': ' ', ';': '', '[': '(', ']': ')'}

    # Apply filters
    for ch in filter_out_list.keys():
        txt = txt.replace(ch, filter_out_list[ch])

    # Remove [...]
    txt = re.sub(r'\([^)]*\)', '', txt)

    # Tokenize
    word_arr = txt.split(" ")

    # Clear ''
    word_arr = [v for v in word_arr if v != '']

    return word_arr, txt


def separate_scenes(txt_arr):
    """
    Splits the list of words into sublist of words. Each sublist constituting a different scene of the play
    :param txt_arr: List of words
    :return: Dictionary of List of words
    """
    # Find start of each act
    scene_index = [index for index, word in enumerate(txt_arr) if word == 'SCENE']
    scene_index.append(len(txt_arr) - 1)

    # Create dictionary for the acts
    act_dict = {}
    for i in range(len(scene_index) - 1):
        act_dict[i] = [word.lower() for word in txt_arr[scene_index[i]:scene_index[i+1]]]

    return act_dict


def load_positive_negative():
    """
    Loads positive and negative words from txt
    :return: Two list of words, one with positive and one with negative meaning.
    """
    positive_txt = import_text('partB/positive-words.txt')
    negative_txt = import_text('partB/negative-words.txt')

    # Retrieve words from dictionaries
    positive_words = [word for word in positive_txt.split(";;\n")[2].split("\n") if word != '']
    negative_words = [word for word in negative_txt.split(";;\n")[2].split("\n") if word != '']

    return positive_words, negative_words


def filter_scene_txt(scene_dict, positive_words, negative_words):
    """
    Removes all the words except those present in the positive or negative dictionary
    :param scene_dict: Dictionary holding a list of words for each scene
    :param positive_words: List of words with positive meaning
    :param negative_words: List of words with negative meaning
    :return: Filtered dictionary of words for each scene
    """
    filtered_scene_dict = {}
    for key in scene_dict.keys():
        filtered_scene_dict[key] = [word for word in scene_dict[key] if (word in positive_words) or (word in negative_words)]

    return filtered_scene_dict


def term_frequency(scene_dict, positive_dict, negative_dict):
    """
    Creates a representation vector for each scene in scene_dict. This representation vector belongs in the space that
    the words of the given dictionary creates. Each element of this vector is given by number of occurrences of jth
    dictionary term in the document divided by the number of all terms of the document.
    :param scene_dict: Dictionary holding a list of words for each scene
    :param positive_dict: List of words with positive meaning
    :param negative_dict: List of words with negative meaning
    :return: Dictionary holding the representation vector for each scene
    """
    scene_vectors = {}
    for key in scene_dict.keys():
        current_scene = scene_dict[key]
        n = len(current_scene)  # number of terms in current scene
        scene_vectors[key] = [current_scene.count(word) / n if n > 0 else 0 for word in positive_dict] + \
                             [-current_scene.count(word) / n if n > 0 else 0 for word in negative_dict]

    return scene_vectors


def binary(scene_dict, positive_dict, negative_dict):
    """
    Creates a representation vector for each scene in scene_dict. This representation vector belongs in the space that
    the words of the given dictionary creates. Each element of this vector is (+/-)1 if the corresponding dictionary
    term exists in the document, otherwise is 0.
    :param scene_dict: Dictionary holding a list of words for each scene
    :param positive_dict: List of words with positive meaning
    :param negative_dict: List of words with negative meaning
    :return: Dictionary holding the representation vector for each scene
    """
    scene_vectors = {}
    for key in scene_dict.keys():
        current_scene = scene_dict[key]
        scene_vectors[key] = [int(word in current_scene) for word in positive_dict] + \
                             [-int(word in current_scene) for word in negative_dict]

        v_norm = math.sqrt(sum([i ** 2 for i in scene_vectors[key]]))

        scene_vectors[key] = [i / v_norm if v_norm > 0 else 0 for i in scene_vectors[key]]

    return scene_vectors


def idf(scene_dict, dictionary):
    """
    Calculates the inverse document frequency for the terms in dictionary i.e log(D/d), where D is the number of
    documents and d is the number of documents this term appears.
    :param scene_dict: Dictionary holding a list of words for each scene
    :param dictionary: List of terms with positive meaning
    :return: List inverse document frequency score for each dictionary term
    """
    number_of_docs = len(scene_dict.keys())
    idfs = []
    for word in dictionary:
        cur_idf = 0
        for key in scene_dict.keys():
            cur_idf += int(word in scene_dict[key])
        if cur_idf != 0:
            idfs.append(math.log(number_of_docs / cur_idf))
        else:
            idfs.append(0)

    return idfs


def tfidf(scene_dict, positive_dict, negative_dict):
    """
    Creates a representation vector for each scene in scene_dict. This representation vector belongs in the space that
    the words of the given dictionary creates. Each element of this vector is equal to the term frequency of the
    corresponding term multiplied by the inverse document frequency i.e log(D/d), where D is the number of documents and
    d is the number of documents this term appears.
    :param scene_dict: Dictionary holding a list of words for each scene
    :param positive_dict: List of words with positive meaning
    :param negative_dict: List of words with negative meaning
    :return: Dictionary holding the representation vector for each scene
    """
    # Get term frequency
    scene_vectors = term_frequency(scene_dict, positive_dict, negative_dict)
    # Get inverse document frequency
    idfs = idf(scene_dict, positive_dict + negative_dict)

    for key in scene_dict.keys():
        current_scene = scene_vectors[key]
        scene_vectors[key] = [tf_v * idf_v for tf_v, idf_v in zip(current_scene, idfs)]

    return scene_vectors


def L1_norm(v):
    """
    Calculates the L1-norm of the given vector, which is the sum of all elements.
    :param v: List of floats
    :return: double
    """
    return sum(v)


def Linf_norm(v):
    """
    Calculates the Linf-norm of the given vector, which is the greatest element of the vector.
    :param v: List of floats
    :return: double
    """
    return max([math.fabs(i) for i in v])


def plot_scenes(scene_dict, positive_dict, negative_dict):
    """
    Plots the sentimental measure of the scenes using different representation schemes and different decision
    strategies. More precisely the following table describes the combinations that the function will calculate.

    +-----------------------+--------+----+--------+
    | representation scheme | binary | tf | tf-idf |
    | --------------------- |        |    |        |
    |    decision method    |        |    |        |
    +-----------------------+--------+----+--------+
    |        L1 norm        |    X   |  X |    X   |
    +-----------------------+--------+----+--------+
    |       Linf norm       |        |  X |    X   |
    +-----------------------+--------+----+--------+

    :param scene_dict: Dictionary holding a list of words for each scene
    :param positive_dict: List of words with positive meaning
    :param negative_dict: List of words with negative meaning
    :return: Dictionary holding the representation vector for each scene
    """
    # binary scheme - L1 norm
    bin_scenes = binary(scene_dict, positive_dict, negative_dict)
    bin_L1 = [L1_norm(bin_scenes[key]) for key in bin_scenes.keys()]

    # tf - L1 norm & Linf norm
    tf_scenes = term_frequency(scene_dict, positive_dict, negative_dict)
    tf_L1 = [L1_norm(tf_scenes[key]) for key in tf_scenes.keys()]
    tf_Linf = [Linf_norm(tf_scenes[key]) for key in tf_scenes.keys()]

    # tfidf - L1 norm & Linf norm
    tfidf_scenes = tfidf(scene_dict, positive_dict, negative_dict)
    tfidf_L1 = [L1_norm(tfidf_scenes[key]) for key in tfidf_scenes.keys()]
    tfidf_Linf = [Linf_norm(tfidf_scenes[key]) for key in tfidf_scenes.keys()]

    # Create plot
    plt.figure()

    scene_ids = list(scene_dict.keys())
    plt.plot(scene_ids, bin_L1)
    plt.plot(scene_ids, tf_L1)
    plt.plot(scene_ids, tf_Linf)
    plt.plot(scene_ids, tfidf_L1)
    plt.plot(scene_ids, tfidf_Linf)

    plt.legend([r'binary $L_1$-norm', r'tf $L_1$-norm', r'tf $L_\infty$-norm', r'tfidf $L_1$-norm',
                r'tfidf $L_\infty$-norm'])

    plt.xticks(scene_ids)
    plt.title('Sentimental Measure per scene')
    plt.xlabel('Scene number')
    plt.ylabel('Sentimental Measure')

    plt.show()


def plot_term_weight(scene_dict, dictionary):
    """
    Illustrate the sentimental intensity of terms in the given dictionary based on their frequency in the texts, using
    idf measure.
    :param scene_dict: Dictionary holding a list of words for each scene
    :param dictionary: List of words with positive meaning
    """
    # keep only words present in both scenes and dictionary
    scene_terms = set([term for term_list in list(scene_dict.values()) for term in term_list])
    terms = list(scene_terms.intersection(set(dictionary)))
    # Calculate inverse document frequency
    idfs = idf(scene_dict, terms)

    # Create dictionary to group terms based on their idf
    unique_idf_val = sorted(list(set(idfs)))
    idf_terms_dict = {index: [] for index, val in enumerate(unique_idf_val)}
    for index, term in enumerate(terms):
        cur_idf = idfs[index]
        idf_terms_dict[unique_idf_val.index(cur_idf)].append(term)

    # Print some example terms for each idf group to gain intuition
    print('**** Terms for each idf group ****\n')
    for index, key in enumerate(idf_terms_dict.keys()):
        print('Idf score = ' + str(unique_idf_val[index]))
        print(idf_terms_dict[key][0:min(10, len(idf_terms_dict[key]))])
    print('**** End ****\n\n')

    # Plot Sentiment intensity distribution
    height = [len(idf_terms_dict[key]) for key in idf_terms_dict.keys()]
    plt.bar(unique_idf_val, height, 0.1)

    plt.xticks(unique_idf_val, rotation='vertical')
    plt.title('Sentimental Intensity Distribution')
    plt.xlabel('Inverse Document Frequency score')
    plt.ylabel('Number of Terms')

    plt.show()


def character_appearances(scene_dict, verbose):
    """
    Find the number of appearances and their words for each character for each scene.
    :param scene_dict: Dictionary holding a list of words for each scene
    :param verbose: if true print character appearances for debugging or illustration reasons
    :return A dictionary holding the number of appearances for each character for each scene
    """
    # Initialize a dictionary to hold the number of appearances for each character for each scene
    cast = {name: [] for name in play_characters}
    for character in cast.keys():
        cast[character] = [scene_dict[key].count(character.lower()) for key in scene_dict.keys()]

    if verbose:
        print(cast)

    return cast


def character_frequency_correlation(cast):
    """
    Illustrates how frequently a character speaks and how characters appearances correlate.
    :param cast: A dictionary holding the number of appearances for each character for each scene
    """
    total_talks = sum([el for val in cast.values() for el in val])  # get total no of talks in play
    freq_dict = {key: sum(cast[key]) / total_talks for key in cast.keys()}

    # Plot frequencies
    height = freq_dict.values()
    plt.bar(range(len(freq_dict.values())), height, 0.1)

    plt.xticks(range(len(freq_dict.values())), list(freq_dict.keys()), rotation='vertical')
    plt.title('Character Appearance Frequency')
    plt.xlabel('Character')
    plt.ylabel('Percentage of Appearance')
    plt.show()

    # Plot Correlation
    n = len(cast.keys())
    fig, ax = plt.subplots(figsize=(13, 8))
    corr_matrix = np.corrcoef(list(cast.values()))
    im = ax.imshow(corr_matrix)

    # Show all ticks
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    # Node labels
    ax.set_xticklabels(list(cast.keys()))
    ax.set_yticklabels(list(cast.keys()))

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    fig.colorbar(im)
    ax.set_title('Character Correlation')
    plt.show()


def character_words_per_scene(txt, positive_dict, negative_dict):
    """
    Creates a dictionary with the words of each character for each scene.
    :param txt: Filtered initial text
    :param positive_dict: List of words with positive meaning
    :param negative_dict: List of words with negative meaning
    :return: Dialog dictionary
    """
    for character in play_characters:
        txt = txt.replace(character, '@@@ ' + character)

    scenes = txt.split('SCENE')
    scenes = scenes[2:]  # offset 2 because of irrelevant text at the beginning of txt

    dialog_dict = {character: {i: [] for i in range(len(scenes))} for character in play_characters}
    for scene_id in range(0, len(scenes)):
        current_scene_dialogs = scenes[scene_id].split('@@@ ')
        for dialog in current_scene_dialogs:
            # Tokenize
            dialog_token = dialog.split(' ')
            character_name = dialog_token[0]  # Name of Character is always the 1st word
            if not(character_name in play_characters):
                continue
            dialog_dict[character_name][scene_id].append(dialog_token[1:])

    # Dictionary for unwanted chars and their replacement
    delimiters = {',': ' ', '.': ' ', '\n': ' ', ';': '', '[': '(', ']': ')'}

    # Flatten each list, keep only dictionary words and filter out unwanted chars
    for character in dialog_dict.keys():
        for scene_id in dialog_dict[character].keys():
            cur_dialogs = dialog_dict[character][scene_id]
            # Replace with filtered
            dialog_dict[character][scene_id] = [word for dialog in cur_dialogs
                                                for word in dialog
                                                if (not(word in delimiters) and
                                                    (word in positive_dict) or
                                                    (word in negative_dict))]

    return dialog_dict


def plot_character_sentiment(dialog_dict, positive_dict, negative_dict):
    """
    Plot sentimental measure for each character for each scene using tfidf representation scheme and L1-norm to make
    decision about the magnitude and sign of the sentiment.
    :param dialog_dict: Dictionary holding the dialogs of each character
    :param positive_dict: List of words with positive meaning
    :param negative_dict: List of words with negative meaning
    """
    character_groups = [['THESEUS', 'HIPPOLYTA', 'PHILOSTRATE'], ['EGEUS', 'LYSANDER', 'DEMETRIUS', 'HERMIA', 'HELENA']]

    f, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(13, 8))

    for index, character_group in enumerate(character_groups):

        for character in character_group:
            tfidf_scenes = tfidf(dialog_dict[character], positive_dict, negative_dict)
            tfidf_L1 = [L1_norm(tfidf_scenes[key]) for key in dialog_dict[character].keys()]

            scene_ids = list(dialog_dict[character].keys())
            ax[index].plot(scene_ids, tfidf_L1)

        ax[index].legend(character_group)

    plt.xticks(scene_ids)
    f.suptitle('Sentimental Measure per Character')
    plt.xlabel('Scene number')
    f.text(0, 0.5, 'Sentimental Measure', va='center', rotation='vertical')

    plt.show()


def compare_measure_multiple_plays(play_dict, positive_dict, negative_dict):
    """
    Imports text from different plays, filters and splits the play into similar way and illustrates the sentiment
    magnitude over time using the same measure scheme for all plays.
    :param play_dict: Dictionary holding the name and txt dir of plays
    :param positive_dict: List of words with positive meaning
    :param negative_dict: List of words with negative meaning
    """
    # Create plot
    plt.figure()

    max_scene_number = -1
    for key in play_dict.keys():
        # Import text
        txt_arr = import_text('partB/%s.txt' % play_dict[key])
        # Filter & Tokenize stage
        txt_arr, txt = filter_tokenize(txt_arr)
        # Separate Scenes
        scene_dict = separate_scenes(txt_arr)
        # Filter Scenes
        play_dict[key] = filter_scene_txt(scene_dict, positive_dict, negative_dict)
        # update max scene number
        max_scene_number = max(max_scene_number, len(scene_dict.keys()))

    for key in play_dict.keys():
        # tf - L1 norm
        vector_scenes = term_frequency(play_dict[key], positive_dict, negative_dict)
        measure_list = [L1_norm(vector_scenes[key]) for key in vector_scenes.keys()]

        # Plot current play sentiment measure
        plt.plot(range(len(measure_list)), measure_list)

    plt.legend(list(plays.keys()))

    plt.xticks(range(max_scene_number))
    plt.title('Term Frequency Sentimental Measure for multiple plays')
    plt.xlabel('Scene number')
    plt.ylabel('Sentimental Measure')

    plt.show()


# Import text
txt_arr = import_text('partB/midsummer.txt')
# Filter & Tokenize stage
txt_arr, txt = filter_tokenize(txt_arr)
# Separate Scenes
scene_dict = separate_scenes(txt_arr)
# Load positive & negative dictionary
positive_words, negative_words = load_positive_negative()
# Filter Scenes
filtered_scene_dict = filter_scene_txt(scene_dict, positive_words, negative_words)
# Plot scene sentiment
plot_scenes(filtered_scene_dict, positive_words, negative_words)
# Term weight
plot_term_weight(filtered_scene_dict, positive_words)
plot_term_weight(filtered_scene_dict, negative_words)
# Get Character appearances
cast = character_appearances(scene_dict, False)
# Illustrate character frequency, correlation, sentiment through play
character_frequency_correlation(cast)
# Get dialogs for each character
dialog_dict = character_words_per_scene(txt, positive_words, negative_words)
# Plot sentiment per character per scene
plot_character_sentiment(dialog_dict, positive_words, negative_words)

# *** Compare other plays ***
plays = {'Romeo and Juliet': 'romeo_juliet',
         'Hamlet': 'hamlet',
         'A Midsummer Night\'s Dream': 'midsummer'}
compare_measure_multiple_plays(plays, positive_words, negative_words)
