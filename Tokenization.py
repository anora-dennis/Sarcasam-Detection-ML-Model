from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]

tokenizer = Tokenizer(num_words = 100, oov_token = '<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

test_sequences = tokenizer.texts_to_sequences(test_data)
sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences, padding="post", maxlen=5)
print(padded)
print(word_index)
print(sequences)
print(test_sequences)