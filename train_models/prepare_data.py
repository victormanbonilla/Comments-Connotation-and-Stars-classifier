
class preprocessing:
    
    def __init__(self, max_vocab_length,
                 max_length):
        
        self.max_vocab_length = max_vocab_length
        self.max_length = max_length

    def preprocess_text(self, data):
        
        from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

        text_vectorizer = TextVectorization(max_tokens=self.max_vocab_length,
                                            standardize='lower_and_strip_punctuation',
                                            split='whitespace',
                                            ngrams=None,
                                            output_mode='int',
                                            output_sequence_length=self.max_length)

        text_vectorizer.adapt(data)  # Train embedding layer
        
        return text_vectorizer
