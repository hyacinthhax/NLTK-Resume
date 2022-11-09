import nltk
import re
from keras.layers import Input, LSTM, Dense
from keras.models import Model


def preprocess(file):
	# global encoder_model, decoder_model, num_decoder_tokens, num_encoder_tokens, input_features_dict, target_features_dict, reverse_target_features_dict, max_decoder_seq_length, max_encoder_seq_length
	# Defining lines as a list of each line
	file.split('\n')
	# Building empty lists to hold sentences
	input_docs = []
	target_docs = []
	# Building empty vocabulary sets
	input_tokens = set()
	target_tokens = set()
	for line in lines:

		# Input and target sentences are separated by tabs
		input_doc, target_doc = line.split('\t')

		# Appending each input sentence to input_docs
		input_docs.append(input_doc)

		# Splitting words from punctuation
		target_doc = " ".join(re.findall(r"[\w']+|[^\s\w]", target_doc))

		# Redefine target_doc below 
		# and append it to target_docs:
		target_doc = "<START> " + target_doc + " <END>"
		target_docs.append(target_doc)

		# Now we split up each sentence into words
		# and add each unique word to our vocabulary set
		for token in re.findall(r"[\w']+|[^\s\w]", input_doc):
    		# print(token)
			if token not in input_tokens:
	  			input_tokens.add(token)

		for token in target_doc.split():
			# print(token)
			if token not in target_tokens:
				target_tokens.add(token)

		# Create num_encoder_tokens and num_decoder_tokens:
		num_encoder_tokens = len(input_tokens)
		num_decoder_tokens = len(target_tokens)
		try:
			max_encoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", input_doc)) for input_doc in input_docs])
			max_decoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", target_doc)) for target_doc in target_docs])
		except ValueError:
			pass

	input_tokens = sorted(list(input_tokens))
	target_tokens = sorted(list(target_tokens))

	input_features_dict = dict([(token, i) for i, token in enumerate(input_tokens)])
	target_features_dict = dict([(token, i) for i, token in enumerate(target_tokens)])

	reverse_input_features_dict = dict((i, token) for token, i in input_features_dict.items())
	reverse_target_features_dict = dict((i, token) for token, i in target_features_dict.items())

	encoder_input_data = np.zeros((len(input_docs), max_encoder_seq_length, num_encoder_tokens),dtype='float32')
	decoder_input_data = np.zeros((len(input_docs), max_decoder_seq_length, num_decoder_tokens),dtype='float32')
	decoder_target_data = np.zeros((len(input_docs), max_decoder_seq_length, num_decoder_tokens),dtype='float32')

	encoder_inputs = Input(shape=(None, num_encoder_tokens))
	encoder_lstm = LSTM(100, return_state=True)
	encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
	encoder_states = [state_hidden, state_cell]

	decoder_inputs = Input(shape=(None, num_decoder_tokens))
	decoder_lstm = LSTM(100, return_state=True)
	decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	decoder_states = [state_hidden, state_cell]

	decoder_dense = Dense(num_decoder_tokens, activation='softmax')
	decoder_outputs = decoder_dense(decoder_outputs)
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	model.compile(optimizer='rmsprop', loss='categorical_entropy', metrics=['accuracy'])
	model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=100, epochs=100, validation_split=0.2)

	encoder_model = Model(encoder_inputs, encoder_states)
	latent_dim = 256
	decoder_state_input_hidden = Input(shape=(latent_dim))
	decoder_state_input_cell = Input(shape=(latent_dim))
	decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]

	decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
	decoder_states = [state_hidden, state_cell]
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = Model([decoder_inputs]+decoder_states_inputs, [decoder_outputs]+decoder_states)
