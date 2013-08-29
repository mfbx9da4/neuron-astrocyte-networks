

1. Parallel processing by astrocyte communication
		"""
		Astrocytes in the same layer are able to communicate with each other via intermediary astrocytes. Here a layer can be considered analgous to an astrocyte subnetwork. I will call these intermediary astrocytes, astrocyte associated astrocytes (AAAs) and the other astrocytes, neuron associated astrocytes (NAAs).
		AAAs may be justified not to be connected with neurons as their neurons are not associated with this synaptic


		Counts those with correct response
		"""
		for all samples:
			* update neurons using inputs
			* update neurons using astrocytes using assocations from previous iteration
			* update  


		net.activate(sample)
		for each layer:
			for each AAA:
				count_correlation(trg == out)

		def count_correlation(output_is_correct):
			# can also try version where not based on output being correct
			# can also try version where bins activations into four types
			if output_is_correct:
				if NAAs are both positively active:
					AAA[x].pos += 1
				if NAAs are both negatively active:
					AAA[x].neg += 1
				if NAAs are not matching:
					AAA[x].mismatch += 1

		def activate(inputs):
			activate input layer:
				tanh(inputs)
			activate hidden layer:
				tanh(input_layer * full_connection)
				for each AAA:
					if AAA acitve and NAAs mismatch:
						acitvate


could just have astroyctes as part of the network and see if it learns using backprop or GA


2. Astrocytes gate plasticity
	"""
	Aastrocytes might form perpendicular associations
	Astrocytes mediate error somewhat
	"""



for each ind:
	for each AAA in layer:
		if association formed:
			if one active but not other:

				# option 1 does incrementing activation potentiate change? if so that makes sense!
				* increment activity of other
				* bp using error with combined network
				
				# option 2 does incrementing
				* increment weights of other
				* bp using error with combined network

				# option 3 does incrementing
				* increment activity of other
				* if correct increase weights to match
				
				# option 4 
				* increment weights of other
				* activate network conserve best