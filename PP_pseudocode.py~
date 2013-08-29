"""PP model"""
for each grand_iteration in max_iterations:
	for each individual in population:
		for each sample in training_pattern:
			for each minor_iteration in astrocyte_processing_iterations:
				for each layer in neural_net:
					neuron_activations = activateLayer()
					for each astro, each neuron in layer:
						astro.counter = updateCounter(neuron.activation)
						astro.activation = updateActivation(astro.counter)
						astro.performActions()
	population = genetic_algorithm(population)
"""update activations but monitor neuron activations"""
for each grand_iteration in max_iterations:
	for each individual in population:
		for each sample in training_pattern:
			for each layer in neuron_astrocyte_net:
				neurons.activations = activateNeurons()
				NAAs.activations = activateNAAs()
				for each AAA in AAAs:
					if AAA.assocation_formed and NAAs mismatch:
						match_astrocyte_activations(NAA1, NAA2)
			evalution_is_correct = neuron_astrocyte_net.output == sample.target
			for each layer in neuron_astrocyte_net:
				for each AAA in AAAs:
					if evalution_is_correct:
						AAA.updateActivation(NAA1, NAA2)
		for each AAA in AAAs:
			AAA.updateAssocications(AAA.pos, AAA.neg, AAA.mismatch)
		individual.assignFitness()
	population = genetic_algorithm(population)
"""
Keep it simple and justifyable:
	* assign random weights
	* monitor weights sum like neuron
		- do i tanh? do i tanh after all samples?
		- do i average over all samples?
	* update weights or activation?
		- weights:
			+ can divide by inputs
			+ do i need a ga? will this learn?
		- input/activation:
			+ more bio accuracy
			+ 
	* ga:
		- mutate configurations?
		- need to maintain:
			+ configurations?
			+ 

"""
for each grand_iteration in max_iterations:
	for each individual in population:
		for each sample in training_pattern:
			for each layer in neuron_astrocyte_net:
				neurons.activations = activateNeurons()
				NAAs.activations = activateNAAs()
				for each AAA in AAAs:
					if AAA.assocation_formed and NAAs mismatch:
						match_astrocyte_activations(NAA1, NAA2)
			evalution_is_correct = neuron_astrocyte_net.output == sample.target
			for each layer in neuron_astrocyte_net:
				for each AAA in AAAs:
					if evalution_is_correct:
						AAA.updateActivation(NAA1, NAA2)
		for each AAA in AAAs:
			AAA.updateAssocications(AAA.pos, AAA.neg, AAA.mismatch)
		individual.assignFitness()
	population = genetic_algorithm(population)

def match_neuron_activations(NAA1, NAA2):
	if NAA1 == NAA2:
		return 'no update'
	if self.assocation_formed = 'pos':
		if NAA1 == 'pos':
			NAA2 = 'pos'
			NAA2.performActions(nn.ah[j2] == nn.ah[j1])
		elif NAA1 == 'pos':
			NAA2 = 'pos'

make so that set_NAA1 can only be [-1, 0, 1]
