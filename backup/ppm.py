def trainNGA(nn, trndata, hiddenAstrocyteLayer, outputAstrocyteLayer):
    inputs = list(trndata['input'])
    random.shuffle(inputs)
    for inpt in trndata['input']:
        nn.activate(inpt)
        for m in range(hiddenAstrocyteLayer.astrocyte_processing_iters):
            hiddenAstrocyteLayer.update()
            outputAstrocyteLayer.update()
        hiddenAstrocyteLayer.reset()
        outputAstrocyteLayer.reset()


def associateAstrocyteLayers(nn):
    in_to_hidden, = nn.connections[nn['in']]
    hidden_to_out, = nn.connections[nn['hidden0']]
    hiddenAstrocyteLayer = AstrocyteLayer(nn['hidden0'], in_to_hidden)
    outputAstrocyteLayer = AstrocyteLayer(nn['out'], hidden_to_out)
    return hiddenAstrocyteLayer, outputAstrocyteLayer