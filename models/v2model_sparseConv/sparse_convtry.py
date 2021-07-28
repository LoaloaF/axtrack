import torch
import sparseconvnet as scn

# Use the GPU if there is one, otherwise CPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = scn.Sequential().add(
    scn.Convolution(2, 1, 24, 4, 2, False)
# ).add(
#     scn.SubmanifoldConvolution(2, 24, 32, 3, False)
).add(
    scn.BatchNormReLU(24)
).add(
    scn.SparseToDense(2, 24)
).to(device)

# output will be 10x10
inputSpatialSize = model.input_spatial_size(torch.LongTensor([10, 10]))
inputSpatialSize = 1000,1200
print(inputSpatialSize)
input_layer = scn.InputLayer(2, inputSpatialSize)
print(input_layer)

msgs = [["X   X  XXX  X    X    XX     X       X   XX   XXX   X    XXX   ",
         " X   X  X    X    X   X  X    X       X  X  X  X  X  X    X  X  ",
         " XXXXX  XX   X    X   X  X    X   X   X  X  X  XXX   X    X   X ",
         " X   X  X    X    X   X  X     X X X X   X  X  X  X  X    X  X  ",
         " X   X  XXX  XXX  XXX  XX       X   X     XX   X  X  XXX  XXX   "],

        ["XXX              XXXXX      x   x     x  xxxxx  xxx ",
         " X  X  X   XXX       X       x   x x   x  x     x  x ",
         " XXX                X        x   xxxx  x  xxxx   xxx ",
         " X     X   XXX       X       x     x   x      x    x ",
         " X     X          XXXX   x   x     x   x  xxxx     x ",]]


# Create Nx3 and Nx1 vectors to encode the messages above:
locations = []
features = []
for batchIdx, msg in enumerate(msgs):
    for y, line in enumerate(msg):
        for x, c in enumerate(line):
            if c == 'X':
                locations.append([y, x, batchIdx])
                features.append([1])
locations = torch.LongTensor(locations)
features = torch.FloatTensor(features).to(device)

# print(locations)
# print(features)

input = input_layer([locations,features])
print(input)
exit()
print('Input SparseConvNetTensor:', input)
output = model(input)

# Output is 2x32x10x10: our minibatch has 2 samples, the network has 32 output
# feature planes, and 10x10 is the spatial size of the output.
print('Output SparseConvNetTensor:', output)