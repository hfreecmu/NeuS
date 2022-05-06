import torch
import torchvision



m = torchvision.models.mobilenet_v3_small()

print(type(m.parameters()))
o = torch.optim.SGD(m.parameters(), lr=1)

foo_1 = torchvision.models.mobilenet_v3_small()
foo_2 = torchvision.models.mobilenet_v3_small()
grads = m.state_dict()
foo_1.load_state_dict(grads)
foo_2.load_state_dict(grads)

exit()
with torch.no_grad():
    for p1, p2, p3 in zip(m.parameters(), foo_1.parameters(), foo_2.parameters()):
        new_p = p2
        for p in [p3]:
            new_p += p3
        p1.grad = new_p

import pdb
pdb.set_trace()
print(m.features[0][0].weight[0][0])
o.step()
print(m.features[0][0].weight[0][0])





